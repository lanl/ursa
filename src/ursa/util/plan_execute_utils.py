"""
Shared utilities for plan_execute workflows.

This module contains common functionality used by both single-repo and multi-repo
plan/execute workflows to reduce duplication and improve maintainability.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import yaml
from langchain.chat_models import init_chat_model
from rich import get_console
from rich.panel import Panel
from rich.text import Text

console = get_console()


# ============================================================================
# YAML Configuration Loading
# ============================================================================


def load_yaml_config(path: str) -> NS:
    """Load a YAML config file and return as a SimpleNamespace."""
    try:
        with open(path, encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
            if not isinstance(raw_cfg, dict):
                raise ValueError("Top-level YAML must be a mapping/object.")
            return NS(**raw_cfg)
    except FileNotFoundError:
        print(f"Config file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"Failed to load config {path}: {exc}", file=sys.stderr)
        sys.exit(2)


# ============================================================================
# Dictionary Merging
# ============================================================================


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return a new dict.
    - dict + dict => deep merge
    - otherwise => override wins
    """
    base = dict(base or {})
    override = dict(override or {})
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


# ============================================================================
# Plan Hashing
# ============================================================================


def hash_plan(plan_steps: list | tuple) -> str:
    """Generate a stable hash of plan steps for change detection."""
    serial = json.dumps(
        [
            step.model_dump() if hasattr(step, "model_dump") else step
            for step in plan_steps
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


# ============================================================================
# Secret Masking for Logging
# ============================================================================

_SECRET_KEY_SUBSTRS = (
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "secret",
    "password",
    "bearer",
)


def looks_like_secret_key(name: str) -> bool:
    """Check if a parameter name looks like it contains sensitive data."""
    n = name.lower()
    return any(s in n for s in _SECRET_KEY_SUBSTRS)


def mask_secret(value: str, keep_start: int = 6, keep_end: int = 4) -> str:
    """
    Mask a secret-like string, keeping only the beginning and end.
    Example: sk-proj-abc123456789xyz -> sk-proj-...9xyz
    """
    if not isinstance(value, str):
        return value
    if len(value) <= keep_start + keep_end + 3:
        return "..."
    return f"{value[:keep_start]}...{value[-keep_end:]}"


def sanitize_for_logging(obj: Any) -> Any:
    """Recursively sanitize secrets from config objects for safe logging."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if looks_like_secret_key(str(k)):
                out[k] = mask_secret(v) if isinstance(v, str) else "..."
            else:
                out[k] = sanitize_for_logging(v)
        return out
    if isinstance(obj, list):
        return [sanitize_for_logging(v) for v in obj]
    return obj


# ============================================================================
# Model Resolution & LLM Setup
# ============================================================================


def resolve_llm_kwargs_for_agent(
    models_cfg: dict | None, agent_name: str | None
) -> dict:
    """
    Given the YAML `models:` dict, compute merged kwargs for init_chat_model(...)
    for a specific agent ('planner' or 'executor').

    Merge order (later wins):
      1) {} (empty)
      2) models.defaults.params (optional)
      3) models.profiles[defaults.profile] (optional)
      4) models.agents[agent_name].profile (optional; merges that profile on top)
      5) models.agents[agent_name].params (optional)
    """
    models_cfg = models_cfg or {}
    profiles = models_cfg.get("profiles") or {}
    defaults = models_cfg.get("defaults") or {}
    agents = models_cfg.get("agents") or {}

    # Start with global defaults
    merged = {}
    merged = deep_merge_dicts(merged, defaults.get("params") or {})

    # Apply default profile
    default_profile_name = defaults.get("profile")
    if default_profile_name and default_profile_name in profiles:
        merged = deep_merge_dicts(merged, profiles[default_profile_name])

    # Apply agent-specific profile + params
    if agent_name and isinstance(agents, dict) and agent_name in agents:
        agent_cfg = agents[agent_name]
        agent_profile = agent_cfg.get("profile")
        if agent_profile and agent_profile in profiles:
            merged = deep_merge_dicts(merged, profiles[agent_profile])
        merged = deep_merge_dicts(merged, agent_cfg.get("params") or {})

    return merged


def resolve_model_choice(model_choice: str, models_cfg: dict):
    """
    Accepts strings like 'openai:gpt-5.2' or 'my_endpoint:openai/gpt-oss-120b'.
    Looks up per-provider settings from cfg.models.providers.

    Returns: (model_provider, pure_model, provider_extra_kwargs_for_init)
    """
    if ":" in model_choice:
        alias, pure_model = model_choice.split(":", 1)
    else:
        alias, pure_model = model_choice, model_choice

    providers = (models_cfg or {}).get("providers", {})
    prov = providers.get(alias, {})

    # Which LangChain integration to use (e.g. "openai", "mistral", etc.)
    model_provider = prov.get("model_provider", alias)

    # auth: prefer env var; optionally load via function if configured
    api_key = None
    if prov.get("api_key_env"):
        api_key = os.getenv(prov["api_key_env"])
    if not api_key and prov.get("token_loader"):
        # Dynamic token loading (omitted for brevity; can import if needed)
        pass

    provider_extra = {}
    if prov.get("base_url"):
        provider_extra["base_url"] = prov["base_url"]
    if api_key:
        provider_extra["api_key"] = api_key

    return model_provider, pure_model, provider_extra


def print_llm_init_banner(
    agent_name: str | None,
    provider: str,
    model_name: str,
    provider_extra: dict,
    llm_kwargs: dict,
    model_obj=None,
) -> None:
    """Print a Rich panel showing LLM initialization details."""
    who = agent_name or "llm"

    safe_provider_extra = sanitize_for_logging(provider_extra or {})
    safe_llm_kwargs = sanitize_for_logging(llm_kwargs or {})

    console.print(
        Panel.fit(
            Text.from_markup(
                f"[bold cyan]LLM init ({who})[/]\n"
                f"[bold]provider[/]: {provider}\n"
                f"[bold]model[/]: {model_name}\n\n"
                f"[bold]provider kwargs[/]: {json.dumps(safe_provider_extra, indent=2)}\n\n"
                f"[bold]llm kwargs (merged)[/]: {json.dumps(safe_llm_kwargs, indent=2)}"
            ),
            border_style="cyan",
        )
    )

    # Best-effort readback from the LangChain model object
    if model_obj is None:
        return

    readback = {}
    for attr in (
        "model_name",
        "model",
        "reasoning",
        "temperature",
        "max_completion_tokens",
        "max_tokens",
    ):
        if hasattr(model_obj, attr):
            val = getattr(model_obj, attr, None)
            if val is not None:
                readback[attr] = val

    for attr in ("model_kwargs", "kwargs"):
        if hasattr(model_obj, attr):
            val = getattr(model_obj, attr, {})
            if isinstance(val, dict) and val:
                readback[attr] = val

    if readback:
        safe_readback = sanitize_for_logging(readback)
        console.print(
            Panel.fit(
                Text.from_markup(
                    f"[dim]Model object readback:[/]\n{json.dumps(safe_readback, indent=2)}"
                ),
                border_style="dim",
            )
        )

    # Attempt a minimal test call
    effort = None
    try:
        from langchain_core.messages import HumanMessage as _HM

        effort = model_obj.invoke([_HM(content="test")])
    except Exception:
        pass

    if effort:
        console.print("[dim]✓ Test invocation succeeded[/dim]")


def setup_llm(
    model_choice: str,
    models_cfg: dict | None = None,
    agent_name: str | None = None,
):
    """
    Build a LangChain chat model via init_chat_model(...), optionally applying
    YAML-driven params from models.profiles, models.defaults, models.agents.
    """
    models_cfg = models_cfg or {}

    provider, pure_model, provider_extra = resolve_model_choice(
        model_choice, models_cfg
    )

    # Hardcoded defaults for backward compatibility
    base_llm_kwargs = {
        "max_completion_tokens": 10000,
        "max_retries": 2,
    }

    # YAML-driven kwargs (safe if absent)
    yaml_llm_kwargs = resolve_llm_kwargs_for_agent(models_cfg, agent_name)

    # Merge: base defaults < YAML overrides
    llm_kwargs = deep_merge_dicts(base_llm_kwargs, yaml_llm_kwargs)

    # Initialize
    model = init_chat_model(
        model=pure_model,
        model_provider=provider,
        **llm_kwargs,
        **(provider_extra or {}),
    )

    # Print confirmation
    print_llm_init_banner(
        agent_name=agent_name,
        provider=provider,
        model_name=pure_model,
        provider_extra=provider_extra,
        llm_kwargs=llm_kwargs,
        model_obj=model,
    )

    return model


# ============================================================================
# Workspace Setup
# ============================================================================


def setup_workspace(
    user_specified_workspace: str | None,
    project: str = "run",
    model_name: str = "openai:gpt-5-mini",
) -> str:
    """
    Set up a workspace directory for a plan/execute run.
    Returns the workspace path as a string.
    """
    if user_specified_workspace is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        workspace = f"{project}_{timestamp}"
    else:
        workspace = user_specified_workspace

    Path(workspace).mkdir(parents=True, exist_ok=True)

    # Choose a fun emoji based on the model family
    if model_name.startswith("openai"):
        model_emoji = "🤖"
    elif "llama" in model_name.lower():
        model_emoji = "🦙"
    else:
        model_emoji = "🧠"

    # Print the panel with model info
    console.print(
        Panel.fit(
            f":rocket:  [bold bright_blue]{workspace}[/bold bright_blue]  :rocket:\n"
            f"{model_emoji}  [bold cyan]{model_name}[/bold cyan]",
            title="[bold green]ACTIVE WORKSPACE[/bold green]",
            border_style="bright_magenta",
            padding=(1, 4),
        )
    )

    return workspace


# ============================================================================
# Interactive Input with Timeout
# ============================================================================


def timed_input_with_countdown(prompt: str, timeout: int) -> str | None:
    """
    Read a line with a per-second countdown. Returns:
      - the user's input (str) if provided,
      - None if timeout expires,
      - None if non-interactive or timeout<=0.
    """
    try:
        is_tty = sys.stdin.isatty()
    except Exception:
        is_tty = False

    if not is_tty:
        # Non-interactive: default immediately
        return None
    if timeout <= 0:
        # Timeout disabled: default immediately
        return None

    deadline = time.time() + timeout
    print(prompt, end="", flush=True)

    try:
        while True:
            remaining = int(deadline - time.time())
            if remaining <= 0:
                print()
                return None

            # Poll stdin with a 1-second timeout
            ready, _, _ = select.select([sys.stdin], [], [], 1.0)
            if ready:
                line = sys.stdin.readline()
                return line.rstrip("\n") if line else None

            # Update countdown display
            print(f"\r{prompt}({remaining}s) ", end="", flush=True)

    except Exception:
        print()
        return None


# ============================================================================
# Checkpoint Snapshotting (SQLite)
# ============================================================================


def snapshot_sqlite_db(src_path: Path, dst_path: Path) -> None:
    """
    Make a consistent copy of the SQLite database at src_path into dst_path,
    using the sqlite3 backup API. Safe with WAL; no need to copy -wal/-shm.
    """
    if not src_path.exists():
        raise FileNotFoundError(f"Source database not found: {src_path}")

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    src_uri = f"file:{Path(src_path).resolve().as_posix()}?mode=ro"
    src = dst = None
    try:
        src = sqlite3.connect(src_uri, uri=True)
        dst = sqlite3.connect(str(dst_path))
        with dst:
            src.backup(dst)
    finally:
        try:
            if src:
                src.close()
        except Exception:
            pass
        try:
            if dst:
                dst.close()
        except Exception:
            pass


# ============================================================================
# Formatted Elapsed Time
# ============================================================================


def fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as compact h:mm:ss or m:ss."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
