"""Run the Monte Carlo option pricing convergence example via URSA ExecutionAgent.

This is the *human-friendly* runner.

Behavior
--------
- Non-interactive.
- Terminal UX:
  - If running on a TTY: prints a readable progress narrative (Rich if available).
  - If NOT running on a TTY: defaults to quiet to avoid control characters/noise.
- `--quiet` or `URSA_QUIET=1` forces *no* terminal output.
- Optional file logging via `--log-file` (default: <output-dir>/run.log).
- All artifacts are written under `--output-dir` (default: ./outputs).

Usage
-----
  python run_example.py
  python run_example.py --quiet
  URSA_QUIET=1 python run_example.py
  python run_example.py --output-dir outputs_alt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from preflight import run_preflight
from ux import UX, add_ux_args, resolve_ux_config, setup_file_logging

from ursa.agents.execution_agent import ExecutionAgent
from ursa.util import Checkpointer

# --- Ensure we import the *local* URSA package (ursa/src/ursa) when running from the repo.
# This example directory may be nested under ./ursa/examples/..., so we search upwards.
_HERE = Path(__file__).resolve().parent
_LOCAL_URSA_SRC: Path | None = None
for parent in [_HERE, *_HERE.parents]:
    candidate = parent / "ursa" / "src" / "ursa"
    if candidate.exists():
        _LOCAL_URSA_SRC = parent / "ursa" / "src"
        break
if _LOCAL_URSA_SRC is not None:
    sys.path.insert(0, str(_LOCAL_URSA_SRC))
# ---


def _json_default(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Best-effort, version-tolerant serialization of LangChain messages."""
    try:
        if hasattr(msg, "model_dump"):
            d = msg.model_dump()
            if isinstance(d, dict):
                return d
        if hasattr(msg, "dict"):
            d = msg.dict()
            if isinstance(d, dict):
                return d
    except Exception:
        pass

    out: dict[str, Any] = {"type": msg.__class__.__name__}
    for key in (
        "content",
        "text",
        "name",
        "id",
        "tool_calls",
        "additional_kwargs",
    ):
        if hasattr(msg, key):
            try:
                out[key] = getattr(msg, key)
            except Exception:
                out[key] = "<unavailable>"
    out["repr"] = repr(msg)
    return out


def _is_under(base: Path, p: Path) -> bool:
    """Return True if p is inside base (after resolve())."""
    try:
        p.relative_to(base)
        return True
    except Exception:
        return False


def _resolve_under_here(
    here: Path, user_path: str | None, *, default_rel: str
) -> Path:
    """Resolve a user-supplied path, requiring that it stays under `here`."""

    raw = (
        default_rel
        if (user_path is None or str(user_path).strip() == "")
        else str(user_path)
    )
    p = Path(raw)
    if not p.is_absolute():
        p = here / p
    p = p.resolve()

    if not _is_under(here.resolve(), p):
        raise ValueError(
            f"Refusing to write outside the example directory: {p}"
        )

    return p


def _verify_outputs(
    *, here: Path, outputs_dir: Path, preflight_details: dict[str, Any]
) -> None:
    """Verify required artifacts.

    Robustness principle:
    - results.csv and report.md are always required.
    - plot.png is optional even if plotting is available.
    """

    required = [outputs_dir / "results.csv", outputs_dir / "report.md"]

    plotting_enabled = bool(
        ((preflight_details or {}).get("optional") or {}).get(
            "plotting_enabled"
        )
    )

    missing_or_empty: list[str] = []
    for p in required:
        try:
            if (not p.exists()) or p.stat().st_size <= 0:
                missing_or_empty.append(str(p.relative_to(here)))
        except Exception:
            missing_or_empty.append(str(p.relative_to(here)))

    plot_path = outputs_dir / "plot.png"
    plot_exists = plot_path.exists()
    plot_nonempty = False
    if plot_exists:
        try:
            plot_nonempty = plot_path.stat().st_size > 0
        except Exception:
            plot_nonempty = False

    verify_payload = {
        "ok": len(missing_or_empty) == 0,
        "missing_or_empty": missing_or_empty,
        "plotting_enabled": plotting_enabled,
        "plot_exists": plot_exists,
        "plot_nonempty": plot_nonempty,
        "outputs_dir": str(outputs_dir.relative_to(here)),
    }
    (outputs_dir / "verify.json").write_text(
        json.dumps(verify_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if missing_or_empty:
        raise RuntimeError(
            "Missing or empty required artifacts: "
            + ", ".join(missing_or_empty)
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the URSA ExecutionAgent Monte Carlo convergence example",
    )
    add_ux_args(p)
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory (relative to this example) where artifacts/logs are written (default: outputs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent

    args = _build_arg_parser().parse_args(argv)

    # Enforce: write only within this example directory.
    outputs_dir = _resolve_under_here(
        here, args.output_dir, default_rel="outputs"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Default log file lives under outputs_dir.
    default_log_file_rel = str(Path(outputs_dir.relative_to(here)) / "run.log")

    cfg = resolve_ux_config(
        args=args, default_log_file=Path(default_log_file_rel)
    )
    ux = UX(cfg)

    # Resolve/validate log file path under here as well.
    log_path = _resolve_under_here(
        here,
        str(cfg.log_file) if cfg.log_file is not None else None,
        default_rel=default_log_file_rel,
    )

    # Truncate log for readability.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    logger = setup_file_logging(log_file=log_path)

    # Ensure *absolute* silence when quiet:
    # - if we have a log file, redirect stdout/stderr into it
    # - otherwise redirect to os.devnull
    redirect_ctx = contextlib.nullcontext()
    log_fh = None
    if cfg.quiet:
        redirect_ctx = contextlib.ExitStack()
        try:
            log_fh = log_path.open("a", encoding="utf-8")
        except Exception:
            log_fh = open(os.devnull, "w", encoding="utf-8")
        redirect_ctx.enter_context(contextlib.redirect_stdout(log_fh))
        redirect_ctx.enter_context(contextlib.redirect_stderr(log_fh))

    with redirect_ctx:
        try:
            os.chdir(here)

            if not cfg.quiet:
                ux.panel(
                    title="URSA ExecutionAgent example",
                    body=(
                        "Monte Carlo option convergence study: the agent writes code, runs it, and generates artifacts.\n"
                        f"Artifacts: [bold]{outputs_dir.relative_to(here)}/[/bold]"
                    ),
                )
                ux.print(f"Log file: [bold]{log_path.relative_to(here)}[/bold]")

            logger.info("run start")
            logger.info("outputs_dir=%s", str(outputs_dir))

            with ux.status("Preflight checks..."):
                preflight = run_preflight(outputs_dir=outputs_dir, strict=True)

            plotting_enabled = bool(
                (preflight.details.get("optional") or {}).get(
                    "plotting_enabled"
                )
            )
            logger.info(
                "preflight ok=%s plotting_enabled=%s",
                preflight.ok,
                plotting_enabled,
            )
            if not cfg.quiet:
                ux.print(f"Preflight OK. plotting_enabled={plotting_enabled}")

            # Remove artifacts that must be generated by this run so stale files cannot mask failures.
            with ux.status("Clearing stale artifacts..."):
                for rel in (
                    "results.csv",
                    "report.md",
                    "plot.png",
                    "agent_response.md",
                    "agent_state.json",
                    "verify.json",
                    "validation.json",
                    "run_error.json",
                ):
                    p = outputs_dir / rel
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

            with ux.status("Loading prompt..."):
                prompt_text = (here / "prompt.md").read_text(encoding="utf-8")
                prompt_text += (
                    "\n\n[Preflight hint] plotting_enabled="
                    + str(plotting_enabled)
                    + "\n"
                )
                prompt_text += (
                    "[Runner hint] outputs_dir="
                    + str(outputs_dir.relative_to(here))
                    + "\n"
                )

            model_name = os.getenv("URSA_MODEL", "openai:gpt-5-mini")
            with ux.status(f"Initializing model ({model_name})..."):
                model = init_chat_model(
                    model=model_name,
                    temperature=0,
                    # Writing code via tool calls can be token-heavy; leave headroom.
                    max_completion_tokens=6000,
                )

            # Start each run from a clean slate so prior runs don't bloat context via persisted checkpointer.
            with ux.status("Preparing checkpointer..."):
                db_dir = outputs_dir / "db"
                db_dir.mkdir(parents=True, exist_ok=True)
                for suffix in ("", "-shm", "-wal"):
                    p = db_dir / ("checkpointer.db" + suffix)
                    if p.exists():
                        try:
                            p.unlink()
                        except Exception:
                            pass

                db_dir_rel = str(
                    (outputs_dir.relative_to(here) / "db").as_posix()
                )

                checkpointer = Checkpointer.from_workspace(
                    here,
                    db_dir=db_dir_rel,
                    db_name="checkpointer.db",
                )

            executor = ExecutionAgent(
                llm=model,
                workspace=here,
                checkpointer=checkpointer,
                thread_id="execution_agent_monte_carlo",
                enable_metrics=False,
                autosave_metrics=False,
                log_state=False,
            )

            with ux.status("Invoking ExecutionAgent (LLM + tools)..."):
                result = executor.invoke(
                    {
                        "messages": [HumanMessage(content=prompt_text)],
                        "workspace": str(here),
                    },
                    recursion_limit=400,
                )

            with ux.status("Writing agent debug artifacts..."):
                state_dump = {
                    "keys": sorted(result.keys()),
                    "code_files": result.get("code_files", []),
                    "messages": [
                        _serialize_message(m)
                        for m in (result.get("messages") or [])
                    ],
                }
                (outputs_dir / "agent_state.json").write_text(
                    json.dumps(
                        state_dump,
                        indent=2,
                        ensure_ascii=False,
                        default=_json_default,
                    ),
                    encoding="utf-8",
                )

                formatted = (executor.format_result(result) or "").strip()
                if not formatted:
                    formatted = "(Agent returned an empty recap. See agent_state.json and run.log in the outputs dir.)"

                (outputs_dir / "agent_response.md").write_text(
                    formatted + "\n", encoding="utf-8"
                )
                logger.info(
                    "wrote %s",
                    str((outputs_dir / "agent_response.md").relative_to(here)),
                )

            with ux.status("Verifying outputs exist..."):
                _verify_outputs(
                    here=here,
                    outputs_dir=outputs_dir,
                    preflight_details=preflight.details,
                )
            logger.info("verification ok")

            logger.info("run end")

            if not cfg.quiet:
                ux.print("Done.")
                ux.print(
                    f"Next: python validate_outputs.py (artifacts are in {outputs_dir.relative_to(here)}/)"
                )

            return 0

        except Exception as e:
            err = {
                "error_type": type(e).__name__,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            (outputs_dir / "run_error.json").write_text(
                json.dumps(err, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            try:
                logger.exception("run failed")
            except Exception:
                pass

            if not cfg.quiet:
                ux.panel(
                    title="Run failed",
                    body=(
                        f"{type(e).__name__}: {e}\n\n"
                        f"See {outputs_dir.relative_to(here)}/run_error.json and {log_path.relative_to(here)} for details."
                    ),
                )
            return 1

        finally:
            try:
                if log_fh is not None:
                    log_fh.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
