from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ursa.runtime import AgentManager, AgentSpec
from ursa.util.http import inject_truststore_into_ssl

logger = logging.getLogger(__name__)

_UNSET = object()


def _normalize_model(model: str, model_provider: Any = None) -> str:
    # Examples use "openai:gpt-5.4-mini".
    #
    # The dashboard targets an OpenAI-compatible client stack by default, so a
    # bare model name (e.g. "llama3.1" or a custom endpoint model) is prefixed
    # with "openai:" so LangChain can resolve a provider. However, we must NOT
    # manufacture a provider prefix when the caller has already specified the
    # provider explicitly via a `model_provider` kwarg: doing so would send a
    # doubly-qualified name like "openai:llama3.1" to the endpoint (LangChain's
    # init_chat_model only strips a provider prefix when model_provider is
    # unset), which the provider then rejects as an unknown model.
    #
    # Behavior:
    #   - already-prefixed model (contains ":")            -> unchanged
    #   - bare model + explicit model_provider given       -> unchanged (bare)
    #   - bare model + no model_provider                   -> prepend "openai:"
    #
    # Note: a user who supplies BOTH a provider-prefixed model (e.g.
    # "openai:gpt-4o") AND a model_provider is genuinely double-specifying; we
    # intentionally leave that as-is and let LangChain surface the error.
    if ":" in model:
        return model
    if model_provider is not None and str(model_provider).strip() != "":
        return model
    return f"openai:{model}"


def _api_key_from_config(
    config: dict[str, Any],
    *,
    label: str,
    override: str | None | object = _UNSET,
) -> str | None:
    from collections.abc import Mapping

    from ursa.util.secrets import SecretReference

    api_key = config.get("api_key")
    # A safe ``{env|keyring: ...}`` reference is an indirection, not a literal
    # secret; only reject literal (string) keys embedded in the config.
    api_key_reference: SecretReference | None = None
    if api_key is not None:
        if isinstance(api_key, Mapping):
            api_key_reference = SecretReference.maybe_validate(dict(api_key))
        if not isinstance(api_key_reference, SecretReference):
            raise ValueError(
                f"Literal {label} API keys are not accepted in worker config"
            )
    if override is not _UNSET:
        return str(override) if override else None

    env_name = str(config.get("api_key_env") or "").strip()
    if not env_name and api_key_reference is not None:
        assert api_key_reference.keyring is None, (
            "Keyring not currently supported here. Use API_KEY_ENV for setting the key."
        )
        env_name = str(api_key_reference.env or "").strip()
    if not env_name:
        return None
    env_val = os.environ.get(env_name)
    if not env_val:
        raise ValueError(
            f"{label} api_key_env '{env_name}' is not set in the worker environment"
        )
    return env_val


def _model_kwargs(
    config: dict[str, Any],
    model: str,
    *,
    label: str,
    api_key_override: str | None | object,
) -> dict[str, Any]:
    """Build provider options shared by chat and embedding initialization."""
    raw_base_url = config.get("base_url")
    base_url = str(raw_base_url).strip() if raw_base_url is not None else None
    api_key = _api_key_from_config(
        config, label=label, override=api_key_override
    )
    model_kwargs = config.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        raise ValueError(f"{label.lower()}.model_kwargs must be a JSON object")
    kwargs = {
        **model_kwargs,
        "model": _normalize_model(model, model_kwargs.get("model_provider")),
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _init_llm(
    llm_cfg: dict[str, Any],
    *,
    api_key_override: str | None | object = _UNSET,
):
    # Avoid importing langchain unless actually executing.
    from langchain.chat_models import init_chat_model

    from ursa.util.model_metadata import load_model_context

    kwargs = _model_kwargs(
        llm_cfg,
        str(llm_cfg.get("model") or "openai:gpt-5.4-mini"),
        label="LLM",
        api_key_override=api_key_override,
    )
    llm = init_chat_model(**kwargs)
    load_model_context(llm, explicit_profile="profile" in kwargs)
    return llm


def _init_embedding(
    embedding_cfg: dict[str, Any],
    *,
    api_key_override: str | None | object = _UNSET,
):
    """Initialize an embedding model from dashboard settings.

    Returns None when no embedding model is configured. Non-secret config is
    snapshotted into the run record; the run manager delivers stored or
    environment-backed secrets through the worker's one-time stdin channel.
    """
    model = str(embedding_cfg.get("model") or "").strip()
    if not model or model.lower() in {"none", "disabled"}:
        return None

    from langchain.embeddings import init_embeddings

    return init_embeddings(
        **_model_kwargs(
            embedding_cfg,
            model,
            label="Embedding",
            api_key_override=api_key_override,
        )
    )


def _read_secrets_stdin(*, max_bytes: int = 65_536) -> dict[str, str | None]:
    payload = sys.stdin.buffer.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("Worker secret payload is too large")
    if not payload.strip():
        return {}
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Worker secret payload must be an object")
    allowed = {"llm_api_key", "embedding_api_key"}
    if set(value) - allowed:
        raise ValueError("Worker secret payload contains unknown fields")
    return {
        key: (str(item) if item is not None else None)
        for key, item in value.items()
    }


def _redact_secrets(text: str, values: list[str]) -> str:
    redacted = text
    for value in values:
        if value:
            redacted = redacted.replace(value, "[REDACTED]")
    return redacted


class _NoopSink:
    """Dashboard adapters render progress to worker stdout directly."""

    def emit(self, _event: Any) -> None:
        return None


@dataclass(slots=True)
class _DashboardProcessAgent:
    """Adapt one dashboard registry target to the shared agent protocol."""

    adapter: Any
    context: Any

    async def ainvoke(
        self, inputs: Any, config: Mapping[str, Any] | None = None
    ) -> str:
        del config
        return await self.adapter.ainvoke(
            ctx=self.context, inputs=inputs, sink=_NoopSink()
        )


async def _attach_mcp_tools(adapter: Any, mcp_cfg: Mapping[str, Any]) -> None:
    """Install dashboard-configured MCP tools before an adapter invokes."""
    mcp_enabled = bool(mcp_cfg.get("enabled", True))
    mcp_servers = mcp_cfg.get("servers") or {}
    if not mcp_enabled or not isinstance(mcp_servers, dict) or not mcp_servers:
        return

    from ursa.agents.base import AgentWithTools
    from ursa.util.mcp import start_mcp_client

    async def attach(agent_obj: Any, _ctx: Any, _inputs: Any) -> None:
        client = start_mcp_client(mcp_servers)
        targets: list[Any] = []
        if isinstance(agent_obj, AgentWithTools):
            targets.append(agent_obj)
        executor = getattr(agent_obj, "executor", None)
        if isinstance(executor, AgentWithTools) and executor is not agent_obj:
            targets.append(executor)
        if not targets:
            logger.warning(
                "No compatible AgentWithTools target found; skipping MCP tools"
            )
            return
        for target in targets:
            # Retain the client for the target's complete invocation lifetime.
            setattr(target, "_ursa_dashboard_mcp_client", client)
            await target.add_mcp_tools(client)

    if hasattr(adapter, "set_setup_hook"):
        adapter.set_setup_hook(attach)


async def _dashboard_process_factory(
    config: Mapping[str, Any], _context: Any
) -> _DashboardProcessAgent:
    """Build a dashboard target inside an AgentManager child process.

    Configuration is intentionally plain data: the outer dashboard worker reads
    secrets once from stdin, then the shared runner serializes the resulting
    process specification without placing credentials in run metadata.
    """
    from ursa_dashboard.adapters import RunContext
    from ursa_dashboard.registry import REGISTRY
    from ursa_dashboard.settings import LLMSettings

    agent_id = str(config["agent_id"])
    try:
        entry = REGISTRY[agent_id]
    except KeyError:
        raise ValueError(f"Unknown agent_id: {agent_id}") from None

    llm_cfg = dict(config["llm"])
    embedding_cfg = dict(config.get("embedding") or {})
    credentials = dict(config.get("credentials") or {})
    llm_disabled = bool(llm_cfg.get("disabled")) or str(
        llm_cfg.get("model") or ""
    ).strip().lower() in {"none", "disabled"}
    llm = (
        None
        if llm_disabled
        else await asyncio.to_thread(
            _init_llm,
            llm_cfg,
            api_key_override=credentials.get("llm_api_key"),
        )
    )
    embedding = await asyncio.to_thread(
        _init_embedding,
        embedding_cfg,
        api_key_override=credentials.get("embedding_api_key"),
    )

    agent_init = dict(config.get("agent_init") or {})
    agent_init.setdefault(
        "tokens_before_summarize",
        LLMSettings.model_validate(llm_cfg).tokens_before_summarize,
    )
    agent_init["thread_id"] = "ursa"
    agent_init.setdefault("enable_metrics", True)
    if embedding is not None:
        if agent_init.get("rag_tools"):
            agent_init.setdefault("rag_tool_embedding", embedding)
        if agent_id == "rag_agent":
            agent_init.setdefault("embedding", embedding)

    adapter = entry.build_adapter(llm, agent_init)
    await _attach_mcp_tools(adapter, dict(config.get("mcp") or {}))
    return _DashboardProcessAgent(
        adapter=adapter,
        context=RunContext(
            run_id=str(config["run_id"]),
            agent_id=agent_id,
            workspace_dir=Path(str(config["workspace_dir"])),
        ),
    )


async def _run_with_shared_agent_manager(
    config: Mapping[str, Any], inputs: Any
) -> str:
    """Run one dashboard target through the shared process runtime."""
    manager = AgentManager()
    runtime_id = "dashboard-run"
    await manager.register(
        AgentSpec(
            runtime_id,
            "ursa_dashboard.worker_main:_dashboard_process_factory",
            config=config,
        )
    )
    try:
        result = await manager.invoke(runtime_id, inputs)
    finally:
        await manager.aclose()
    if not isinstance(result, str):
        raise TypeError("Dashboard agent returned a non-string result")
    return result


def main() -> int:
    inject_truststore_into_ssl()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workspace-dir", required=True)
    ap.add_argument(
        "--params-json", required=True, help="UI params as JSON file"
    )
    ap.add_argument("--agent-init-json", required=True)
    ap.add_argument("--llm-json", required=True)
    ap.add_argument("--embedding-json", required=False)
    ap.add_argument("--mcp-json", required=False)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--secrets-stdin", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.output_json)
    secret_values: list[str] = []
    try:
        secrets_payload = _read_secrets_stdin() if args.secrets_stdin else {}
        secret_values = [
            value
            for value in secrets_payload.values()
            if isinstance(value, str) and value
        ]
        params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        agent_init = json.loads(
            Path(args.agent_init_json).read_text(encoding="utf-8")
        )
        llm_cfg = json.loads(Path(args.llm_json).read_text(encoding="utf-8"))
        embedding_cfg: dict[str, Any] = {}
        if args.embedding_json:
            try:
                embedding_cfg = json.loads(
                    Path(args.embedding_json).read_text(encoding="utf-8")
                )
            except Exception:
                embedding_cfg = {}
        mcp_cfg: dict[str, Any] = {}
        if args.mcp_json:
            try:
                mcp_cfg = json.loads(
                    Path(args.mcp_json).read_text(encoding="utf-8")
                )
            except Exception:
                mcp_cfg = {}

        # Validate and translate dashboard UI input before sending it through
        # the shared runner. Registry imports stay late to keep worker startup
        # cheap and avoid loading every optional agent in the parent process.
        from ursa_dashboard.registry import REGISTRY

        try:
            entry = REGISTRY[args.agent_id]
        except KeyError:
            raise ValueError(f"Unknown agent_id: {args.agent_id}")

        inputs_obj = entry.build_inputs(params)
        final_text = asyncio.run(
            _run_with_shared_agent_manager(
                {
                    "agent_id": args.agent_id,
                    "run_id": args.run_id,
                    "workspace_dir": args.workspace_dir,
                    "agent_init": agent_init,
                    "llm": llm_cfg,
                    "embedding": embedding_cfg,
                    "mcp": mcp_cfg,
                    # This mapping exists only in memory in the outer worker
                    # and its managed child; it is never written to the run
                    # files alongside the non-secret configuration.
                    "credentials": secrets_payload,
                },
                inputs_obj,
            )
        )
        secrets_payload = {}

        out = {
            "status": "succeeded",
            "content_type": "text/markdown",
            "text": final_text,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 0
    except Exception as e:
        message = _redact_secrets(str(e), secret_values)
        stack = _redact_secrets(traceback.format_exc(), secret_values)
        err = {
            "status": "failed",
            "content_type": "text/plain",
            "text": f"Run failed: {message}",
            "error_type": e.__class__.__name__,
            "message": message,
            "stack": stack,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
