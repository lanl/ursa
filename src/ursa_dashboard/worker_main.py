from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

from ursa.util.http import inject_truststore_into_ssl

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
    if "api_key" in config:
        raise ValueError(
            f"Literal {label} API keys are not accepted in worker config"
        )
    if override is not _UNSET:
        return str(override) if override else None

    env_name = str(config.get("api_key_env") or "").strip()
    if not env_name:
        return None
    env_val = os.environ.get(env_name)
    if not env_val:
        raise ValueError(
            f"{label} api_key_env '{env_name}' is not set in the worker environment"
        )
    return env_val


def _init_llm(
    llm_cfg: dict[str, Any],
    *,
    api_key_override: str | None | object = _UNSET,
):
    # Avoid importing langchain unless actually executing.
    from langchain.chat_models import init_chat_model  # type: ignore

    raw_base_url = llm_cfg.get("base_url")
    base_url = str(raw_base_url).strip() if raw_base_url is not None else None

    api_key = _api_key_from_config(
        llm_cfg, label="LLM", override=api_key_override
    )

    model_kwargs = llm_cfg.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        raise ValueError("llm.model_kwargs must be a JSON object")

    model = _normalize_model(
        str(llm_cfg.get("model") or "openai:gpt-5.4-mini"),
        model_kwargs.get("model_provider"),
    )

    kwargs: dict[str, Any] = {**model_kwargs, "model": model}

    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return init_chat_model(**kwargs)


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

    from langchain.embeddings import init_embeddings  # type: ignore

    raw_base_url = embedding_cfg.get("base_url")
    base_url = str(raw_base_url).strip() if raw_base_url is not None else None

    api_key = _api_key_from_config(
        embedding_cfg, label="Embedding", override=api_key_override
    )

    model_kwargs = embedding_cfg.get("model_kwargs") or {}
    if not isinstance(model_kwargs, dict):
        raise ValueError("embedding.model_kwargs must be a JSON object")

    kwargs: dict[str, Any] = {
        **model_kwargs,
        "model": _normalize_model(model, model_kwargs.get("model_provider")),
    }
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url

    return init_embeddings(**kwargs)


def _maybe_run_async(result):
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


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

        # Late import to keep worker startup cheap.
        from ursa_dashboard.registry import REGISTRY

        if args.agent_id not in REGISTRY:
            raise ValueError(f"Unknown agent_id: {args.agent_id}")

        entry = REGISTRY[args.agent_id]

        # Demo/support: allow runs without an LLM (for smoke tests / demo mode).
        llm_disabled = bool(llm_cfg.get("disabled")) or str(
            llm_cfg.get("model") or ""
        ).strip().lower() in {"none", "disabled"}
        llm = (
            None
            if llm_disabled
            else _init_llm(
                llm_cfg,
                api_key_override=secrets_payload.get("llm_api_key"),
            )
        )
        embedding = _init_embedding(
            embedding_cfg,
            api_key_override=secrets_payload.get("embedding_api_key"),
        )
        secrets_payload = {}
        # Ensure consistent cross-session threading.
        agent_init = dict(agent_init)
        agent_init["thread_id"] = "ursa"
        agent_init.setdefault("enable_metrics", True)
        if embedding is not None:
            if agent_init.get("rag_tools"):
                agent_init.setdefault("rag_tool_embedding", embedding)
            if args.agent_id == "rag_agent":
                agent_init.setdefault("embedding", embedding)

        adapter = entry.build_adapter(llm, agent_init)

        # Optionally attach MCP tools to ExecutionAgent (or an executor inside workflows).
        # MCP configuration is snapshotted into the run record at creation time, so changes
        # apply to new runs only.
        mcp_enabled = bool((mcp_cfg or {}).get("enabled", True))
        mcp_servers = (mcp_cfg or {}).get("servers") or {}
        if mcp_enabled and isinstance(mcp_servers, dict) and mcp_servers:

            async def _attach_mcp_tools(agent_obj: Any) -> None:
                from ursa.agents.base import AgentWithTools  # type: ignore
                from ursa.util.mcp import start_mcp_client  # type: ignore

                client = start_mcp_client(mcp_servers)

                def _targets(root: Any) -> list[Any]:
                    t: list[Any] = []
                    seen: set[int] = set()

                    def add(x: Any) -> None:
                        if x is None:
                            return
                        ix = id(x)
                        if ix in seen:
                            return
                        seen.add(ix)
                        t.append(x)

                    if isinstance(root, AgentWithTools):
                        add(root)
                    ex = getattr(root, "executor", None)
                    if ex is not None and isinstance(ex, AgentWithTools):
                        add(ex)
                    return t

                targets = _targets(agent_obj)
                if not targets:
                    print(
                        "[mcp] No compatible AgentWithTools target found; skipping MCP tool attachment",
                        file=sys.stderr,
                    )
                    return

                for tgt in targets:
                    # Keep a reference so the client stays alive for the run.
                    setattr(tgt, "_ursa_dashboard_mcp_client", client)
                    await tgt.add_mcp_tools(client)

            # Install hook if the adapter supports it.
            if hasattr(adapter, "set_setup_hook"):
                adapter.set_setup_hook(
                    lambda agent_obj, _ctx, _inputs: _attach_mcp_tools(
                        agent_obj
                    )
                )

        inputs_obj = entry.build_inputs(params)

        from ursa_dashboard.adapters import RunContext

        ctx = RunContext(
            run_id=args.run_id,
            agent_id=args.agent_id,
            workspace_dir=Path(args.workspace_dir),
        )

        # Adapter expects an EventSink, but for subprocess isolation we do not
        # use the event stream from inside the worker. Provide a no-op sink.
        class _NoopSink:
            def emit(self, _event):
                return None

        final_text = asyncio.run(
            adapter.ainvoke(ctx=ctx, inputs=inputs_obj, sink=_NoopSink())
        )

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
