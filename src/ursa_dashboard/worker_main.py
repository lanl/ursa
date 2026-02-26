from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


def _normalize_model(model: str) -> str:
    # Examples use "openai:gpt-5-mini".
    if ":" in model:
        return model
    return f"openai:{model}"


def _init_llm(llm_cfg: dict[str, Any]):
    # Avoid importing langchain unless actually executing.
    from langchain.chat_models import init_chat_model  # type: ignore

    base_url = llm_cfg.get("base_url")

    # Security: prefer reading the API key from an environment variable.
    # - If llm_cfg.api_key is provided (advanced / legacy), it is used directly.
    # - Otherwise, if llm_cfg.api_key_env_var is set, copy that env var's value
    #   into OPENAI_API_KEY for the OpenAI-compatible client stack.
    api_key = llm_cfg.get("api_key")
    if api_key is not None and str(api_key).strip() != "":
        os.environ["OPENAI_API_KEY"] = str(api_key)
    else:
        env_name = llm_cfg.get("api_key_env_var")
        if env_name is not None:
            env_name = str(env_name).strip()
        if env_name:
            env_val = os.environ.get(env_name)
            if not env_val:
                raise ValueError(
                    f"LLM api_key_env_var '{env_name}' is not set in the dashboard environment"
                )
            os.environ["OPENAI_API_KEY"] = str(env_val)
    if base_url:
        # Support multiple common env names.
        os.environ["OPENAI_BASE_URL"] = str(base_url)
        os.environ["OPENAI_API_BASE"] = str(base_url)

    model = _normalize_model(str(llm_cfg.get("model") or "openai:gpt-5-mini"))

    kwargs: dict[str, Any] = {"model": model}
    if llm_cfg.get("max_tokens") is not None:
        kwargs["max_completion_tokens"] = int(llm_cfg["max_tokens"])
    if llm_cfg.get("temperature") is not None:
        kwargs["temperature"] = float(llm_cfg["temperature"])

    return init_chat_model(**kwargs)


def _maybe_run_async(result):
    if asyncio.iscoroutine(result):
        return asyncio.run(result)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--workspace-dir", required=True)
    ap.add_argument(
        "--params-json", required=True, help="UI params as JSON file"
    )
    ap.add_argument("--agent-init-json", required=True)
    ap.add_argument("--llm-json", required=True)
    ap.add_argument("--mcp-json", required=False)
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    out_path = Path(args.output_json)
    try:
        params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        agent_init = json.loads(
            Path(args.agent_init_json).read_text(encoding="utf-8")
        )
        llm_cfg = json.loads(Path(args.llm_json).read_text(encoding="utf-8"))
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
        llm = None if llm_disabled else _init_llm(llm_cfg)
        # Ensure deterministic run id threading.
        agent_init = dict(agent_init)
        agent_init.setdefault("thread_id", args.run_id)
        agent_init.setdefault("enable_metrics", True)

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

        final_text = adapter.invoke(
            ctx=ctx, inputs=inputs_obj, sink=_NoopSink()
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
        err = {
            "status": "failed",
            "content_type": "text/plain",
            "text": f"Run failed: {e}",
            "error_type": e.__class__.__name__,
            "message": str(e),
            "stack": traceback.format_exc(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
