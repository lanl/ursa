from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

from ursa.util.http import inject_truststore_into_ssl

from .worker_main import _init_llm, _redact_secrets


def _read_secrets_stdin(*, max_bytes: int = 1_000_000) -> dict[str, Any]:
    payload = sys.stdin.buffer.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("Worker secret payload is too large")
    if not payload.strip():
        return {}
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict) or set(value) - {
        "llm_api_key",
        "member_api_keys",
    }:
        raise ValueError("Worker secret payload contains unknown fields")
    member_keys = value.get("member_api_keys") or {}
    if not isinstance(member_keys, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in member_keys.items()
    ):
        raise ValueError("Worker member credentials are invalid")
    return {
        "llm_api_key": value.get("llm_api_key"),
        "member_api_keys": member_keys,
    }


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("final", "result", "prompt"):
            if isinstance(result.get(key), str):
                return result[key]
    return str(result)


async def _run(args: argparse.Namespace, secrets: dict[str, Any]) -> Any:
    from ursa.environments import (
        AgentSymposiumEnvironment,
        AgentTeamEnvironment,
        arun_with_visualization,
    )

    launch = json.loads(Path(args.llm_json).read_text(encoding="utf-8"))
    llm_cfg = launch.get("llm") or {}
    config = yaml.safe_load(Path(args.config_yaml).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Environment configuration must be a mapping")
    config["group"] = args.group
    task = json.loads(Path(args.task_json).read_text(encoding="utf-8"))[
        "prompt"
    ]
    member_keys = secrets.get("member_api_keys") or {}
    previous = {name: os.environ.get(name) for name in member_keys}
    try:
        os.environ.update(member_keys)
        llm = _init_llm(
            llm_cfg,
            api_key_override=secrets.get("llm_api_key"),
        )
        if args.environment_type == "agent_team":
            environment = AgentTeamEnvironment(llm=llm, config=config)
        elif args.environment_type == "agent_symposium":
            environment = AgentSymposiumEnvironment(llm=llm, config=config)
        else:  # pragma: no cover - argparse constrains this
            raise ValueError(
                f"Unsupported environment type: {args.environment_type}"
            )
        return await arun_with_visualization(
            environment,
            task,
            run_id=args.run_id,
        )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def main() -> int:
    inject_truststore_into_ssl()
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument(
        "--environment-type",
        required=True,
        choices=["agent_team", "agent_symposium"],
    )
    parser.add_argument("--config-yaml", required=True)
    parser.add_argument("--task-json", required=True)
    parser.add_argument("--llm-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--secrets-stdin", action="store_true")
    args = parser.parse_args()
    output_path = Path(args.output_json)
    secret_values: list[str] = []
    try:
        secrets = _read_secrets_stdin() if args.secrets_stdin else {}
        secret_values = [
            value
            for value in [
                secrets.get("llm_api_key"),
                *(secrets.get("member_api_keys") or {}).values(),
            ]
            if isinstance(value, str) and value
        ]
        result = asyncio.run(_run(args, secrets))
        secrets = {}
        output_path.write_text(
            json.dumps(
                {"text": _result_text(result), "content_type": "text/markdown"},
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return 0
    except BaseException as exc:
        message = _redact_secrets(str(exc), secret_values)
        trace = _redact_secrets(traceback.format_exc(), secret_values)
        output_path.write_text(
            json.dumps(
                {
                    "error_type": type(exc).__name__,
                    "message": message,
                    "traceback": trace,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(trace, file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
