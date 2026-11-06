"""Exercise delayed loading in fresh processes and response-before-cleanup."""

import json
import os
import subprocess
import sys
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ursa.cli.config import EmbModelConfig, UrsaConfig
from ursa.cli.one_shot import run_once
from ursa.cli.runtime import HITL


@pytest.mark.parametrize("scenario", ["prompt", "mcp", "exec"])
def test_cli_defers_unused_imports(scenario, tmp_path, monkeypatch):
    for key in tuple(os.environ):
        if key.startswith(("URSA_", "OTEL_", "LANGSMITH_", "LANGCHAIN_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("URSA_SYSTEM_CONFIG", str(tmp_path / "system.yaml"))
    monkeypatch.setenv(
        "PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring"
    )
    servers = {
        "stdio": {"command": "ursa-startup-test-must-not-connect"},
        "sse": {"transport": "sse", "url": "http://127.0.0.1:9/sse"},
        "http": {
            "transport": "streamable-http",
            "url": "http://127.0.0.1:9/mcp",
            "headers": {"Authorization": {"env": "UNSET_STARTUP_TEST_TOKEN"}},
        },
    }
    config_path = tmp_path / "config" / "ursa" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps({
            "workspace": str(tmp_path / "workspace"),
            "mcp_servers": servers if scenario == "mcp" else {},
        })
    )
    result = tmp_path / "modules.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("_startup_probe.py")),
            scenario,
            str(result),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    if scenario == "exec":
        assert completed.stdout.strip() == "Hello!"
    modules = json.loads(result.read_text())
    forbidden = {"chromadb", "mcp", "fastmcp", "langchain_openai", "pymupdf"}
    if scenario == "exec":
        forbidden.add("textual")
        assert "langchain_core.language_models.fake_chat_models" in modules
    else:
        forbidden.update({
            "ursa.agents.base",
            "langchain_core.language_models.fake_chat_models",
            "numpy",
        })
    for prefix in forbidden:
        assert not any(
            name == prefix or name.startswith(prefix + ".") for name in modules
        ), prefix


@pytest.mark.parametrize("rag_tools", [[], ["saved-rag"]])
async def test_embeddings_wait_for_rag_tools(tmp_path, monkeypatch, rag_tools):
    runtime = HITL(
        UrsaConfig(
            workspace=tmp_path,
            rag_tools=rag_tools,
            emb_model=EmbModelConfig(model="unused", model_provider="openai"),
        )
    )
    runtime.model = object()
    calls = []
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", lambda *a: None
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.enforce_model_group_policy", lambda *a: None
    )
    monkeypatch.setattr(
        EmbModelConfig, "init_embedding", lambda self: calls.append(1)
    )
    runtime._agent_dependencies()
    assert bool(calls) == bool(rag_tools)
    await runtime.aclose()


def test_exec_displays_response_before_cleanup(tmp_path):
    output = StringIO()

    async def close():
        assert output.getvalue().strip() == "Hello!"

    runtime = SimpleNamespace(
        workspace=tmp_path,
        agents={"chat": object()},
        run_agent=AsyncMock(return_value="Hello!"),
        aclose=AsyncMock(side_effect=close),
    )
    assert run_once(runtime, "say hello", stdout=output) == "Hello!"
    runtime.aclose.assert_awaited_once()
