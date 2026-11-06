"""Run the command in a fresh process, through a real local HTTP model API."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from benchmark.benchmarks._exec import measure_exec
from ursa.cli.one_shot import run_once


def test_exec_retains_tools_without_loading_unused_integrations():
    result = measure_exec(profile_imports=True)
    assert result["requests"][-1] == "/v1/chat/completions"
    assert result["request_seconds"] < result["response_seconds"]
    assert result["response_seconds"] <= result["exit_seconds"]
    imports = {
        line.rsplit("|", 1)[-1].strip()
        for line in result["stderr"].splitlines()
        if line.startswith("import time:")
    }
    for prefix in (
        "textual",
        "chromadb",
        "mcp",
        "fastmcp",
        "pymupdf",
        "dsi",
        "langchain_ollama",
        "ursa.tools.search_tools",
        "ursa.util.parse",
    ):
        assert not any(
            name == prefix or name.startswith(prefix + ".") for name in imports
        ), prefix


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
