from __future__ import annotations

import pytest

from ursa.cli.config import UrsaConfig
from ursa.runtime import AgentRuntime
from ursa.util.mcp import parse_server_parameters


class _Manager:
    def __init__(self, mcp_result=None) -> None:
        self.specs = {}
        self.calls = []
        self.mcp_result = mcp_result or {
            "complete": True,
            "tools": [
                {
                    "name": "remote_lookup",
                    "description": "Find remote information.",
                    "schema": {"type": "object", "properties": {}},
                    "mcp_input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    "mcp_server": "demo",
                }
            ],
        }

    async def register(self, spec) -> None:
        self.specs[spec.agent_id] = spec

    def set_runtime_request_handler(self, *_args) -> None:
        pass

    async def start_agent(self, _agent_id) -> None:
        pass

    async def restart_agent(self, _agent_id) -> None:
        pass

    async def call(self, agent_id, method, *_args) -> dict:
        self.calls.append((agent_id, method))
        assert method == "describe_mcp_tools"
        return self.mcp_result

    async def unregister(self, agent_id, **_kwargs) -> None:
        self.specs.pop(agent_id, None)

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_child_reuses_root_mcp_tool_specs(tmp_path):
    manager = _Manager()
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            mcp_servers={
                "demo": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        assert manager.calls == [(root.runtime_id, "describe_mcp_tools")]
        assert runtime._mcp_tool_specs == [
            {
                "server": "demo",
                "name": "remote_lookup",
                "description": "Find remote information.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        child_path = await runtime._spawn_runtime_subagent(
            root.runtime_id, "worker"
        )
        child_id = runtime._hierarchies[root.runtime_id][child_path]
        child_spec = manager.specs[child_id]
        assert child_spec.config["mcp_tool_specs"] == runtime._mcp_tool_specs
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_incomplete_mcp_description_does_not_suppress_discovery(tmp_path):
    manager = _Manager(mcp_result={"complete": False, "tools": []})
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            mcp_servers={
                "demo": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        assert runtime._mcp_tool_specs is None
        await runtime._spawn_runtime_subagent(root.runtime_id, "worker")
        child_id = runtime._hierarchies[root.runtime_id]["root/worker"]
        assert "mcp_tool_specs" not in manager.specs[child_id].config
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_reconfigure_invalidates_mcp_tool_cache(tmp_path):
    manager = _Manager()
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            mcp_servers={
                "demo": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        assert runtime._mcp_tool_specs is not None
        await runtime.reconfigure_models(
            runtime.config.llm_model, runtime.config.emb_model
        )
        assert runtime._mcp_tool_specs is None
        assert runtime._mcp_tool_specs_fingerprint is None
        assert root.runtime_id not in manager.specs
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_changed_server_with_same_name_bypasses_cached_specs(tmp_path):
    manager = _Manager()
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            mcp_servers={
                "demo": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        runtime.config.mcp_servers["demo"] = parse_server_parameters({
            "transport": "streamable-http",
            "url": "https://changed.example.test/mcp",
        })
        await runtime._spawn_runtime_subagent(root.runtime_id, "worker")
        child_id = runtime._hierarchies[root.runtime_id]["root/worker"]
        assert "mcp_tool_specs" not in manager.specs[child_id].config
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_root_restart_recaptures_changed_remote_tools(tmp_path):
    manager = _Manager()
    runtime = AgentRuntime(
        UrsaConfig(
            workspace=tmp_path,
            llm_model={"model": "fake:chat"},
            mcp_servers={
                "demo": {
                    "transport": "streamable-http",
                    "url": "https://example.test/mcp",
                }
            },
        ),
        manager=manager,
    )
    try:
        root = await runtime.get_agent("chat")
        manager.mcp_result["tools"][0]["name"] = "updated_lookup"
        await runtime.restart_agent("chat")
        await runtime._spawn_runtime_subagent(root.runtime_id, "worker")
        child_id = runtime._hierarchies[root.runtime_id]["root/worker"]
        assert manager.specs[child_id].config["mcp_tool_specs"][0]["name"] == (
            "updated_lookup"
        )
    finally:
        await runtime.aclose()
