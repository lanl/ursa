import asyncio

import pytest
from langchain_core.runnables.config import var_child_runnable_config

from ursa.util.agent_invocation import ainvoke_agent, invoke_agent


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_named_invocation_restores_parent_context_after_failure(
    asynchronous,
):
    parent = {
        "configurable": {
            "thread_id": "parent",
            "checkpoint_ns": "parent:task",
            "checkpoint_id": "old-checkpoint",
            "checkpoint_map": {"": "old-checkpoint"},
            "__pregel_checkpointer": object(),
            "__pregel_runtime": object(),
            "user_option": "retained",
        },
        "metadata": {"experiment": "test", "thread_id": "parent"},
        "tags": ["research"],
    }

    def check(config):
        assert config["configurable"] == {
            "thread_id": "ursa",
            "checkpoint_ns": "",
            "user_option": "retained",
        }
        assert config["metadata"] == {"experiment": "test", "thread_id": "ursa"}
        assert config["tags"] == ["research"]
        assert var_child_runnable_config.get() is config

    class FailingAgent:
        agent_name = "member"

        def invoke(self, inputs, config):
            check(config)
            raise ValueError("failed")

        async def ainvoke(self, inputs, config):
            check(config)
            raise asyncio.CancelledError

    token = var_child_runnable_config.set(parent)
    try:
        if asynchronous:
            with pytest.raises(asyncio.CancelledError):
                await ainvoke_agent(FailingAgent(), "task", config=parent)
        else:
            with pytest.raises(ValueError, match="failed"):
                invoke_agent(FailingAgent(), "task", config=parent)
        assert var_child_runnable_config.get() is parent
        assert parent["configurable"]["thread_id"] == "parent"
        assert parent["configurable"]["checkpoint_ns"] == "parent:task"
        assert parent["metadata"]["thread_id"] == "parent"
    finally:
        var_child_runnable_config.reset(token)
