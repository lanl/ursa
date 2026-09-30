import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import tool

from ursa.observability.jsonl_logger import JSONLLogEventHandler
from ursa.util.events import AgentEvents


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@tool
async def echo_payload(text: str) -> dict[str, str]:
    """Echo text in a structured tool result."""
    return {"echo": text}


@tool
async def failing_tool(text: str) -> str:
    """Raise a deterministic tool failure."""
    raise RuntimeError(f"tool failed: {text}")


async def _raise_chain(_value: Any) -> None:
    raise RuntimeError("chain failed")


class FailingChatModel(FakeMessagesListChatModel):
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("chat failed")


class MixedUsageChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mixed-usage-test"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="normalized",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                ),
                ChatGeneration(
                    message=AIMessage(
                        content="raw",
                        response_metadata={
                            "token_usage": {
                                "prompt_tokens": 2,
                                "completion_tokens": 3,
                                "completion_tokens_details": {
                                    "reasoning_tokens": 4
                                },
                                "prompt_tokens_details": {"cached_tokens": 6},
                            }
                        },
                    )
                ),
                ChatGeneration(
                    message=AIMessage(
                        content="invalid",
                        response_metadata={
                            "token_usage": {
                                "prompt_tokens": True,
                                "completion_tokens": -1,
                                "cached_tokens": 1.5,
                            }
                        },
                    )
                ),
            ]
        )


@pytest.mark.asyncio
async def test_real_callbacks_record_complete_lifecycles(tmp_path):
    path = tmp_path / "ursa.jsonl"
    response = AIMessage(
        content="model answer",
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 7,
            "total_tokens": 19,
        },
        response_metadata={
            "token_usage": {
                "prompt_tokens": 12,
                "completion_tokens": 7,
                "completion_tokens_details": {"reasoning_tokens": 4},
                "prompt_tokens_details": {"cached_tokens": 6},
            }
        },
    )
    model = FakeMessagesListChatModel(responses=[response])

    async def lifecycle(
        inputs: dict[str, str], config: RunnableConfig
    ) -> dict[str, Any]:
        await AgentEvents(agent="ExecutionAgent", config=config).aemit(
            "Inspecting the workspace",
            stage="work",
        )
        answer = await model.ainvoke(
            [HumanMessage(content=inputs["question"])], config=config
        )
        echoed = await echo_payload.ainvoke({"text": "needle"}, config=config)
        return {"answer": answer, "tool": echoed}

    runnable = RunnableLambda(lifecycle).with_config({
        "run_name": "observed_chain"
    })
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        await runnable.ainvoke(
            {"question": "solve it"},
            config={"callbacks": [JSONLLogEventHandler(stream)]},
        )

    records = _read_records(path)
    assert [(record["type"], record["event"]) for record in records] == [
        ("chain", "start"),
        ("agent", "progress"),
        ("chat", "start"),
        ("chat", "end"),
        ("tool", "start"),
        ("tool", "end"),
        ("chain", "end"),
    ]
    timestamps = [record["timestamp_monotonic_ns"] for record in records]
    assert all(isinstance(timestamp, int) for timestamp in timestamps)
    assert timestamps == sorted(timestamps)

    (
        chain_start,
        progress,
        chat_start,
        chat_end,
        tool_start,
        tool_end,
        chain_end,
    ) = records
    chain_id = chain_start["run_id"]
    assert chain_start["name"] == "observed_chain"
    assert chain_start["input"] == {"question": "solve it"}
    assert chain_end["run_id"] == chain_id
    assert chain_end["name"] == "observed_chain"
    assert "model answer" in json.dumps(chain_end["output"])
    assert progress["run_id"] == chain_id
    assert progress["parent_run_id"] is None
    assert all(
        record["parent_run_id"] == chain_id
        for record in (chat_start, chat_end, tool_start, tool_end)
    )
    assert chat_start["input"][0][0]["content"] == "solve it"
    assert "model answer" in json.dumps(chat_end["output"])
    assert chat_end["usage"] == {
        "input_tokens": 12,
        "output_tokens": 7,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }
    assert tool_start["name"] == "echo_payload"
    assert tool_start["input"] == {"text": "needle"}
    assert tool_end["output"] == {"echo": "needle"}
    assert progress["name"] == "ExecutionAgent"
    assert progress["data"]["message"] == "Inspecting the workspace"


@pytest.mark.asyncio
async def test_chat_without_usage_does_not_invent_zero_counts(tmp_path):
    path = tmp_path / "ursa.jsonl"
    model = FakeMessagesListChatModel(responses=[AIMessage(content="answer")])

    with path.open("w", encoding="utf-8", buffering=1) as stream:
        await model.ainvoke(
            [HumanMessage(content="question")],
            config={"callbacks": [JSONLLogEventHandler(stream)]},
        )

    end = _read_records(path)[-1]
    assert (end["type"], end["event"]) == ("chat", "end")
    assert end["usage"] == {}


@pytest.mark.asyncio
async def test_usage_is_selected_per_generation_and_validated(tmp_path):
    path = tmp_path / "ursa.jsonl"
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        await MixedUsageChatModel().ainvoke(
            [HumanMessage(content="question")],
            config={"callbacks": [JSONLLogEventHandler(stream)]},
        )

    assert _read_records(path)[-1]["usage"] == {
        "input_tokens": 12,
        "output_tokens": 8,
        "reasoning_tokens": 4,
        "cached_tokens": 6,
    }


@pytest.mark.asyncio
async def test_callback_manager_records_agent_action_and_finish(tmp_path):
    path = tmp_path / "ursa.jsonl"
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        manager = AsyncCallbackManager([JSONLLogEventHandler(stream)])
        run = await manager.on_chain_start(
            {"name": "agent_executor"}, {"input": "question"}
        )
        await run.on_agent_action(
            AgentAction(tool="echo_payload", tool_input="hello", log="call")
        )
        await run.on_agent_finish(
            AgentFinish(return_values={"output": "done"}, log="finish")
        )
        await run.on_chain_end({"output": "done"})

    records = _read_records(path)
    assert [(record["type"], record["event"]) for record in records] == [
        ("chain", "start"),
        ("agent", "action"),
        ("agent", "end"),
        ("chain", "end"),
    ]
    assert records[1]["input"]["tool"] == "echo_payload"
    assert records[2]["output"]["return_values"] == {"output": "done"}


@pytest.mark.asyncio
async def test_real_callbacks_record_chain_chat_and_tool_errors(tmp_path):
    path = tmp_path / "ursa.jsonl"
    handler: JSONLLogEventHandler
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        handler = JSONLLogEventHandler(stream)
        config = {"callbacks": [handler]}

        with pytest.raises(RuntimeError, match="chain failed"):
            await RunnableLambda(_raise_chain).ainvoke("input", config=config)
        with pytest.raises(RuntimeError, match="chat failed"):
            await FailingChatModel(
                responses=[AIMessage(content="unused")]
            ).ainvoke([HumanMessage(content="question")], config=config)
        with pytest.raises(RuntimeError, match="tool failed"):
            await failing_tool.ainvoke({"text": "input"}, config=config)

    records = _read_records(path)
    assert [(record["type"], record["event"]) for record in records] == [
        ("chain", "start"),
        ("chain", "error"),
        ("chat", "start"),
        ("chat", "error"),
        ("tool", "start"),
        ("tool", "error"),
    ]
    for start, error in zip(records[::2], records[1::2], strict=True):
        assert error["run_id"] == start["run_id"]
        assert error["name"] == start["name"]
        assert error["error"]["error_type"] == "RuntimeError"
