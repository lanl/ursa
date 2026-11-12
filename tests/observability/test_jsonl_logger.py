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
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from ursa.observability.jsonl_logger import JSONLLogEventHandler


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


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
@pytest.mark.parametrize(
    ("model", "expected_usage"),
    [
        (
            FakeMessagesListChatModel(responses=[AIMessage(content="answer")]),
            {},
        ),
        (
            MixedUsageChatModel(),
            {
                "input_tokens": 12,
                "output_tokens": 8,
                "reasoning_tokens": 4,
                "cached_tokens": 6,
            },
        ),
    ],
)
async def test_chat_usage_is_recorded_without_inventing_counts(
    tmp_path, model, expected_usage
):
    path = tmp_path / "ursa.jsonl"
    with path.open("w", encoding="utf-8", buffering=1) as stream:
        await model.ainvoke(
            [HumanMessage(content="question")],
            config={"callbacks": [JSONLLogEventHandler(stream)]},
        )

    end = _read_records(path)[-1]
    assert (end["type"], end["event"]) == ("chat", "end")
    assert end["usage"] == expected_usage


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
