"""Shared fakes for agent tests."""

from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def message_stream(content: str) -> Iterator[AIMessage]:
    """Yield fresh messages indefinitely for a deterministic fake model."""
    while True:
        yield AIMessage(content=content)


def message_with_usage(content: str, **kwargs: Any) -> AIMessage:
    """Build a fresh fake response with one input and one output token."""
    return AIMessage(
        content=content,
        usage_metadata={
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
        **kwargs,
    )


def tool_call_message(name, args, call_id, *, content="", **call_fields):
    """Build an assistant response requesting one tool with explicit inputs."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": args, "id": call_id, **call_fields}],
    )


def message_stream_with_usage(content: str) -> Iterator[AIMessage]:
    while True:
        yield message_with_usage(content=content)


def simple_plan(schema):
    return schema(
        steps=[
            {
                "name": "Add numbers",
                "description": "Add 1 and 2.",
                "requires_code": False,
                "expected_outputs": ["sum"],
                "success_criteria": ["sum equals 3"],
            }
        ]
    )


class ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def recording_model(log: list, content: str) -> ToolReadyFakeChatModel:
    """Record each request while yielding fresh, token-bearing responses."""

    class _Recorder(ToolReadyFakeChatModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            log.append(list(messages))
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )

    return _Recorder(messages=message_stream_with_usage(content))


class RecordingChatModel(ToolReadyFakeChatModel):
    """Fake chat model that records every request an agent sends.

    ``calls`` collects the full message list of each LLM call in order,
    covering plain and tool-bound ``invoke`` calls as well as
    ``with_structured_output`` invocations. ``response`` is returned for
    plain calls; ``structured_factory`` builds the object returned for
    structured calls. An optional ``script`` supplies messages before falling
    back to ``response``, for tests that exercise tool-call sequences.
    """

    messages: Iterator[AIMessage | str] = Field(
        default_factory=lambda: iter([AIMessage(content="unused")])
    )
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    response: str = "OK"
    script: list = Field(default_factory=list)
    structured_factory: Callable[[type], Any] | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if self.script:
            message = self.script.pop(0)
        else:
            message = AIMessage(content=self.response)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def model_copy(self, update=None):
        return self

    def with_structured_output(self, schema, **kwargs):
        include_raw = kwargs.get("include_raw", False)

        def invoke(messages, config=None):
            self.calls.append(list(messages))
            if self.structured_factory is None:
                raise NotImplementedError(
                    "RecordingChatModel needs a structured_factory to "
                    f"answer with_structured_output({schema!r})"
                )
            parsed = self.structured_factory(schema)
            if include_raw:
                return {
                    "raw": AIMessage(content=""),
                    "parsed": parsed,
                    "parsing_error": None,
                }
            return parsed

        return SimpleNamespace(invoke=invoke)


def assert_requests_provider_valid(calls: list[list[BaseMessage]]) -> None:
    """Assert every recorded request has a shape chat providers accept.

    Claude 4.6 and newer reject any request whose final message is an
    assistant turn (assistant-message prefill was removed); providers that
    still accept such requests treat them as continuations of the model's
    own output rather than a new turn.
    """
    assert calls, "agent made no LLM calls"
    for index, call in enumerate(calls):
        assert call, f"call {index} sent an empty message list"
        assert not isinstance(call[-1], AIMessage), (
            f"call {index} ends on an assistant turn: "
            f"{[message.type for message in call]}"
        )
        past_system_prefix = False
        for message in call:
            if isinstance(message, SystemMessage):
                assert not past_system_prefix, (
                    f"call {index} has a system message after the leading "
                    f"system prefix, which providers reject: "
                    f"{[m.type for m in call]}"
                )
            else:
                past_system_prefix = True


class FakeDoc:
    def __init__(self, material_id: str, metadata: dict):
        self.material_id = material_id
        self._metadata = metadata

    def dict(self) -> dict:
        return self._metadata


def patch_materials_client(monkeypatch, summary_factory):
    """Install a Materials Project client with a test-specific search service."""

    class FakeMPRester:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            self.materials = SimpleNamespace(summary=summary_factory())
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ursa.agents.mp_agent.MPRester", FakeMPRester)
