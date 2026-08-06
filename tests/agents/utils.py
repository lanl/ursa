"""Shared fakes for agent tests."""

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class RecordingChatModel(GenericFakeChatModel):
    """Fake chat model that records every request an agent sends.

    ``calls`` collects the full message list of each LLM call in order,
    covering plain and tool-bound ``invoke`` calls as well as
    ``with_structured_output`` invocations. ``response`` is returned for
    plain calls; ``structured_factory`` builds the object returned for
    structured calls.
    """

    messages: Iterator[AIMessage | str] = Field(
        default_factory=lambda: iter([AIMessage(content="unused")])
    )
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    response: str = "OK"
    structured_factory: Callable[[type], Any] | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=self.response))
            ]
        )

    def bind_tools(self, tools, **kwargs):
        return self

    def model_copy(self, update=None):
        return self

    def with_structured_output(self, schema, **kwargs):
        model = self
        include_raw = kwargs.get("include_raw", False)

        class StructuredOutput:
            def invoke(self, messages, config=None):
                model.calls.append(list(messages))
                if model.structured_factory is None:
                    raise NotImplementedError(
                        "RecordingChatModel needs a structured_factory to "
                        f"answer with_structured_output({schema!r})"
                    )
                parsed = model.structured_factory(schema)
                if include_raw:
                    return {
                        "raw": AIMessage(content=""),
                        "parsed": parsed,
                        "parsing_error": None,
                    }
                return parsed

        return StructuredOutput()


class ScriptedRecordingChatModel(RecordingChatModel):
    """Recording model that pops scripted responses before the default.

    Seed ``script`` with messages (for example ``AIMessage`` instances
    carrying ``tool_calls``) to drive an agent's tool loop; once the
    script is exhausted, the model falls back to ``response``.
    """

    script: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if self.script:
            message = self.script.pop(0)
        else:
            message = AIMessage(content=self.response)
        return ChatResult(generations=[ChatGeneration(message=message)])


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
