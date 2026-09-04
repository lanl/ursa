"""Property-based coverage for provider-valid message sequences.

Requires the optional ``hypothesis`` test dependency; the module skips
cleanly when it is absent.

Caller contract: generated histories always end on a human message,
because agents receive input through ``format_query`` after a user
turn; a caller-supplied assistant-final history flows straight through
``_normalize_inputs`` and tests caller error, not agent logic.

Known-bug configurations are pinned as deterministic strict xfails
rather than silently avoided: summarization with a plain ``BaseAgent``
crashes on the missing ``tool_llm`` (upstream #295). The
``messages_to_keep=0`` and summarized-away tool-tail cases (upstream
#296) are now fixed on main -- the summary lands as a human-role
message -- so they are asserted as passing regression guards instead.
The properties therefore explore the remaining healthy space.
"""

import tempfile

import pytest

hypothesis = pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from tests.agents.utils import (  # noqa: E402
    RecordingChatModel,
    assert_requests_provider_valid,
)
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent  # noqa: E402

_PAD = "pad " * 40


@st.composite
def chat_history(draw, min_pairs=0, max_pairs=4):
    messages = []
    if draw(st.booleans()):
        messages.append(SystemMessage("system preamble"))
    long_form = draw(st.booleans())
    for index in range(draw(st.integers(min_pairs, max_pairs))):
        body = _PAD if long_form else "short"
        messages.append(HumanMessage(f"q{index} {body}"))
        messages.append(AIMessage(f"a{index} {body}"))
    messages.append(HumanMessage("latest question"))
    return messages


@settings(max_examples=15, deadline=None, derandomize=True)
@given(history=chat_history())
def test_property_shapes_without_summarization(history):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
        llm = RecordingChatModel()
        agent = BasicChatAgent(llm=llm, workspace=workspace)
        try:
            agent.invoke({"messages": list(history)})
        finally:
            agent.close()

        assert_requests_provider_valid(llm.calls)


@settings(max_examples=15, deadline=None, derandomize=True)
@given(
    history=chat_history(min_pairs=1),
    messages_to_keep=st.sampled_from([1, 2, 20]),
)
def test_property_summarization_healthy_space(history, messages_to_keep):
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workspace:
        llm = RecordingChatModel()
        agent = ChatAgent(
            llm=llm,
            workspace=workspace,
            tokens_before_summarize=1,
            messages_to_keep=messages_to_keep,
        )
        try:
            agent.invoke({"messages": list(history)})
        finally:
            agent.close()

        assert_requests_provider_valid(llm.calls)


@pytest.mark.xfail(
    reason=(
        "summarization uses self.tool_llm, which only AgentWithTools "
        "assigns, so plain BaseAgent subclasses crash when history "
        "crosses tokens_before_summarize; see upstream issue #295"
    ),
    strict=True,
    raises=AttributeError,
)
async def test_pin_summarization_crashes_non_tool_agents(tmp_path):
    llm = RecordingChatModel()
    agent = BasicChatAgent(
        llm=llm,
        workspace=tmp_path,
        tokens_before_summarize=1,
        messages_to_keep=2,
    )
    history = []
    for index in range(4):
        history.append(HumanMessage(f"q{index} {_PAD}"))
        history.append(AIMessage(f"a{index} {_PAD}"))
    history.append(HumanMessage("latest question"))

    await agent.ainvoke({"messages": history})

    assert_requests_provider_valid(llm.calls)


async def test_messages_to_keep_zero_stays_provider_valid(tmp_path):
    # Regression guard for upstream #296: the post-summarization state was
    # [first message, assistant summary], ending on an assistant turn. The
    # summary now lands as a human-role message, so the next request stays
    # provider-valid even when messages_to_keep=0.
    llm = RecordingChatModel()
    agent = ChatAgent(
        llm=llm,
        workspace=tmp_path,
        tokens_before_summarize=1,
        messages_to_keep=0,
    )

    await agent.ainvoke({
        "messages": [
            HumanMessage(f"q {_PAD}"),
            AIMessage(f"a {_PAD}"),
            HumanMessage("latest question"),
        ]
    })

    assert_requests_provider_valid(llm.calls)


async def test_tool_tail_summarization_stays_provider_valid(tmp_path):
    # Regression guard for upstream #296: a kept tool tail whose calls were
    # summarized away used to be folded into the summary block, leaving the
    # conversation ending on the assistant summary. The human-role summary
    # keeps the request provider-valid even with a nonzero messages_to_keep.
    llm = RecordingChatModel()
    agent = ChatAgent(
        llm=llm,
        workspace=tmp_path,
        tokens_before_summarize=1,
        messages_to_keep=1,
    )

    await agent.ainvoke({
        "messages": [
            HumanMessage(f"q {_PAD}"),
            AIMessage(
                f"calling {_PAD}",
                tool_calls=[{"name": "read_file", "args": {}, "id": "call_1"}],
            ),
            ToolMessage("tool output", tool_call_id="call_1"),
        ]
    })

    assert_requests_provider_valid(llm.calls)
