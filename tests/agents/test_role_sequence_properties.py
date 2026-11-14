"""Property-based coverage for provider-valid message sequences.

Requires the optional ``hypothesis`` test dependency; the module skips
cleanly when it is absent.

Caller contract: generated histories always end on a human message,
because agents receive input through ``format_query`` after a user
turn; a caller-supplied assistant-final history flows straight through
``_normalize_inputs`` and tests caller error, not agent logic.

Formerly known-bug configurations are asserted as passing regression
guards rather than silently avoided: summarization with a plain
``BaseAgent`` (upstream #295, the summarizer now falls back to ``llm``
when an agent has no ``tool_llm``), and the ``messages_to_keep=0`` and
summarized-away tool-tail cases (upstream #296, the summary lands as a
human-role message). The properties therefore explore the remaining
healthy space.
"""

import tempfile
from contextlib import closing

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
        with closing(agent):
            agent.invoke({"messages": list(history)})

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
        with closing(agent):
            agent.invoke({"messages": list(history)})

        assert_requests_provider_valid(llm.calls)


async def test_summarization_runs_on_non_tool_agents(tmp_path):
    # Regression guard for upstream #295: this was a strict xfail pinning
    # the AttributeError on the missing tool_llm.
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


@pytest.mark.parametrize(
    "messages_to_keep", [0, 1], ids=["keep-none", "keep-tool-tail"]
)
async def test_summarized_tails_stay_provider_valid(tmp_path, messages_to_keep):
    # Regression guards for upstream #296: an assistant-role summary formerly
    # left an invalid request when keeping no messages or when the kept tool
    # tail's calls had been summarized away. Both cases need a human summary.
    llm = RecordingChatModel()
    agent = ChatAgent(
        llm=llm,
        workspace=tmp_path,
        tokens_before_summarize=1,
        messages_to_keep=messages_to_keep,
    )
    history = [HumanMessage(f"q {_PAD}")]
    if messages_to_keep:
        history.extend([
            AIMessage(
                f"calling {_PAD}",
                tool_calls=[{"name": "read_file", "args": {}, "id": "call_1"}],
            ),
            ToolMessage("tool output", tool_call_id="call_1"),
        ])
    else:
        history.extend([
            AIMessage(f"a {_PAD}"),
            HumanMessage("latest question"),
        ])

    await agent.ainvoke({"messages": history})

    assert_requests_provider_valid(llm.calls)
