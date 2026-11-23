"""Acceptance tests for the chat prompt on a shared message history.

Pre-registered before the fix. A CLI session keeps one message history for
every agent type: named sessions already did, and unnamed sessions do since
#404. ``ChatAgent.format_query`` writes the chat system prompt into that
history as a message, once, when it is handed no prior state, and the chat
agent never applies it to a request. When another agent wrote to the thread
first, the seed arrives after earlier messages and the history sanitizer,
which keeps only a leading system message, removes it: plain chat then runs
under the first agent's prompt (execution, prompt refinement, the DSI
context) or under none at all (planning, deep review), and without the
skills catalog the chat prompt carries. The seed is also written once, so a
later change to the prompt never reaches the model.

The chat request must lead with the chat prompt whichever agent spoke first,
in unnamed and named sessions alike, and the prompt must be rebuilt for every
request, including the one that follows a tool result. The guards keep what
already works: a caller who chose their own system message keeps it (also
when they continue through ``format_query``), the stored history keeps
whatever leads it, a thread chat started still stores the chat prompt, and no
stray system message is stored or sent.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.agents.utils import (
    RecordingChatModel,
    ScriptedRecordingChatModel,
    assert_requests_provider_valid,
)
from ursa import security
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.execution_agent import ReviewAssessment
from ursa.agents.planning_agent import Plan, PlanStep
from ursa.cli.config import ChatModelConfig, UrsaConfig
from ursa.cli.runtime import HITL

CHAT_PROMPT = "CHAT PROMPT SENTINEL"

# Agents that read and write the session's message history, besides chat.
SHARING_AGENTS = ["plan", "execute", "prompt", "deep_review", "dsi"]


@pytest.fixture(autouse=True)
def _isolate_ursa_cache(monkeypatch, tmp_path_factory):
    # Every session owns an agent directory under the cache.
    monkeypatch.setattr(
        security, "URSA_CACHE_DIR", tmp_path_factory.mktemp("ursa_cache")
    )


@pytest.fixture
def chat_prompt(monkeypatch):
    monkeypatch.setattr(
        "ursa.agents.chat_agent.get_chatter_system_prompt",
        lambda: CHAT_PROMPT,
    )
    return CHAT_PROMPT


def _structured(schema):
    if schema is Plan:
        step = PlanStep(
            name="Stub step",
            description="A stubbed plan step.",
            requires_code=False,
            expected_outputs=["stub"],
            success_criteria=["stub"],
        )
        return Plan(steps=[step])
    if schema is ReviewAssessment:
        return ReviewAssessment(is_complete=True, reason="Stubbed review.")
    raise NotImplementedError(schema)


def _recording_models(monkeypatch):
    """Every chat model the runtime builds, in order."""
    built = []

    def init_chat_model(self):
        built.append(ScriptedRecordingChatModel(structured_factory=_structured))
        return built[-1]

    monkeypatch.setattr(ChatModelConfig, "init_chat_model", init_chat_model)
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.enforce_model_group_policy", lambda *a, **k: None
    )
    return built


async def _turn(hitl, built, agent, prompt):
    """Route one message to ``agent`` and return the requests it sent."""
    if agent not in hitl.agents:
        pytest.skip(f"the {agent} agent is not available in this environment")
    marks = [len(model.calls) for model in built]
    await hitl.run_agent(agent, prompt)
    requests = []
    for index, model in enumerate(built):
        start = marks[index] if index < len(marks) else 0
        requests.extend(model.calls[start:])
    assert requests, f"the {agent} turn made no model call"
    return requests


def _system_contents(messages):
    return [m.content for m in messages if isinstance(m, SystemMessage)]


def _assert_led_by(requests, prompt):
    for request in requests:
        assert isinstance(request[0], SystemMessage)
        assert _system_contents(request) == [prompt]


@pytest.mark.parametrize("first", ["chat", *SHARING_AGENTS])
async def test_chat_leads_with_its_prompt_whichever_agent_spoke_first(
    first, tmp_path, monkeypatch, chat_prompt
):
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, first, "Do something small.")
        requests = await _turn(hitl, built, "chat", "A plain chat message.")
    finally:
        await hitl.aclose()

    _assert_led_by(requests, chat_prompt)


@pytest.mark.parametrize("first", ["plan", "execute"])
async def test_named_session_chat_leads_with_its_prompt(
    first, tmp_path, monkeypatch, chat_prompt
):
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path, agent_name="shared-history"))
    try:
        await _turn(hitl, built, first, "Do something small.")
        requests = await _turn(hitl, built, "chat", "A plain chat message.")
    finally:
        await hitl.aclose()

    _assert_led_by(requests, chat_prompt)


async def test_chat_after_plan_then_execute_runs_under_the_chat_prompt(
    tmp_path, monkeypatch, chat_prompt
):
    # The documented CLI flow: plan, execute the plan, then keep talking.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, "plan", "Write a small workflow.")
        await _turn(hitl, built, "execute", "Execute the plan.")
        requests = await _turn(hitl, built, "chat", "Thanks, what did you do?")
    finally:
        await hitl.aclose()

    _assert_led_by(requests, chat_prompt)


async def test_chat_prompt_is_rebuilt_for_every_request(tmp_path, monkeypatch):
    # The prompt carries the skills catalog, which can change while a session
    # is open; the stored copy from the first turn must not be what is sent.
    current = {"prompt": "CHAT PROMPT ONE"}
    monkeypatch.setattr(
        "ursa.agents.chat_agent.get_chatter_system_prompt",
        lambda: current["prompt"],
    )
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        first = await _turn(hitl, built, "chat", "First message.")
        current["prompt"] = "CHAT PROMPT TWO"
        second = await _turn(hitl, built, "chat", "Second message.")
    finally:
        await hitl.aclose()

    _assert_led_by(first, "CHAT PROMPT ONE")
    _assert_led_by(second, "CHAT PROMPT TWO")


async def test_chat_prompt_leads_every_request_of_a_tool_loop(
    tmp_path, monkeypatch, chat_prompt
):
    # respond, tool node, respond, on a history another agent started: the
    # prompt must reach both chat requests.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, "execute", "Do something small.")
        await hitl.get_agent("chat")
        built[-1].script.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "list_experiences", "args": {}, "id": "call_1"}
                ],
            )
        )
        requests = await _turn(hitl, built, "chat", "Use a tool.")
    finally:
        await hitl.aclose()

    assert len(requests) == 2
    _assert_led_by(requests, chat_prompt)


async def test_chat_first_session_still_stores_the_prompt_it_started_with(
    tmp_path, monkeypatch, chat_prompt
):
    # The stored history keeps its shape: a thread chat started still begins
    # with the chat prompt, so what summarization counts does not change.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, "chat", "A plain chat message.")
        stored = hitl.agents["chat"].state["messages"]
    finally:
        await hitl.aclose()

    assert isinstance(stored[0], SystemMessage)
    assert _system_contents(stored) == [chat_prompt]


@pytest.mark.parametrize("agent_class", [BasicChatAgent, ChatAgent])
def test_a_callers_own_system_message_is_left_alone(
    agent_class, tmp_path, chat_prompt
):
    # Library use without format_query: the caller chose the system message,
    # and keeps it when the returned state continues through format_query.
    model = RecordingChatModel()
    agent = agent_class(llm=model, workspace=tmp_path)
    try:
        state = agent.invoke({
            "messages": [
                SystemMessage(content="CALLER PROMPT"),
                HumanMessage(content="hello"),
            ]
        })
        agent.invoke(agent.format_query("and again", state=state))
    finally:
        agent.close()

    assert len(model.calls) == 2
    _assert_led_by(model.calls, "CALLER PROMPT")


async def test_stored_history_keeps_its_leading_message_when_chat_rewrites_it(
    tmp_path, monkeypatch, chat_prompt
):
    # The chat prompt goes into the request only. Summarization rewrites the
    # stored history, and what leads it (here the executor prompt, which the
    # execution agent's later requests rely on) must survive that.
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, "execute", "Do something small.")
        await _turn(hitl, built, "chat", "First message.")
        chat = hitl.agents["chat"]._agent
        chat.tokens_before_summarize = 1
        chat.messages_to_keep = 1
        await _turn(hitl, built, "chat", "Second message.")
        stored = hitl.agents["chat"].state["messages"]
        executor_prompt = hitl.agents["execute"]._agent.executor_prompt
    finally:
        await hitl.aclose()

    assert any("[Summary" in str(m.content) for m in stored)
    assert isinstance(stored[0], SystemMessage)
    assert _system_contents(stored) == [executor_prompt]


async def test_mixed_session_stores_and_sends_no_stray_system_message(
    tmp_path, monkeypatch, chat_prompt
):
    built = _recording_models(monkeypatch)
    hitl = HITL(UrsaConfig(workspace=tmp_path))
    try:
        await _turn(hitl, built, "execute", "Do something small.")
        await _turn(hitl, built, "chat", "A plain chat message.")
        await _turn(hitl, built, "chat", "Another plain chat message.")
        stored = hitl.agents["chat"].state["messages"]
    finally:
        await hitl.aclose()

    stored_system = [
        index
        for index, message in enumerate(stored)
        if isinstance(message, SystemMessage)
    ]
    assert stored_system in ([], [0])
    assert_requests_provider_valid([
        request for model in built for request in model.calls
    ])
