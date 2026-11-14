"""Simulator recap accepts mixed histories and keeps error events in-run."""

from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from tests.agents.utils import (
    RecordingChatModel,
    assert_requests_provider_valid,
)
from ursa.agents.simulator_agent import SimulatorAgent
from ursa.util.events import AgentEvents


class FailingModel(RecordingChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        raise ValueError("recap provider failed")


class Events(BaseCallbackHandler):
    def __init__(self):
        self.payloads = []

    def on_custom_event(self, name, data, **kwargs):
        self.payloads.append(data)


def agent(primary, fallback):
    return SimpleNamespace(
        name="simulator",
        llm=primary,
        runner=SimpleNamespace(llm=fallback),
        recap_prompt="Summarize the simulation",
        log_state=False,
        build_config=lambda **kwargs: kwargs,
        events=lambda config: AgentEvents(agent="simulator", config=config),
    )


def history():
    return {
        "messages": [
            HumanMessage("Run the simulation"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "orphan",
                        "name": "run_command",
                        "args": {"query": "ls"},
                    }
                ],
            ),
        ]
    }


def test_recap_sends_plain_transcript_without_dangling_provider_calls():
    llm = RecordingChatModel(response="Simulation recap")
    state = history()
    result = SimulatorAgent._summarize(agent(llm, llm), state)
    assert_requests_provider_valid(llm.calls)
    assert "run_command" in llm.calls[0][-1].content
    assert result["messages"][:-1] == state["messages"]
    assert result["messages"][-1].content == "Simulation recap"


def test_recap_fallback_uses_the_same_safe_transcript():
    primary, fallback = (
        FailingModel(),
        RecordingChatModel(response="Fallback recap"),
    )
    result = SimulatorAgent._summarize(agent(primary, fallback), history())
    assert_requests_provider_valid(primary.calls + fallback.calls)
    assert primary.calls == fallback.calls
    assert result["messages"][-1].content == "Fallback recap"


@pytest.mark.parametrize("in_run", [False, True])
def test_recap_double_failure_returns_errors_without_event_crash(in_run):
    primary, fallback = FailingModel(), FailingModel()
    instance = agent(primary, fallback)
    handler = Events()

    def recap(state, config):
        return SimulatorAgent._summarize(instance, state, config)

    result = (
        RunnableLambda(recap).invoke(history(), {"callbacks": [handler]})
        if in_run
        else SimulatorAgent._summarize(instance, history())
    )
    assert "recap provider failed" in result["messages"][-1].content
    assert len(handler.payloads) == (2 if in_run else 0)
