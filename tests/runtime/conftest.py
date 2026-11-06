"""Reusable host setup with teardown even when a test assertion fails."""

import multiprocessing
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from ursa.cli.config import UrsaConfig
from ursa.runtime import AgentManager, AgentRuntime
from ursa.runtime.agent_runtime import _UrsaProcessAgent


class RecordingAgent:
    def __init__(self, result=None, reset_query=False):
        self.result = result
        self.reset_query = reset_query
        self.seen = []
        self.formatted = []
        self.configs = []

    def format_query(self, prompt, state=None):
        self.formatted.append(prompt)
        if state is None or self.reset_query:
            state = {
                "messages": [SystemMessage(content="Agent instructions")],
                "task": prompt,
            }
        state["messages"].append(HumanMessage(content=prompt))
        return state

    async def ainvoke(self, query, config=None):
        self.seen.append(query)
        self.configs.append(config or {})
        return query

    def format_result(self, state):
        return (
            self.result
            if self.result is not None
            else state["messages"][-1].content
        )


@pytest.fixture
async def process_runner():
    runners = []

    def create(path="root", **options):
        context = SimpleNamespace(
            active_request_id="request",
            _enter_request=lambda request_id: request_id,
            _leave_request=lambda token: None,
        )
        runner = _UrsaProcessAgent(RecordingAgent(**options), context, path)
        runners.append(runner)
        return runner

    yield create
    for runner in runners:
        await runner.aclose()


@pytest.fixture
async def runtime_factory(tmp_path, monkeypatch):
    runtimes = []

    def create(*, manager=None, fork=False, **config):
        options = {}
        if fork:
            if "fork" not in multiprocessing.get_all_start_methods():
                pytest.skip("test fixtures are inherited through fork")
            monkeypatch.setattr(
                "ursa.runtime.agent_runtime.validate_model_provider",
                lambda *_: None,
            )
            options = dict(
                start_method="fork",
                heartbeat_interval=0.05,
                heartbeat_timeout=1,
            )
        manager = manager or AgentManager(**options)
        values = dict(workspace=tmp_path, llm_model={"model": "fake:chat"})
        runtime = AgentRuntime(UrsaConfig(**(values | config)), manager=manager)
        runtimes.append((runtime, manager))
        return runtime, manager

    yield create
    for runtime, manager in reversed(runtimes):
        await runtime.aclose()
        await manager.aclose()
