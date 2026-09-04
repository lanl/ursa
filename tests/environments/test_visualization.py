from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ursa import security
from ursa.environments import (
    AgentSymposiumEnvironment,
    AgentTeamEnvironment,
    arun_with_visualization,
    environment_run_recorder,
    run_with_visualization,
)
from ursa.environments.visualization import (
    EnvironmentEventRecorder,
    read_environment_run_events,
    read_environment_run_manifest,
)


class Message:
    def __init__(self, text: str):
        self.text = text


class ToolCapableFakeChatModel(FakeListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def fake_llm():
    return ToolCapableFakeChatModel(responses=["unused"])


class VisualMember:
    invoke_kwargs: list[dict[str, Any]] = []
    ainvoke_kwargs: list[dict[str, Any]] = []

    def __init__(self, llm=None, workspace=None, agent_name=None, **kwargs):
        self.agent_name = agent_name or kwargs.get("name") or "member"

    def invoke(self, prompt, **kwargs):
        type(self).invoke_kwargs.append(kwargs)
        return {"messages": [Message(f"{self.agent_name}: {str(prompt)[:30]}")]}

    async def ainvoke(self, prompt, **kwargs):
        type(self).ainvoke_kwargs.append(kwargs)
        return {"messages": [Message(f"{self.agent_name}: {str(prompt)[:30]}")]}


class VisualPI(VisualMember):
    def add_tool(self, tools):
        self.delegation_tools = tools

    def invoke(self, prompt, **kwargs):
        delegated = self.delegation_tools[0].invoke({
            "task": "delegated task",
            "context": "delegated context",
        })
        return {"final": f"pi final: {delegated}"}

    async def ainvoke(self, prompt, **kwargs):
        delegated = await self.delegation_tools[0].ainvoke({
            "task": "delegated task",
            "context": "delegated context",
        })
        return {"final": f"pi final: {delegated}"}


class VisualOrganizer(VisualMember):
    async def ainvoke(self, prompt, **kwargs):
        type(self).ainvoke_kwargs.append(kwargs)
        return {"final": f"organizer final: {str(prompt)[:30]}"}


@pytest.fixture(autouse=True)
def reset_visual_fakes():
    VisualMember.invoke_kwargs = []
    VisualMember.ainvoke_kwargs = []
    yield


def test_run_with_visualization_records_team_delegation_and_forwards_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    team = AgentTeamEnvironment(
        llm=fake_llm(),
        name="visual_team",
        group="default",
        pi={
            "name": "pi",
            "role": "lead",
            "agent": "tests.environments.test_visualization.VisualPI",
        },
        members=[
            {
                "name": "analyst",
                "role": "analysis",
                "agent": "tests.environments.test_visualization.VisualMember",
            }
        ],
        trace_delegation=False,
    )

    result = run_with_visualization(
        team, {"task": "solve visually"}, run_id="team-run"
    )

    assert "pi final" in result["final"]
    manifest = read_environment_run_manifest("default", "team-run")
    assert manifest["status"] == "succeeded"
    events = read_environment_run_events("default", "team-run")
    event_types = {event["event_type"] for event in events}
    assert "topology_declared" in event_types
    assert "delegation_started" in event_types
    assert "delegation_completed" in event_types
    delegated_kwargs = team.members["analyst"].__class__.ainvoke_kwargs[-1]
    assert "config" in delegated_kwargs
    callbacks = delegated_kwargs["config"].get("callbacks", [])
    assert any(
        isinstance(callback, EnvironmentEventRecorder) for callback in callbacks
    )
    metadata = delegated_kwargs["config"].get("metadata", {})
    assert metadata["environment_id"] == "visual_team"
    assert metadata["environment_member"] == "analyst"
    assert metadata["environment_member_id"] == "visual_team.analyst"
    assert metadata["environment_member_role"] == "analysis"
    assert metadata["environment_member_path"] == ["visual_team", "analyst"]
    assert metadata["agent"] == "analyst"
    assert metadata["agent_id"] == "visual_team.analyst"


@pytest.mark.asyncio
async def test_arun_with_visualization_records_symposium_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    symposium = AgentSymposiumEnvironment(
        llm=fake_llm(),
        name="visual_symposium",
        group="default",
        organizer={
            "name": "organizer",
            "role": "synthesis",
            "agent": "tests.environments.test_visualization.VisualOrganizer",
        },
        members=[
            {
                "name": "alpha",
                "role": "first view",
                "agent": "tests.environments.test_visualization.VisualMember",
            },
            {
                "name": "beta",
                "role": "second view",
                "agent": "tests.environments.test_visualization.VisualMember",
            },
        ],
        revision_rounds=1,
    )

    result = await arun_with_visualization(
        symposium, {"task": "compare approaches"}, run_id="sym-run"
    )

    assert result["final"].startswith("organizer final")
    events = read_environment_run_events("default", "sym-run", limit=10000)
    event_types = {event["event_type"] for event in events}
    assert "symposium_started" in event_types
    assert "initial_work_started" in event_types
    assert "review_round_started" in event_types
    assert "revision_round_completed" in event_types
    assert "synthesis_completed" in event_types
    assert "symposium_completed" in event_types


def test_environment_run_recorder_marks_success_on_normal_exit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    team = AgentTeamEnvironment(
        llm=fake_llm(),
        name="context_team",
        group="default",
        pi={
            "name": "pi",
            "agent": "tests.environments.test_visualization.VisualMember",
        },
        members=[],
    )

    with environment_run_recorder(team, task="manual", run_id="context-run"):
        pass

    manifest = read_environment_run_manifest("default", "context-run")
    assert manifest["status"] == "succeeded"


def test_recorder_normalizes_tool_owner_as_source_and_tool_as_target(
    tmp_path: Path,
) -> None:
    recorder = EnvironmentEventRecorder(
        run_id="manual-run",
        group="default",
        environment_name="manual_env",
        environment_type="agent_team",
        run_dir=tmp_path / "manual-run",
    )

    attributed = recorder.normalize_event({
        "tool": "run_web_search",
        "tool_call_id": "call-1",
        "agent": "analyst",
        "agent_id": "team.analyst",
        "environment_member": "analyst",
        "environment_member_id": "team.analyst",
        "environment_member_path": ["team", "analyst"],
        "stage": "search",
        "message": "Searching Web",
    })
    assert attributed["source"] == {
        "id": "team.analyst",
        "name": "analyst",
        "kind": "agent",
        "path": ["team", "analyst"],
    }
    assert attributed["target"] == {
        "id": "call-1",
        "name": "run_web_search",
        "kind": "tool",
        "path": ["run_web_search"],
    }

    unattributed = recorder.normalize_event({
        "tool": "run_web_search",
        "tool_call_id": "call-2",
        "stage": "search",
        "message": "Searching Web",
    })
    assert unattributed["source"]["kind"] == "tool"
    assert unattributed["source"]["id"] == "call-2"
    assert unattributed["target"] is None


def test_recorder_normalizes_non_json_payload_and_truncates(
    tmp_path: Path,
) -> None:
    recorder = EnvironmentEventRecorder(
        run_id="manual-run",
        group="default",
        environment_name="manual_env",
        environment_type="agent_team",
        run_dir=tmp_path / "manual-run",
        max_payload_chars=8,
    )
    recorder.on_custom_event(
        "ursa_agent_progress",
        {
            "environment": "manual_env",
            "stage": "manual",
            "message": "x" * 20,
            "path_value": tmp_path,
            "tags": "single-tag",
        },
        run_id="lc-run",
    )

    event = json.loads((tmp_path / "manual-run" / "events.jsonl").read_text())
    assert event["seq"] == 1
    assert event["payload"]["message"].startswith("xxxxxxxx")
    assert event["payload"]["path_value"] == str(tmp_path)
    assert event["tags"] == ["single-tag"]
