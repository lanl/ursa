from __future__ import annotations

import asyncio

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ursa import security
from ursa.environments import (
    AgentSymposiumConfig,
    AgentSymposiumEnvironment,
    AgentTeamConfig,
    AgentTeamEnvironment,
    load_team_config,
    save_symposium_config,
    save_team_config,
)
from ursa.environments.base import result_to_text
from ursa.environments.config import (
    EnvironmentMemberConfig,
    load_object,
    symposium_cache_dir,
    team_cache_dir,
)


class RecordingMember:
    prompts: list[str] = []
    kwargs_seen: list[dict] = []

    def __init__(
        self,
        llm=None,
        workspace=None,
        agent_name=None,
        group="default",
        **kwargs,
    ):
        self.llm = llm
        self.workspace = workspace
        self.agent_name = agent_name
        self.group = group
        self.kwargs = kwargs
        self.invocations: list[str] = []
        self.workspaces_seen: list = []
        type(self).kwargs_seen.append(kwargs)

    def invoke(self, prompt, **kwargs):
        prompt = str(prompt)
        self.invocations.append(prompt)
        self.workspaces_seen.append(self.workspace)
        type(self).prompts.append(prompt)
        return {
            "messages": [
                Message(f"{self.agent_name or 'member'} saw {prompt[:40]}")
            ]
        }


class RecordingPI(RecordingMember):
    def add_tool(self, tools):
        self.delegation_tools = tools

    def invoke(self, prompt, **kwargs):
        self.invocations.append(str(prompt))
        self.workspaces_seen.append(self.workspace)
        if getattr(self, "delegation_tools", None):
            delegated = self.delegation_tools[0].invoke({
                "task": "delegated subtask",
                "context": "global context",
            })
            return {"final": f"PI synthesized after delegation: {delegated}"}
        return {"final": f"PI synthesized directly: {str(prompt)[:40]}"}


class RecordingOrganizer(RecordingMember):
    def invoke(self, prompt, **kwargs):
        self.invocations.append(str(prompt))
        self.workspaces_seen.append(self.workspace)
        return {"final": f"organizer synthesis from {str(prompt)[:60]}"}


class AsyncRecordingMember(RecordingMember):
    def invoke(self, prompt, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("async environment path should use ainvoke")

    async def ainvoke(self, prompt, **kwargs):
        prompt = str(prompt)
        self.invocations.append(prompt)
        self.workspaces_seen.append(self.workspace)
        type(self).prompts.append(prompt)
        await asyncio.sleep(0)
        return {
            "messages": [
                Message(
                    f"{self.agent_name or 'member'} async saw {prompt[:40]}"
                )
            ]
        }


class AsyncRecordingPI(AsyncRecordingMember):
    def add_tool(self, tools):
        self.delegation_tools = tools

    async def ainvoke(self, prompt, **kwargs):
        self.invocations.append(str(prompt))
        self.workspaces_seen.append(self.workspace)
        if getattr(self, "delegation_tools", None):
            delegated = await self.delegation_tools[0].ainvoke({
                "task": "delegated async subtask",
                "context": "async global context",
            })
            return {"final": f"async PI synthesized: {delegated}"}
        return {"final": f"async PI synthesized directly: {str(prompt)[:40]}"}


class ConcurrentAsyncMember(AsyncRecordingMember):
    active = 0
    max_active = 0

    async def ainvoke(self, prompt, **kwargs):
        type(self).active += 1
        type(self).max_active = max(type(self).max_active, type(self).active)
        try:
            await asyncio.sleep(0.01)
            return await super().ainvoke(prompt, **kwargs)
        finally:
            type(self).active -= 1


class ConcurrentAsyncOrganizer(ConcurrentAsyncMember):
    async def ainvoke(self, prompt, **kwargs):
        await super().ainvoke(prompt, **kwargs)
        return {"final": f"async organizer synthesis from {str(prompt)[:60]}"}


class Message:
    def __init__(self, text):
        self.text = text


class ToolCapableFakeChatModel(FakeListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def fake_llm():
    return ToolCapableFakeChatModel(responses=["unused"])


def test_load_object_resolves_environment_short_names():
    assert load_object("AgentTeamEnvironment") is AgentTeamEnvironment
    assert load_object("AgentSymposiumEnvironment") is AgentSymposiumEnvironment


def test_environment_default_workspace_uses_group_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(security, "URSA_CACHE_DIR", cache_root)

    team = AgentTeamEnvironment(
        llm=fake_llm(),
        name="workspace_team",
        group="science",
        pi={
            "name": "pi",
            "agent": "tests.environments.test_environments.RecordingPI",
        },
        members=[
            {
                "name": "analyst",
                "agent": "tests.environments.test_environments.RecordingMember",
            }
        ],
        persist_members=True,
    )

    expected_workspace = (
        cache_root
        / "science"
        / "environments"
        / "workspaces"
        / "workspace_team"
    )
    assert team.workspace == expected_workspace
    assert team.workspace.is_dir()
    assert team.pi.workspace == expected_workspace
    assert team.members["analyst"].workspace == expected_workspace


def test_team_pi_receives_member_delegation_tool_and_invokes_member(
    tmp_path, capsys
):
    config = {
        "name": "team_test",
        "pi": {
            "name": "pi",
            "role": "lead",
            "agent": "tests.environments.test_environments.RecordingPI",
        },
        "members": [
            {
                "name": "analyst",
                "role": "analysis",
                "agent": "tests.environments.test_environments.RecordingMember",
            }
        ],
        "workspace": str(tmp_path),
    }
    team = AgentTeamEnvironment(
        llm=fake_llm(), config=config, persist_members=True
    )

    result = team.invoke("solve the user problem")

    assert "PI synthesized after delegation" in result["final"]
    assert "analyst" in team.members
    assert team.pi.agent_name == "pi"
    assert team.members["analyst"].agent_name == "analyst"
    assert team.pi.workspace == tmp_path
    assert team.members["analyst"].workspace == tmp_path
    assert team.members["analyst"].invocations
    assert (
        "Delegated task:\ndelegated subtask"
        in team.members["analyst"].invocations[0]
    )
    assert "User task:\nsolve the user problem" in team.pi.invocations[0]
    assert "clean, coherent, easily shareable answer" in team.pi.invocations[0]
    trace_output = capsys.readouterr().out
    assert "[AgentTeam:team_test] PI -> analyst" in trace_output
    assert "[AgentTeam:team_test] analyst -> PI" in trace_output


def test_team_default_pi_is_execution_agent():
    team = AgentTeamEnvironment(
        llm=fake_llm(),
        name="default_pi_team",
        members=[],
        persist_members=False,
    )
    assert team.config.pi.agent == "ExecutionAgent"


async def test_team_ainvoke_uses_async_pi_and_async_delegation_tool(tmp_path):
    config = {
        "name": "async_team_test",
        "pi": {
            "name": "pi",
            "role": "lead",
            "agent": "tests.environments.test_environments.AsyncRecordingPI",
        },
        "members": [
            {
                "name": "analyst",
                "role": "analysis",
                "agent": "tests.environments.test_environments.AsyncRecordingMember",
            }
        ],
        "workspace": str(tmp_path),
    }
    team = AgentTeamEnvironment(
        llm=fake_llm(), config=config, persist_members=False
    )

    result = await team.ainvoke("solve asynchronously")

    assert "async PI synthesized" in result["final"]
    assert team.members["analyst"].invocations
    assert (
        "Delegated task:\ndelegated async subtask"
        in team.members["analyst"].invocations[0]
    )
    assert "User task:\nsolve asynchronously" in team.pi.invocations[0]


def test_team_invoke_preserves_sync_api_via_async_implementation(tmp_path):
    config = {
        "name": "sync_wrapper_team_test",
        "pi": {
            "name": "pi",
            "role": "lead",
            "agent": "tests.environments.test_environments.AsyncRecordingPI",
        },
        "members": [
            {
                "name": "analyst",
                "role": "analysis",
                "agent": "tests.environments.test_environments.AsyncRecordingMember",
            }
        ],
        "workspace": str(tmp_path),
    }
    team = AgentTeamEnvironment(
        llm=fake_llm(), config=config, persist_members=False
    )

    result = team.invoke("sync caller using async internals")

    assert "async PI synthesized" in result["final"]
    assert team.members["analyst"].invocations


async def test_team_invoke_from_running_loop_raises_clear_error(tmp_path):
    team = AgentTeamEnvironment(
        llm=fake_llm(),
        name="running_loop_team_test",
        members=[],
        persist_members=False,
        workspace=tmp_path,
    )

    try:
        team.invoke("called from async context")
    except RuntimeError as exc:
        assert "await environment.ainvoke" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected .invoke() to reject running event loop")


def test_symposium_runs_independent_review_revision_and_synthesis(tmp_path):
    config = {
        "name": "symposium_test",
        "organizer": {
            "name": "organizer",
            "role": "synth",
            "agent": "tests.environments.test_environments.RecordingOrganizer",
        },
        "members": [
            {
                "name": "alpha",
                "role": "first solver",
                "agent": "tests.environments.test_environments.RecordingMember",
            },
            {
                "name": "beta",
                "role": "second solver",
                "agent": "tests.environments.test_environments.RecordingMember",
            },
        ],
        "revision_rounds": 2,
        "workspace": str(tmp_path),
    }
    symposium = AgentSymposiumEnvironment(
        llm=fake_llm(), config=config, persist_members=False
    )

    result = symposium.invoke("hard user problem")

    assert set(result["initial_writeups"]) == {"alpha", "beta"}
    assert len(result["review_rounds"]) == 2
    assert set(result["reviews"]) == {"alpha", "beta"}
    assert set(result["final_writeups"]) == {"alpha", "beta"}
    assert "organizer synthesis" in result["final"]

    alpha_prompts = symposium.members["alpha"].invocations
    assert "Work independently" in alpha_prompts[0]
    assert "Do not change, edit, overwrite, or reorganize" in alpha_prompts[1]
    assert "You may change only your own work/artifacts" in alpha_prompts[2]


def test_symposium_uses_stable_names_parent_workspace_and_reviewer_flag(
    tmp_path,
):
    config = {
        "name": "soccer_symposium",
        "organizer": {
            "name": "organizer",
            "role": "synth",
            "agent": "tests.environments.test_environments.RecordingOrganizer",
        },
        "members": [
            {
                "name": "nemo_soccer",
                "role": "first solver and reviewer",
                "agent": "tests.environments.test_environments.RecordingMember",
            },
            {
                "name": "honeybee_soccer",
                "role": "second solver only",
                "agent": "tests.environments.test_environments.RecordingMember",
                "reviewer": False,
            },
        ],
        "workspace": str(tmp_path),
    }
    symposium = AgentSymposiumEnvironment(
        llm=fake_llm(), config=config, persist_members=True
    )

    assert symposium.organizer.agent_name == "organizer"
    assert symposium.organizer.workspace == tmp_path
    assert symposium.members["nemo_soccer"].agent_name == "nemo_soccer"
    assert symposium.members["honeybee_soccer"].agent_name == "honeybee_soccer"
    assert (
        symposium.members["nemo_soccer"].workspace == tmp_path / "nemo_soccer"
    )
    assert symposium.members["honeybee_soccer"].workspace == (
        tmp_path / "honeybee_soccer"
    )
    assert symposium.config.members[0].reviewer is True
    assert symposium.config.members[1].reviewer is False

    result = symposium.invoke("hard soccer problem")

    assert set(result["initial_writeups"]) == {
        "nemo_soccer",
        "honeybee_soccer",
    }
    assert set(result["reviews"]) == {"nemo_soccer"}
    assert set(result["final_writeups"]) == {
        "nemo_soccer",
        "honeybee_soccer",
    }

    reviewer = symposium.members["nemo_soccer"]
    non_reviewer = symposium.members["honeybee_soccer"]
    assert reviewer.workspaces_seen == [
        tmp_path / "nemo_soccer",
        tmp_path,
        tmp_path / "nemo_soccer",
    ]
    assert non_reviewer.workspaces_seen == [
        tmp_path / "honeybee_soccer",
        tmp_path / "honeybee_soccer",
    ]
    assert reviewer.workspace == tmp_path / "nemo_soccer"
    assert non_reviewer.workspace == tmp_path / "honeybee_soccer"
    assert "honeybee_soccer/" in reviewer.invocations[1]
    assert "parent symposium workspace" in reviewer.invocations[1]
    assert (
        "Do not change, edit, overwrite, or reorganize"
        in (reviewer.invocations[1])
    )
    assert (
        "Do not change, edit, overwrite, or reorganize"
        not in (non_reviewer.invocations[1])
    )
    assert symposium.organizer.workspaces_seen == [tmp_path]
    assert "honeybee_soccer/" in symposium.organizer.invocations[0]


async def test_symposium_ainvoke_gathers_independent_member_phases(tmp_path):
    ConcurrentAsyncMember.active = 0
    ConcurrentAsyncMember.max_active = 0
    config = {
        "name": "async_symposium_test",
        "organizer": {
            "name": "organizer",
            "role": "synth",
            "agent": "tests.environments.test_environments.ConcurrentAsyncOrganizer",
        },
        "members": [
            {
                "name": "alpha",
                "role": "first solver",
                "agent": "tests.environments.test_environments.ConcurrentAsyncMember",
            },
            {
                "name": "beta",
                "role": "second solver",
                "agent": "tests.environments.test_environments.ConcurrentAsyncMember",
            },
        ],
        "revision_rounds": 1,
        "workspace": str(tmp_path),
    }
    symposium = AgentSymposiumEnvironment(
        llm=fake_llm(), config=config, persist_members=False
    )

    result = await symposium.ainvoke("hard async user problem")

    assert set(result["initial_writeups"]) == {"alpha", "beta"}
    assert len(result["review_rounds"]) == 1
    assert set(result["reviews"]) == {"alpha", "beta"}
    assert set(result["final_writeups"]) == {"alpha", "beta"}
    assert "async organizer synthesis" in result["final"]
    assert type(symposium.members["alpha"]).max_active >= 2


def test_yaml_load_and_save_helpers_round_trip(tmp_path):
    yaml_path = tmp_path / "team.yaml"
    yaml_path.write_text(
        """
name: yaml_team
group: default
pi:
  name: pi
  role: lead
  agent: ExecutionAgent
members:
  - name: analyst
    role: analysis
    agent: ChatAgent
""".strip(),
        encoding="utf-8",
    )

    loaded = load_team_config(yaml_path)
    assert loaded.name == "yaml_team"
    assert loaded.members[0].name == "analyst"

    saved_team = save_team_config(loaded, tmp_path / "saved_team.yaml")
    assert saved_team.exists()
    assert load_team_config(saved_team).name == "yaml_team"

    symposium = AgentSymposiumConfig(
        name="saved_symposium",
        organizer=EnvironmentMemberConfig(name="organizer"),
        members=[EnvironmentMemberConfig(name="member")],
    )
    saved_symposium = save_symposium_config(
        symposium, tmp_path / "saved_symposium.yaml"
    )
    assert saved_symposium.exists()


def test_environment_config_defaults_use_group_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "ursa"
    monkeypatch.setattr(security, "URSA_CACHE_DIR", cache_root)

    team = AgentTeamConfig(
        name="science_team",
        group="science",
        pi=EnvironmentMemberConfig(name="pi"),
        members=[EnvironmentMemberConfig(name="analyst")],
    )
    symposium = AgentSymposiumConfig(
        name="science_symposium",
        group="science",
        organizer=EnvironmentMemberConfig(name="organizer"),
        members=[EnvironmentMemberConfig(name="reviewer")],
    )

    assert team_cache_dir("science", "science_team") == (
        cache_root / "science" / "environments" / "agent_teams" / "science_team"
    )
    assert symposium_cache_dir("science", "science_symposium") == (
        cache_root
        / "science"
        / "environments"
        / "agent_symposia"
        / "science_symposium"
    )

    saved_team = save_team_config(team)
    saved_symposium = save_symposium_config(symposium)

    assert saved_team == team_cache_dir("science", "science_team") / "team.yaml"
    assert saved_team.exists()
    assert saved_symposium == (
        symposium_cache_dir("science", "science_symposium") / "symposium.yaml"
    )
    assert saved_symposium.exists()


def test_result_to_text_extracts_common_result_shapes():
    assert result_to_text({"final": "done"}) == "done"
    assert (
        result_to_text({"messages": [Message("last message")]})
        == "last message"
    )
    assert result_to_text("plain") == "plain"
