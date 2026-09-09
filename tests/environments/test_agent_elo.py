"""Exercise the Elo lifecycle without external model services."""

import asyncio
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from ursa.environments.agent_elo import AgentEloEnvironment
from ursa.environments.agent_elo_judge import AgentEloJudge, JudgeDecision


class LocalMember:
    """Small persistent worker used in place of a live agent."""

    def __init__(self, env, config):
        self.config = config
        self.workspace = env._member_workspace(config.name)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.database = env._member_den(config.name) / "db" / "checkpointer.db"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database) as db:
            db.execute("CREATE TABLE IF NOT EXISTS work (prompt TEXT)")
        self.closed = False
        self.fail = False

    async def ainvoke(self, prompt, **kwargs):
        if self.fail:
            raise RuntimeError("worker failed")
        with sqlite3.connect(self.database) as db:
            db.execute("INSERT INTO work VALUES (?)", (prompt,))
        artifact = self.workspace / "research.txt"
        with artifact.open("a") as stream:
            stream.write(self.config.name + "\n")
        return {"final": "Completed research"}

    def close(self):
        self.closed = True


@pytest.fixture
def elo_factory(monkeypatch, tmp_path):
    created = []
    judgment = SimpleNamespace(winner="A", calls=0)

    def build(env, config):
        member = LocalMember(env, config)
        created.append(member)
        return member

    async def judge(self, **kwargs):
        judgment.calls += 1
        return JudgeDecision(judgment.winner, "Deterministic test judgment")

    monkeypatch.setattr(AgentEloEnvironment, "build_member", build)
    monkeypatch.setattr(AgentEloJudge, "judge_match", judge)
    monkeypatch.setattr(
        "ursa.environments.agent_elo.group_agents_dir",
        lambda group: tmp_path / "dens",
    )

    def make(**overrides):
        config = {
            "name": "lifecycle",
            "workspace": str(tmp_path / "workspace"),
            "seed": 42,
            "deaths_per_round": 2,
            "inference_providers": {
                "local": {
                    "base_url": "http://localhost:1234/v1",
                    "api_key": {"env": "ELO_TEST_API_KEY"},
                    "temperature": 0.7,
                },
            },
            "members": [
                {
                    "name": name,
                    "model": {
                        "model": "openai:test-model",
                        "inference_provider": "local",
                        "temperature": 0.2,
                    },
                }
                for name in ("a", "b", "c", "d")
            ],
        }
        config.update(overrides)
        return AgentEloEnvironment(llm=None, config=config)

    return SimpleNamespace(make=make, created=created, judgment=judgment)


def test_multiple_generations_preserve_population_and_lineage(elo_factory):
    env = elo_factory.make(generations=3)
    result = env.invoke("Improve the research")
    assert result["completed_generations"] == 3
    assert result["ending_generation"] == 3
    assert result["population_size"] == 4
    for generation in result["generations"]:
        assert len(generation["matches"]) == 2
        assert len(generation["children"]) == 2
        assert len(generation["eliminated"]) == 2
        assert generation["population_size"] == 4
        paired = [name for pair in generation["pairs"] for name in pair]
        assert len(set(paired)) == 4
        assert not set(generation["reproducing_parents"]) & set(
            generation["eliminated"]
        )
        for child in generation["children"]:
            row = next(r for r in generation["standings"] if r["name"] == child)
            parent = next(
                r for r in generation["standings"] if r["name"] == row["parent"]
            )
            assert row["rating"] == parent["rating"]
            assert row["generation"] == parent["generation"] + 1
    assert len(env.members) == len(env.players) == len(env.member_configs) == 4


def test_children_inherit_independent_files_and_persistence(elo_factory):
    env = elo_factory.make()
    generation = env.invoke("Research")["generations"][0]
    for child_name in generation["children"]:
        child = env.members[child_name]
        parent = env.members[env.players[child_name].parent]
        assert child.workspace != parent.workspace
        assert child.database != parent.database
        artifact = child.workspace / "research.txt"
        parent_artifact = parent.workspace / "research.txt"
        assert artifact.read_text() == parent_artifact.read_text()
        with sqlite3.connect(child.database) as db:
            assert db.execute("SELECT count(*) FROM work").fetchone() == (1,)
            db.execute("INSERT INTO work VALUES ('child-only work')")
        with sqlite3.connect(parent.database) as db:
            assert db.execute("SELECT count(*) FROM work").fetchone() == (1,)
        artifact.write_text("child-only work")
        assert parent_artifact.read_text() != artifact.read_text()


@pytest.mark.parametrize("legacy_snapshot", [False, True])
def test_restart_preserves_configuration_and_continues_evolution(
    elo_factory, legacy_snapshot
):
    env = elo_factory.make()
    result = env.invoke("Research")
    snapshot = Path(result["environment_state"])
    state = json.loads(snapshot.read_text())
    for player in state["active_players"]:
        model = player["member_config"]["model"]
        assert "inference_provider" not in model
        if legacy_snapshot:
            model["inference_provider"] = "local"
    snapshot.write_text(json.dumps(state))
    restored = elo_factory.make(
        restart_from_json=str(snapshot), members=[], inference_providers={}
    )
    assert restored.standings() == env.standings()
    assert restored.generation_index == env.generation_index
    for member in restored.members.values():
        model = member.config.model
        assert model.model == "test-model"
        assert model.base_url == "http://localhost:1234/v1"
        assert model.api_key.env == "ELO_TEST_API_KEY"
        assert model.model_extra["temperature"] == 0.2
    assert restored._rng.getstate() == env._rng.getstate()
    resumed = restored.invoke("Research")
    assert resumed["starting_generation"] == 1
    assert resumed["ending_generation"] == 2
    assert resumed["population_size"] == 4


@pytest.mark.parametrize("outcome", ["draw", "failure"])
def test_draws_and_member_failures_do_not_change_population(
    elo_factory, outcome
):
    env = elo_factory.make()
    if outcome == "draw":
        elo_factory.judgment.winner = "DRAW"
    else:
        for member in env.members.values():
            member.fail = True
    original = env.standings()
    generation = env.invoke("Research")["generations"][0]
    assert generation["children"] == []
    assert generation["eliminated"] == []
    assert env.standings() == original
    if outcome == "failure":
        assert len(generation["failed"]) == 4
        assert elo_factory.judgment.calls == 0


@pytest.mark.parametrize("stage", ["copy", "fork", "build"])
@pytest.mark.parametrize("failed_child", [1, 2])
def test_reproduction_failure_preserves_population_for_retry(
    elo_factory, monkeypatch, stage, failed_child
):
    env = elo_factory.make()
    original = dict(env.members)
    standings = env.standings()
    method = {
        "copy": "_copy_parent_workspace",
        "fork": "_fork_parent_persistence",
        "build": "build_member",
    }[stage]
    operation = getattr(env, method)
    calls = 0

    def fail(*args):
        nonlocal calls
        calls += 1
        if calls == failed_child:
            raise OSError("reproduction failed")
        return operation(*args)

    with monkeypatch.context() as patch:
        patch.setattr(env, method, fail)
        with pytest.raises(OSError, match="reproduction failed"):
            env.invoke("Research")
    assert env.members == original
    assert env.standings() == standings
    assert all(not member.closed for member in original.values())
    assert all(member.closed for member in elo_factory.created[4:])
    assert {path.name for path in env.workspace.iterdir()} == set(original)
    assert len(list(env._member_den("a").parent.iterdir())) == 4
    result = env.invoke("Research")
    assert result["ending_generation"] == 1
    assert result["population_size"] == 4
    assert len(result["generations"][0]["children"]) == 2


def test_elimination_continues_when_member_cleanup_fails(
    elo_factory, monkeypatch, caplog
):
    env = elo_factory.make(generations=2)
    attempts = []

    for name, member in env.members.items():

        def fail_close(name=name):
            attempts.append(name)
            raise OSError("cleanup failed")

        monkeypatch.setattr(member, "close", fail_close)

    result = env.invoke("Research")
    assert result["ending_generation"] == 2
    assert len(env.members) == len(env.players) == len(env.member_configs) == 4
    assert set(env.members) == set(env.players)
    assert set(env.members) == {m.name for m in env.member_configs}
    assert len(attempts) >= 2
    assert "Failed to close eliminated Elo member" in caplog.text
    snapshot = json.loads(Path(result["environment_state"]).read_text())
    assert {p["name"] for p in snapshot["active_players"]} == set(env.members)


@pytest.mark.parametrize("setting", ["workspace", "agent_name"])
@pytest.mark.parametrize("value", ["shared", None])
def test_new_population_validates_members_before_building(
    elo_factory, setting, value
):
    with pytest.raises(ValueError, match=f"sets config.{setting}"):
        elo_factory.make(
            members=[
                {"name": "a"},
                {"name": "b", "config": {setting: value}},
            ]
        )
    assert elo_factory.created == []


def test_duplicate_names_are_rejected_before_building(elo_factory):
    with pytest.raises(ValueError, match="Elo member name 'a' is duplicated"):
        elo_factory.make(members=[{"name": "a"}, {"name": "a"}])
    assert elo_factory.created == []


def test_restart_rejects_duplicate_names_before_building(elo_factory):
    env = elo_factory.make()
    result = env.invoke("Research")
    path = Path(result["environment_state"])
    state = json.loads(path.read_text())
    state["active_players"][-1] = state["active_players"][0]
    name = state["active_players"][0]["name"]
    path.write_text(json.dumps(state))
    created_before = len(elo_factory.created)
    with pytest.raises(
        ValueError, match=f"Elo member name '{name}' is duplicated"
    ):
        elo_factory.make(restart_from_json=str(path), members=[])
    assert len(elo_factory.created) == created_before


@pytest.mark.parametrize("setting", ["workspace", "agent_name"])
@pytest.mark.parametrize("value", ["shared", None])
def test_restart_validates_saved_members_before_building(
    elo_factory, setting, value
):
    env = elo_factory.make()
    result = env.invoke("Research")
    path = Path(result["environment_state"])
    state = json.loads(path.read_text())
    saved_member = state["active_players"][-1]["member_config"]
    saved_member["config"][setting] = value
    path.write_text(json.dumps(state))
    created_before = len(elo_factory.created)
    with pytest.raises(ValueError, match=f"sets config.{setting}"):
        elo_factory.make(restart_from_json=str(path), members=[])
    assert len(elo_factory.created) == created_before


@pytest.mark.parametrize(
    "failure_stage", ["initialization", "invocation", "parsing"]
)
@pytest.mark.parametrize("fallback_fails", [False, True])
def test_judge_falls_back_on_agent_errors(
    monkeypatch, tmp_path, failure_stage, fallback_fails, caplog
):
    class JudgeAgent:
        def __init__(self, **kwargs):
            if failure_stage == "initialization":
                raise RuntimeError("initialization failed")

        async def ainvoke(self, prompt):
            if failure_stage == "invocation":
                raise RuntimeError("invocation failed")
            return "invalid JSON"

        def close(self):
            # Cleanup must not override a fallback decision either.
            raise OSError("judge cleanup failed")

    class LLM:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            if fallback_fails:
                raise RuntimeError("fallback failed")
            return '{"winner": "B", "reasoning": "better evidence"}'

    monkeypatch.setattr(
        "ursa.environments.agent_elo_judge.ChatAgent", JudgeAgent
    )
    llm = LLM()
    judge = AgentEloJudge(
        llm=llm, workspace=tmp_path, group="default", judge_prompt="Compare"
    )
    decision = asyncio.run(
        judge.judge_match(
            task="Research",
            player_a="a",
            agent_type_a="ExecutionAgent",
            output_a="A",
            player_b="b",
            agent_type_b="ExecutionAgent",
            output_b="B",
        )
    )
    assert llm.calls == 1
    assert decision.winner == ("DRAW" if fallback_fails else "B")
    assert decision.method == (
        "failed_draw" if fallback_fails else "llm_fallback"
    )
    if failure_stage != "initialization":
        assert "Failed to close Elo judge" in caplog.text
