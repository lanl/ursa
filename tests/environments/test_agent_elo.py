"""Exercise the Elo lifecycle without external model services."""

import asyncio
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ursa.environments.agent_elo import AgentEloEnvironment, MemberRunResult
from ursa.environments.agent_elo_judge import AgentEloJudge, JudgeDecision


class LocalMember:
    """Small persistent worker used in place of a live agent."""

    def __init__(self, env, config):
        self.config = config
        self.workspace = env._member_workspace(config.name)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.database = env._member_den(config.name) / "db" / "checkpointer.db"
        self.database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.database)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS work (prompt TEXT)")
        self.closed = False
        self.fail = False

    async def ainvoke(self, prompt, **kwargs):
        if self.fail:
            raise RuntimeError("worker failed")
        with closing(sqlite3.connect(self.database)) as db, db:
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
    judgment = SimpleNamespace(winner="A", calls=0, submissions=[])

    def build(env, config):
        member = LocalMember(env, config)
        created.append(member)
        return member

    async def judge(self, **kwargs):
        judgment.calls += 1
        judgment.submissions.append(kwargs)
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
        with closing(sqlite3.connect(child.database)) as db, db:
            assert db.execute("SELECT count(*) FROM work").fetchone() == (1,)
            db.execute("INSERT INTO work VALUES ('child-only work')")
        with closing(sqlite3.connect(parent.database)) as db, db:
            assert db.execute("SELECT count(*) FROM work").fetchone() == (1,)
        artifact.write_text("child-only work")
        assert parent_artifact.read_text() != artifact.read_text()


@pytest.mark.parametrize("backup_fails", [False, True])
def test_persistence_backup_releases_database_handles(
    monkeypatch, tmp_path, backup_fails
):
    source = tmp_path / "parent.db"
    destination = tmp_path / "child.db"
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("CREATE TABLE research (value TEXT)")
        db.execute("INSERT INTO research VALUES ('inherited work')")

    class Connection(sqlite3.Connection):
        def backup(self, target, **kwargs):
            if backup_fails:
                raise sqlite3.OperationalError("backup failed")
            return super().backup(target, **kwargs)

    connect = sqlite3.connect
    connections = []

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, factory=Connection, **kwargs)
        # Retain references so garbage collection cannot hide leaked handles.
        connections.append(connection)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", tracked_connect)
        if backup_fails:
            with pytest.raises(sqlite3.OperationalError, match="backup failed"):
                AgentEloEnvironment._backup_sqlite_database(source, destination)
        else:
            AgentEloEnvironment._backup_sqlite_database(source, destination)

    try:
        assert len(connections) == 2
        for connection in connections:
            with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                connection.execute("SELECT 1")
        if not backup_fails:
            with closing(connect(destination)) as db:
                assert db.execute("SELECT value FROM research").fetchone() == (
                    "inherited work",
                )
        destination.unlink()
        source.unlink()
    finally:
        for connection in connections:
            connection.close()


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


@pytest.mark.parametrize("with_report", [False, True])
@pytest.mark.parametrize("both_timeout", [False, True])
def test_timeouts_are_judged_using_available_work(
    elo_factory, monkeypatch, with_report, both_timeout
):
    env = elo_factory.make(
        members=[{"name": "a"}, {"name": "b"}],
        member_timeout_seconds=1,
        deaths_per_round=1,
    )
    timed_names = {"a", "b"} if both_timeout else {"b"}
    for name in timed_names:
        member = env.members[name]

        async def work(prompt, member=member, **kwargs):
            if with_report:
                report = member.workspace / "_elo_progress" / "generation_1.md"
                report.parent.mkdir(parents=True)
                report.write_text("Verified result: 0.746824", encoding="utf-8")
            await asyncio.Event().wait()

        monkeypatch.setattr(member, "ainvoke", work)

    async def choose_timeout(**kwargs):
        elo_factory.judgment.submissions.append(kwargs)
        return JudgeDecision(
            "A" if kwargs["player_a"] == "b" else "B", "Stronger saved evidence"
        )

    monkeypatch.setattr(env.judge, "judge_match", choose_timeout)
    generation = env.invoke("Compare numerical accuracy")["generations"][0]
    assert set(generation["timed_out"]) == timed_names
    assert generation["matches"][0]["winner"] == "b"
    assert generation["reproducing_parents"] == ["b"]
    submission = elo_factory.judgment.submissions[0]
    for side in ("a", "b"):
        name = submission[f"player_{side}"]
        text = submission[f"output_{side}"]
        if name in timed_names:
            assert "Execution status: timed_out" in text
            assert generation["outputs"][name] is None
            report = generation["member_runs"][name]["progress_report"]
            if with_report:
                assert report == "Verified result: 0.746824"
                assert report in text
            else:
                assert report is None
                assert "Briefly inspect" in text
        else:
            assert "Final response:" in text


@pytest.mark.parametrize("final_response", ["Final result", ""])
def test_submission_prefers_final_response_then_current_report(
    elo_factory, monkeypatch, final_response
):
    env = elo_factory.make()
    member = env.member_configs[0]
    report_dir = env._member_workspace(member.name) / "_elo_progress"
    report_dir.mkdir()
    (report_dir / "generation_0.md").write_text("Old inherited report")
    assert env._read_progress_report(member.name) is None
    report = "verified evidence " * 600
    (report_dir / "generation_1.md").write_text(report, encoding="utf-8")

    async def work(*args, **kwargs):
        return {"final": final_response}

    monkeypatch.setattr(env.members[member.name], "ainvoke", work)
    run = asyncio.run(env._run_member(member, "task", {}))
    submission = env._judge_submission(run)
    if final_response:
        assert final_response in submission
        assert "verified evidence" not in submission
    else:
        assert run.progress_report == report.strip()
        assert report.strip() in submission
    # Capture does not change when the file is modified later.
    (report_dir / "generation_1.md").write_text("Later update")
    assert env._judge_submission(run) == submission


@pytest.mark.parametrize(
    "status_a,status_b,score",
    [
        ("completed", "failed", 1.0),
        ("failed", "completed", 0.0),
        ("timed_out", "failed", 0.5),
        ("failed", "timed_out", 0.5),
        ("failed", "failed", 0.5),
    ],
)
def test_execution_failures_keep_existing_match_rules(
    elo_factory, status_a, status_b, score
):
    env = elo_factory.make()
    result = asyncio.run(
        env._resolve_match(
            task="task",
            player_a="a",
            player_b="b",
            run_a=MemberRunResult(
                "a", status_a, "Result" if status_a == "completed" else None
            ),
            run_b=MemberRunResult(
                "b", status_b, "Result" if status_b == "completed" else None
            ),
        )
    )
    assert result.score_a == score
    assert elo_factory.judgment.calls == 0


def test_prompts_share_task_criteria_and_preserve_lineage_guidance(elo_factory):
    env = elo_factory.make()
    member = env.member_configs[0]
    task = "Evaluate accuracy first, then reproducibility."
    founding = env._member_prompt(member, task)
    assert "independent approach" in founding
    assert "Deadline" not in founding
    env.generation_index = 1
    surviving = env._member_prompt(member, task)
    assert "Favor refinement" in surviving
    env.players[member.name].parent = "ancestor"
    deadline = datetime.now(timezone.utc) + timedelta(seconds=30)
    descendant = env._member_prompt(member, task, deadline=deadline)
    assert "alternative approach" in descendant
    assert "terminated at this deadline" in descendant
    assert "generation_2.md" in descendant
    assert "If you can write files" in descendant
    assert "500" not in descendant
    assert task in descendant
    assert env.judge_prompt == ""
    prompt = env.judge._judge_prompt(
        task=task,
        player_a="a",
        player_b="b",
        agent_type_a="ExecutionAgent",
        agent_type_b="ExecutionAgent",
        output_a="result",
        output_b="partial",
    )
    assert task in prompt
    assert "task's evaluation criteria" in prompt
    assert "briefly inspect" in prompt
    assert "timeout alone is not a loss" in prompt
    assert '"winner"' in prompt


def test_expired_deadline_captures_progress_without_invoking_member(
    elo_factory, monkeypatch
):
    env = elo_factory.make()
    member = env.member_configs[0]
    report = (
        env._member_workspace(member.name)
        / env._progress_report_relative_path()
    )
    report.parent.mkdir()
    report.write_text("Saved progress")

    async def unexpected(*args, **kwargs):
        pytest.fail("Expired deadline must not start member execution")

    monkeypatch.setattr(env.members[member.name], "ainvoke", unexpected)
    run = asyncio.run(
        env._run_member(
            member,
            "task",
            {},
            deadline=datetime.now(timezone.utc) - timedelta(seconds=1),
        )
    )
    assert run.timed_out
    assert run.progress_report == "Saved progress"
