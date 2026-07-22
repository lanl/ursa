from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.environment_run_manager import (
    EnvironmentRunManager,
    EnvironmentRunManagerConfig,
    validate_environment_launch,
)


def test_environment_run_api_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")
    run_dir = tmp_path / "ursa" / "default" / "environment_runs" / "run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({
            "run_id": "run-1",
            "group": "default",
            "environment_name": "team",
            "environment_type": "agent_team",
            "status": "succeeded",
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-01-01T00:00:01Z",
            "task_preview": "demo",
        }),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        "\n".join([
            json.dumps({
                "seq": 1,
                "event_type": "topology_declared",
                "payload": {
                    "topology": {
                        "kind": "agent_team",
                        "nodes": [
                            {"id": "team.pi", "name": "PI", "kind": "agent"},
                            {
                                "id": "team.analyst",
                                "name": "analyst",
                                "kind": "agent",
                            },
                        ],
                        "edges": [
                            {
                                "source": "team.pi",
                                "target": "team.analyst",
                                "kind": "delegates_to",
                            }
                        ],
                    }
                },
            }),
            json.dumps({
                "seq": 2,
                "event_type": "team_completed",
                "message": "Team completed",
                "payload": {"result": "final answer", "elapsed_seconds": 1.2},
            }),
        ])
        + "\n",
        encoding="utf-8",
    )

    client = TestClient(create_app())
    list_response = client.get("/environment-runs")
    assert list_response.status_code == 200
    assert list_response.json()["runs"][0]["run_id"] == "run-1"

    detail_response = client.get("/environment-runs/run-1")
    assert detail_response.status_code == 200
    detail_json = detail_response.json()
    assert detail_json["environment_name"] == "team"
    assert detail_json["paths"]["run_dir"] == str(run_dir)
    assert detail_json["paths"]["artifacts_dir"] == str(run_dir / "artifacts")

    events_response = client.get("/environment-runs/run-1/events?after_seq=1")
    assert events_response.status_code == 200
    assert [event["seq"] for event in events_response.json()["events"]] == [2]

    cancel_response = client.post(
        "/environment-runs/run-1/cancel", json={"reason": "test"}
    )
    assert cancel_response.status_code == 409
    assert "launched by this dashboard" in cancel_response.json()["detail"]

    ui_response = client.get("/ui/environment-runs")
    assert ui_response.status_code == 200
    assert "Environment Runs" in ui_response.text
    assert "Open work replay" in ui_response.text

    detail_ui_response = client.get("/ui/environment-runs/run-1")
    assert detail_ui_response.status_code == 200
    assert "Environment Graph" in detail_ui_response.text
    assert "Work Timeline" in detail_ui_response.text
    assert "Current Activity" in detail_ui_response.text
    assert "Final Result" in detail_ui_response.text
    assert "Workspace" in detail_ui_response.text
    assert "Raw Events" in detail_ui_response.text
    assert "Participants" not in detail_ui_response.text
    assert "Inspector" not in detail_ui_response.text
    assert "cytoscape" in detail_ui_response.text


def _team_yaml(name: str = "dashboard_team") -> str:
    return f"""
name: {name}
group: attempted_override
pi:
  name: pi
  role: Lead
  agent: ExecutionAgent
members:
  - name: analyst
    role: Analyst
    agent: ChatAgent
"""


def test_environment_launch_validation_is_group_scoped_and_safe(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    launch = validate_environment_launch(
        "agent_team", _team_yaml(), group="science"
    )
    assert launch.config.group == "science"
    assert launch.config_mapping["group"] == "science"

    unsafe_workspace = _team_yaml().replace(
        "pi:\n", "workspace: ../../outside\npi:\n"
    )
    try:
        validate_environment_launch(
            "agent_team", unsafe_workspace, group="default"
        )
    except ValueError as exc:
        assert "workspace" in str(exc).lower()
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Unsafe workspace was accepted")

    raw_key = _team_yaml().replace(
        "agent: ChatAgent",
        "agent: ChatAgent\n    model:\n      model: openai:test\n      api_key: secret",
    )
    try:
        validate_environment_launch("agent_team", raw_key, group="default")
    except ValueError as exc:
        assert "literal api key" in str(exc).lower()
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Literal API key was accepted")

    custom_class = _team_yaml().replace(
        "agent: ChatAgent", "agent: package.module.CustomAgent"
    )
    try:
        validate_environment_launch("agent_team", custom_class, group="default")
    except ValueError as exc:
        assert "not available from the dashboard" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Custom class was accepted")


def test_dashboard_can_validate_create_and_replace_environment_run(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")
    monkeypatch.setenv(
        "URSA_DASHBOARD_WORKSPACE_ROOT", str(tmp_path / "dashboard")
    )
    monkeypatch.setattr(
        EnvironmentRunManager,
        "validate_credentials",
        lambda *args, **kwargs: None,
    )

    async def do_not_spawn(self, run_id):
        return None

    monkeypatch.setattr(EnvironmentRunManager, "_execute", do_not_spawn)
    payload = {
        "environment_type": "agent_team",
        "config_yaml": _team_yaml("api_team"),
        "prompt": "Analyze the supplied evidence.",
    }
    with TestClient(
        create_app(credential_store=MemoryCredentialStore())
    ) as client:
        validation = client.post(
            "/environment-runs/validate",
            json={
                "environment_type": payload["environment_type"],
                "config_yaml": payload["config_yaml"],
            },
        )
        assert validation.status_code == 200
        assert validation.json()["environment_name"] == "api_team"

        created = client.post("/environment-runs", json=payload)
        assert created.status_code == 201
        record = created.json()
        assert record["status"] == "queued"
        assert record["group"] == "default"
        assert created.headers["location"].endswith(record["run_id"])

        run_dir = (
            tmp_path
            / "ursa"
            / "default"
            / "environment_runs"
            / record["run_id"]
        )
        assert (run_dir / "environment.yaml").exists()
        assert (run_dir / "task.json").exists()
        assert (run_dir / "launch.json").exists()
        assert (
            tmp_path
            / "ursa"
            / "default"
            / "environments"
            / "agent_teams"
            / "api_team"
            / "team.yaml"
        ).exists()

        duplicate = client.post("/environment-runs", json=payload)
        assert duplicate.status_code == 409
        replaced = client.post(
            "/environment-runs", json={**payload, "replace_existing": True}
        )
        assert replaced.status_code == 201

        page = client.get("/ui/environment-runs")
        assert "New team" in page.text
        assert "New symposium" in page.text
        assert "environmentModal" in page.text
        assert "Search environment runs" in page.text


async def test_environment_run_manager_cancels_queued_run(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    manager = EnvironmentRunManager(
        group="default",
        credential_store=MemoryCredentialStore(),
        config=EnvironmentRunManagerConfig(concurrency=0),
    )
    launch = validate_environment_launch(
        "agent_team", _team_yaml("cancel_team"), group="default"
    )
    manifest = await manager.create_run(
        launch=launch,
        prompt="A queued task",
        llm={"model": "none", "credential_source": "none"},
        runner={},
    )
    cancelled = await manager.cancel(manifest["run_id"], reason="test_request")
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_reason"] == "test_request"


async def test_environment_run_manager_marks_interrupted_run_failed(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")
    manager = EnvironmentRunManager(
        group="default",
        credential_store=MemoryCredentialStore(),
        config=EnvironmentRunManagerConfig(concurrency=0),
    )
    launch = validate_environment_launch(
        "agent_team", _team_yaml("recovery_team"), group="default"
    )
    manifest = await manager.create_run(
        launch=launch,
        prompt="An interrupted task",
        llm={"model": "none", "credential_source": "none"},
        runner={},
    )
    manager._update_manifest(manifest["run_id"], {"status": "running"})

    recovered = EnvironmentRunManager(
        group="default",
        credential_store=MemoryCredentialStore(),
        config=EnvironmentRunManagerConfig(concurrency=0),
    )
    await recovered.start()
    result = security.group_root_dir("default") / "environment_runs"
    recovered_manifest = json.loads(
        (result / manifest["run_id"] / "manifest.json").read_text()
    )
    assert recovered_manifest["status"] == "failed"
    assert "restarted" in recovered_manifest["error"].lower()
    await recovered.shutdown()


async def test_environment_worker_builds_team_and_restores_member_secrets(
    monkeypatch, tmp_path
):
    import ursa.environments as environments
    from ursa_dashboard import environment_worker_main

    config_path = tmp_path / "environment.yaml"
    config_path.write_text(_team_yaml("worker_team"), encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps({"prompt": "Worker task"}), encoding="utf-8"
    )
    launch_path = tmp_path / "launch.json"
    launch_path.write_text(
        json.dumps({"llm": {"model": "openai:test"}}), encoding="utf-8"
    )
    seen = {}

    class FakeTeam:
        def __init__(self, *, llm, config):
            seen["llm"] = llm
            seen["config"] = config

    async def fake_run(environment, task, *, run_id):
        seen["environment"] = environment
        seen["task"] = task
        seen["run_id"] = run_id
        seen["member_key"] = __import__("os").environ.get("MEMBER_TEST_KEY")
        return "complete"

    monkeypatch.setattr(
        environment_worker_main, "_init_llm", lambda *a, **k: "llm"
    )
    monkeypatch.setattr(environments, "AgentTeamEnvironment", FakeTeam)
    monkeypatch.setattr(environments, "arun_with_visualization", fake_run)
    monkeypatch.delenv("MEMBER_TEST_KEY", raising=False)
    args = SimpleNamespace(
        run_id="worker-run",
        group="default",
        environment_type="agent_team",
        config_yaml=str(config_path),
        task_json=str(task_path),
        llm_json=str(launch_path),
    )
    result = await environment_worker_main._run(
        args,
        {
            "llm_api_key": "main",
            "member_api_keys": {"MEMBER_TEST_KEY": "member"},
        },
    )
    assert result == "complete"
    assert seen["task"] == "Worker task"
    assert seen["run_id"] == "worker-run"
    assert seen["config"]["group"] == "default"
    assert seen["member_key"] == "member"
    assert __import__("os").environ.get("MEMBER_TEST_KEY") is None
