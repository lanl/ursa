from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app


def test_environment_run_api_routes(monkeypatch, tmp_path):
    monkeypatch.setattr(security, "AGENT_GROUPS_DIR", tmp_path / "ursa")
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
            json.dumps({"seq": 1, "event_type": "team_started"}),
            json.dumps({"seq": 2, "event_type": "team_completed"}),
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
    assert detail_response.json()["environment_name"] == "team"

    events_response = client.get("/environment-runs/run-1/events?after_seq=1")
    assert events_response.status_code == 200
    assert [event["seq"] for event in events_response.json()["events"]] == [2]

    ui_response = client.get("/ui/environment-runs")
    assert ui_response.status_code == 200
    assert "Environment Runs" in ui_response.text
