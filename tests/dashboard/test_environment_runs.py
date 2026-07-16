from __future__ import annotations

import json

from fastapi.testclient import TestClient

from ursa import security
from ursa_dashboard.app import create_app


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
