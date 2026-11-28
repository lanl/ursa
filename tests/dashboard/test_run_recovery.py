"""Restart recovery releases sessions whose active run has ended."""

import pytest

from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.run_manager import RunManager
from ursa_dashboard.sessions import (
    create_session,
    session_paths,
    update_session,
)
from ursa_dashboard.storage import read_json, write_json
from ursa_dashboard.ulid import new_ulid


def session_with_run(manager, status):
    """A session whose active run has the given status, as on disk."""
    root = manager.dashboard_root
    session = create_session(root, agent_id="model_card_agent", title="demo")
    run_id = new_ulid()
    paths = manager._run_paths(agent_id="model_card_agent", run_id=run_id)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        paths.meta_path,
        {
            "run_id": run_id,
            "agent_id": "model_card_agent",
            "status": status,
            "session_id": session["session_id"],
        },
    )
    update_session(root, session["session_id"], {"active_run_id": run_id})
    return session["session_id"], run_id


@pytest.mark.asyncio
async def test_restart_releases_sessions_of_ended_runs(tmp_path):
    manager = RunManager(
        dashboard_root=tmp_path / "dashboard",
        credential_store=MemoryCredentialStore(),
        dashboard_group="default",
    )
    # A run that ended while the dashboard was stopping left its session
    # "running"; the page then replayed the finished run without end.
    stale, stale_run = session_with_run(manager, "failed")
    interrupted, interrupted_run = session_with_run(manager, "running")

    await manager._recover_runs()

    for session_id, run_id in (
        (stale, stale_run),
        (interrupted, interrupted_run),
    ):
        sess = read_json(
            session_paths(manager.dashboard_root, session_id).meta_path
        )
        assert sess["active_run_id"] is None
        assert sess["last_run_id"] == run_id
    assert manager._read_run(interrupted_run)["status"] == "failed"
