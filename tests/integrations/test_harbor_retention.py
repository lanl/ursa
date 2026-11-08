import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytest.importorskip("harbor")

from ursa.integrations.harbor import app
from ursa.integrations.harbor.retention import (
    RetentionPlugin,
    apply_job_retention,
    load_retention_config,
)


def job(tmp_path: Path):
    for name in (
        "trial/artifacts/data",
        "trial/verifier/data",
        "trial/steps/one/artifacts/data",
        "trial/steps/two/artifacts/data",
        "trial/verifier/reward.txt",
        "trial/verifier/test-stdout.txt",
        "trial/agent/trace",
        "trial/artifacts/manifest.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    result = {
        "finished_at": "2026-11-06",
        "exception_info": {"exception_type": "AgentTimeoutError"},
    }
    (tmp_path / "result.json").write_text(json.dumps(result))
    (tmp_path / "trial/result.json").write_text(json.dumps(result))
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "retention:\n  collections: [artifacts, verifier]\n  default_action: archive\n  archive: {}\n"
    )
    return policy


async def test_plugin_only_runs_after_job_end_including_failed_trials(tmp_path):
    plugin = RetentionPlugin(str(job(tmp_path)))
    await plugin.on_job_start(SimpleNamespace(job_dir=tmp_path))
    assert (tmp_path / "trial/artifacts/data").exists()
    await plugin.on_job_end(SimpleNamespace(finished_at="finished"))
    assert not (tmp_path / "trial/artifacts/data").exists()
    for name in (
        "trial/result.json",
        "trial/verifier/reward.txt",
        "trial/verifier/test-stdout.txt",
        "trial/agent/trace",
        "trial/artifacts/manifest.json",
    ):
        assert (tmp_path / name).exists()
    assert (
        json.loads((tmp_path / "retention-status.json").read_text())["status"]
        == "complete"
    )


async def test_plugin_records_failure_and_incomplete_job_preserves_files(
    tmp_path,
):
    plugin = RetentionPlugin(str(job(tmp_path)))
    await plugin.on_job_start(SimpleNamespace(job_dir=tmp_path))
    with pytest.raises(ValueError, match="finished"):
        await plugin.on_job_end(SimpleNamespace(finished_at=None))
    assert (tmp_path / "trial/artifacts/data").exists()
    assert (
        json.loads((tmp_path / "retention-status.json").read_text())["status"]
        == "failed"
    )


def test_standalone_requires_finished_trial(tmp_path):
    policy = job(tmp_path)
    (tmp_path / "trial/result.json").write_text('{"finished_at": null}')
    with pytest.raises(ValueError, match="finished trial"):
        apply_job_retention(tmp_path, load_retention_config(policy))


def test_cli_preview_apply_restore(tmp_path):
    policy = job(tmp_path)
    runner = CliRunner()
    args = ["--run-dir", str(tmp_path), "--policy", str(policy)]
    preview = runner.invoke(app, ["artifacts", "plan", *args])
    assert preview.exit_code == 0, preview.output
    assert not (tmp_path / "retention.json").exists()
    assert runner.invoke(app, ["artifacts", "apply", *args]).exit_code == 0
    assert (
        json.loads((tmp_path / "retention-status.json").read_text())["status"]
        == "complete"
    )
    result = runner.invoke(
        app,
        [
            "artifacts",
            "restore",
            "--manifest",
            str(tmp_path / "retention.json"),
            "--destination",
            str(tmp_path / "restored"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        tmp_path / "restored/trial/steps/two/artifacts/data"
    ).read_text() == "trial/steps/two/artifacts/data"


def test_policy_cannot_replace_retention_status(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        "default_action: archive\narchive:\n  path: ./retention-status.json\n"
    )
    with pytest.raises(ValueError, match="conflicts with retention status"):
        load_retention_config(policy)
