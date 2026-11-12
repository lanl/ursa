import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

pytest.importorskip("harbor")

from ursa.artifacts.retention import restore_artifacts
from ursa.integrations.harbor import app
from ursa.integrations.harbor.retention import (
    RetentionPlugin,
    apply_job_retention,
    load_retention_config,
    plan_job_retention,
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


@pytest.mark.parametrize("whole_trial", [False, True])
def test_cli_preview_apply_restore(tmp_path, whole_trial):
    policy = whole_trial_job(tmp_path) if whole_trial else job(tmp_path)
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


def whole_trial_job(tmp_path):
    policy = job(tmp_path)
    policy.write_text(
        "collections: [trial]\ndefault_action: archive\narchive: {}\n"
    )
    for name in (
        "trial/config.json",
        "trial/lock.json",
        "trial/trial.log",
        "trial/exception.txt",
        "trial/.hidden",
        "trial/agent/db/checkpointer.db",
        "trial/steps/one/agent/trace",
        "trial/steps/two/verifier/reward.txt",
        "trial/steps/two/artifacts/manifest.json",
        "second/agent/trace",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    (tmp_path / "second/result.json").write_bytes(
        (tmp_path / "trial/result.json").read_bytes()
    )
    (tmp_path / "job.log").write_text("job-level log")
    return policy


async def test_whole_trial_plugin_archives_and_restores_all_trial_files(
    tmp_path,
):
    policy = whole_trial_job(tmp_path)
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for name in ("trial", "second")
        for path in (tmp_path / name).rglob("*")
        if path.is_file()
    }
    config = load_retention_config(policy)
    planned = plan_job_retention(tmp_path, config)
    assert planned.roots == ["second", "trial"]
    assert {entry.path for entry in planned.entries} == before.keys()
    assert all(entry.action == "archive" for entry in planned.entries)
    plugin = RetentionPlugin(str(policy))
    await plugin.on_job_start(SimpleNamespace(job_dir=tmp_path))
    await plugin.on_job_end(SimpleNamespace(finished_at="finished"))
    assert all(not (tmp_path / name).exists() for name in before)
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "job.log").read_text() == "job-level log"
    with tarfile.open(tmp_path / "retained-artifacts.tar.xz") as archive:
        assert set(archive.getnames()) == before.keys()
    # A completed run stays discoverable after its trial results are removed.
    assert apply_job_retention(tmp_path, config)["status"] == "complete"
    destination = tmp_path / "restored"
    assert restore_artifacts(tmp_path / "retention.json", destination) == []
    assert {
        path.relative_to(destination).as_posix(): path.read_bytes()
        for path in destination.rglob("*")
        if path.is_file()
    } == before


def test_whole_trial_resume_after_result_removal(tmp_path, monkeypatch):
    config = load_retention_config(whole_trial_job(tmp_path))
    unlink = Path.unlink

    def interrupt(path, *args, **kwargs):
        unlink(path, *args, **kwargs)
        if path == tmp_path / "trial/result.json":
            raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupt)
        with pytest.raises(KeyboardInterrupt):
            apply_job_retention(tmp_path, config)
    assert not (tmp_path / "trial/result.json").exists()
    assert (tmp_path / "trial/trial.log").exists()
    assert apply_job_retention(tmp_path, config)["status"] == "complete"
    assert not (tmp_path / "trial/trial.log").exists()


@pytest.mark.parametrize("collections", ["[artifacts]", "[trial]"])
def test_whole_trial_rejects_unfinished_trials(tmp_path, collections):
    policy = whole_trial_job(tmp_path)
    policy.write_text(
        f"collections: {collections}\ndefault_action: archive\narchive: {{}}\n"
    )
    (tmp_path / "trial/result.json").write_text('{"finished_at": null}')
    with pytest.raises(ValueError, match="finished trial"):
        apply_job_retention(tmp_path, load_retention_config(policy))
    assert (tmp_path / "second/agent/trace").exists()
    assert not (tmp_path / "retained-artifacts.tar.xz").exists()


def test_whole_trial_rules_are_relative_to_trial_directory(tmp_path):
    policy = whole_trial_job(tmp_path)
    policy.write_text(
        "collections: [trial]\n"
        "rules:\n  - match: ['agent/**']\n    action: keep\n"
        "  - match: ['artifacts/data']\n    action: delete\n"
        "default_action: archive\narchive: {}\n"
    )
    config = load_retention_config(policy)
    apply_job_retention(tmp_path, config)
    assert (tmp_path / "trial/agent/trace").exists()
    assert not (tmp_path / "trial/artifacts/data").exists()
    with tarfile.open(tmp_path / "retained-artifacts.tar.xz") as archive:
        assert "trial/agent/trace" not in archive.getnames()
        assert "trial/artifacts/data" not in archive.getnames()
        assert "trial/steps/one/agent/trace" in archive.getnames()
        assert "trial/result.json" in archive.getnames()


def test_cannot_mix_trial_and_payload_collections(tmp_path):
    policy = tmp_path / "policy.yaml"
    policy.write_text("collections: [trial, artifacts]\ndefault_action: keep\n")
    with pytest.raises(ValueError, match="on its own"):
        load_retention_config(policy)


def test_whole_trial_does_not_absorb_new_trials_on_resume(tmp_path):
    config = load_retention_config(whole_trial_job(tmp_path))
    apply_job_retention(tmp_path, config)
    new_trial = tmp_path / "new-trial"
    new_trial.mkdir()
    result = new_trial / "result.json"
    result.write_text('{"finished_at": "finished"}')
    with pytest.raises(ValueError, match="policy/inventory differs"):
        apply_job_retention(tmp_path, config)
    assert result.is_file()


def test_missing_trial_result_before_cleanup_is_not_treated_as_archived(
    tmp_path, monkeypatch
):
    config = load_retention_config(whole_trial_job(tmp_path))

    def fail(*args, **kwargs):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "add", fail)
        with pytest.raises(OSError, match="disk full"):
            apply_job_retention(tmp_path, config)
    (tmp_path / "trial/result.json").unlink()
    with pytest.raises(ValueError, match="finished trial"):
        apply_job_retention(tmp_path, config)
    assert (tmp_path / "trial/agent/trace").is_file()
