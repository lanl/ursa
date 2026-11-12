import json
import os
import tarfile
from pathlib import Path

import pytest

from ursa.artifacts import retention
from ursa.artifacts.retention import (
    ArchiveConfig,
    RetentionPolicy,
    apply_retention,
    plan_retention,
    restore_artifacts,
    retain_artifacts,
)


@pytest.fixture
def payloads(tmp_path):
    files = {
        "trial/artifacts/keep.txt": "keep me",
        "trial/artifacts/drop.phdf": "discard",
        "trial/artifacts/nested/drop.phdf": "discard nested",
        "trial/artifacts/nested/data.bin": "compress me" * 100,
        "trial/result.json": "result is protected by scope",
    }
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return tmp_path


@pytest.fixture
def policy():
    return RetentionPolicy(
        rules=[
            {"match": ["keep.txt"], "action": "keep"},
            {"match": ["**/*.phdf"], "action": "delete"},
        ],
        default_action="archive",
        archive=ArchiveConfig(),
    )


def plan(root, policy):
    return plan_retention(root, policy, roots=["trial/artifacts"])


def test_round_trip_and_repeat(payloads, policy):
    before = {
        p.relative_to(payloads): p.read_bytes()
        for p in payloads.rglob("*")
        if p.is_file()
    }
    planned = plan(payloads, policy)
    assert not (payloads / "retention.json").exists()
    assert {
        p.relative_to(payloads): p.read_bytes()
        for p in payloads.rglob("*")
        if p.is_file()
    } == before
    actions = {e.path: e.action for e in planned.entries}
    assert actions["trial/artifacts/drop.phdf"] == "delete"
    assert actions["trial/artifacts/nested/drop.phdf"] == "delete"
    assert actions["trial/artifacts/keep.txt"] == "keep"
    state = apply_retention(planned)
    assert state["status"] == "complete"
    assert len(state["removed"]) == 3
    journal = (payloads / "retention.json").read_bytes()
    repeated = retain_artifacts(payloads, policy, roots=["trial/artifacts"])
    assert (payloads / "retention.json").read_bytes() == journal
    assert repeated["removed"] == state["removed"]
    assert repeated["archive"] == state["archive"]
    restored = payloads / "restored"
    lost = restore_artifacts(payloads / "retention.json", restored)
    assert set(lost) == {
        "trial/artifacts/drop.phdf",
        "trial/artifacts/nested/drop.phdf",
    }
    assert (
        restored / "trial/artifacts/nested/data.bin"
    ).read_bytes() == before[Path("trial/artifacts/nested/data.bin")]
    assert (payloads / "trial/result.json").read_bytes() == before[
        Path("trial/result.json")
    ]


@pytest.mark.parametrize(
    "failure", ["compression", "duplicate", "missing", "unexpected"]
)
def test_archive_failure_preserves_every_source(
    payloads, policy, monkeypatch, failure
):
    planned = plan(payloads, policy)
    original = tarfile.TarFile.add

    def broken_add(archive, path, *, arcname, recursive):
        if failure == "compression":
            raise OSError("disk full")
        if failure == "duplicate":
            original(archive, path, arcname=arcname, recursive=recursive)
        if failure != "missing":
            name = (
                arcname + ".unexpected" if failure == "unexpected" else arcname
            )
            original(archive, path, arcname=name, recursive=recursive)

    error = OSError if failure == "compression" else ValueError
    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "add", broken_add)
        with pytest.raises(error):
            apply_retention(planned)
    assert all((payloads / e.path).exists() for e in planned.entries)
    assert (
        json.loads((payloads / "retention.json").read_text())["status"]
        == "failed"
    )
    assert apply_retention(planned)["status"] == "complete"


def test_resume_interrupted_cleanup(payloads, policy, monkeypatch):
    planned = plan(payloads, policy)
    original = Path.unlink
    removed = []

    def interrupted(path, *args, **kwargs):
        if path.suffix == ".phdf":
            removed.append(path)
            if len(removed) == 2:
                raise KeyboardInterrupt()
        return original(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", interrupted)
        with pytest.raises(KeyboardInterrupt):
            apply_retention(planned)
    assert not removed[0].exists()
    assert removed[1].exists()
    assert (payloads / policy.archive.path).is_file()
    assert apply_retention(planned)["status"] == "complete"


def test_recover_archive_published_before_journal_update(
    payloads, policy, monkeypatch
):
    planned = plan(payloads, policy)
    original = retention._publish_archive

    def interrupted(root, plan, state):
        original(root, plan, state)
        state["archive"] = None
        raise KeyboardInterrupt()

    with monkeypatch.context() as patch:
        patch.setattr(retention, "_publish_archive", interrupted)
        with pytest.raises(KeyboardInterrupt):
            apply_retention(planned)
    assert all((payloads / e.path).exists() for e in planned.entries)
    assert apply_retention(planned)["status"] == "complete"


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_changed_inventory_refuses_cleanup(payloads, policy, change):
    planned = plan(payloads, policy)
    target = payloads / "trial/artifacts/nested/data.bin"
    if change == "modify":
        target.write_text("new contents")
    elif change == "add":
        (target.parent / "extra").touch()
    else:
        target.unlink()
    with pytest.raises(ValueError):
        apply_retention(planned)
    assert (payloads / "trial/artifacts/drop.phdf").exists()


def test_archive_collision_is_not_overwritten(payloads, policy):
    planned = plan(payloads, policy)
    archive = payloads / policy.archive.path
    archive.write_text("unrelated")
    with pytest.raises(FileExistsError):
        apply_retention(planned)
    assert archive.read_text() == "unrelated"
    assert all((payloads / e.path).exists() for e in planned.entries)


def test_corrupt_published_archive_cannot_resume_or_restore(payloads, policy):
    planned = plan(payloads, policy)
    apply_retention(planned)
    (payloads / policy.archive.path).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Published archive changed"):
        apply_retention(planned)
    with pytest.raises(ValueError, match="checksum"):
        restore_artifacts(payloads / "retention.json", payloads / "restored")


@pytest.mark.parametrize("target", ["nested/data.bin", "./nested/data.bin"])
def test_links_are_not_followed_and_hardlinks_roundtrip(
    payloads, policy, target
):
    directory = payloads / "trial/artifacts"
    outside = payloads / "outside.txt"
    outside.write_text("outside")
    (directory / "external-link").symlink_to(outside)
    (directory / "relative-link").symlink_to(target)
    os.link(directory / "nested/data.bin", directory / "hard-link")
    state = apply_retention(plan(payloads, policy))
    assert outside.read_text() == "outside"
    with tarfile.open(payloads / state["archive"]["path"]) as archive:
        assert archive.getmember("trial/artifacts/external-link").issym()
        assert archive.getmember("trial/artifacts/hard-link").isfile()
    with pytest.raises(tarfile.FilterError):
        restore_artifacts(payloads / "retention.json", payloads / "restored")
    assert not (payloads / "restored").exists()


def test_keep_archived_originals_and_protected_files(payloads):
    policy = RetentionPolicy(
        default_action="archive", archive=ArchiveConfig(remove_originals=False)
    )
    planned = plan_retention(
        payloads,
        policy,
        roots=["trial/artifacts"],
        protected=["trial/artifacts/keep.txt"],
    )
    state = apply_retention(planned)
    assert state["removed"] == []
    assert all((payloads / e.path).exists() for e in planned.entries)
    with tarfile.open(payloads / state["archive"]["path"]) as archive:
        assert "trial/artifacts/keep.txt" not in archive.getnames()


@pytest.mark.parametrize(
    "archive",
    [
        "../escape.tar.xz",
        "/tmp/out.tar.xz",
        "retention.json",
        ".retention.lock",
    ],
)
def test_reject_unsafe_archive_paths(archive):
    with pytest.raises(ValueError):
        RetentionPolicy(archive=ArchiveConfig(path=archive))


def test_symlinked_root_and_nested_output_are_rejected(payloads, policy):
    (payloads / "linked").symlink_to(payloads / "trial/artifacts")
    with pytest.raises(ValueError, match="real directory"):
        plan_retention(payloads, policy, roots=["linked"])
    unsafe = policy.model_copy(
        update={"archive": ArchiveConfig(path="trial/artifacts/out.tar.xz")}
    )
    with pytest.raises(ValueError, match="outside artifact roots"):
        plan(payloads, unsafe)
