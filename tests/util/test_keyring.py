from __future__ import annotations

import json
import multiprocessing
import os
import stat
from pathlib import Path

import pytest
from filelock import FileLock
from keyring.backends.fail import Keyring as FailingSystemKeyring
from keyring.backends.null import Keyring as NullSystemKeyring
from keyring.errors import KeyringError, PasswordDeleteError

from ursa.util.secrets import (
    FallbackKeyring,
    FileKeyring,
    InsecureKeyringPermissionsError,
    get_keyring,
)

_POSIX_TEST = pytest.mark.skipif(
    os.name != "posix", reason="Requires POSIX file security"
)


def _set_password_in_process(
    path: Path,
    started: multiprocessing.synchronize.Event,
) -> None:
    started.set()
    FileKeyring(path).set_password("ursa", "openai", "token")


def test_file_keyring_creates_pretty_json_with_mode_600(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config" / "ursa" / "keyring.json"
    keyring = FileKeyring(path)

    keyring.set_password("ursa", "openai", "token")
    keyring.set_password("ursa-ai", "dashboard:default:llm", "other")

    expected = {
        "ursa": {"openai": "token"},
        "ursa-ai": {"dashboard:default:llm": "other"},
    }
    assert path.read_text(encoding="utf-8") == (
        f"{json.dumps(expected, indent=2, sort_keys=True)}\n"
    )
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_file_keyring_uses_highest_priority_user_config_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_home = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    assert FileKeyring().path == config_home / "ursa/keyring.json"


@pytest.mark.parametrize(
    ("mode", "allowed"),
    [(0o400, True), (0o600, True), (0o700, False), (0o604, False)],
)
@_POSIX_TEST
def test_file_keyring_checks_permissions(
    tmp_path: Path, mode: int, allowed: bool
) -> None:
    path = tmp_path / "keyring.json"
    path.write_text('{"ursa":{"openai":"token"}}\n', encoding="utf-8")
    path.chmod(mode)
    keyring = FileKeyring(path)

    if allowed:
        assert keyring.get_password("ursa", "openai") == "token"
    else:
        with pytest.raises(
            InsecureKeyringPermissionsError,
            match=f"permissions {mode:03o}",
        ):
            keyring.get_password("ursa", "openai")


@_POSIX_TEST
def test_file_keyring_preserves_restrictive_mode_when_updating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.json"
    keyring = FileKeyring(path)
    keyring.set_password("ursa", "openai", "old")
    path.chmod(0o400)

    keyring.set_password("ursa", "openai", "new")

    assert keyring.get_password("ursa", "openai") == "new"
    assert stat.S_IMODE(path.stat().st_mode) == 0o400


@_POSIX_TEST
def test_file_keyring_rejects_file_owned_by_another_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "keyring.json"
    FileKeyring(path).set_password("ursa", "openai", "token")
    monkeypatch.setattr(os, "getuid", lambda: path.stat().st_uid + 1)

    with pytest.raises(InsecureKeyringPermissionsError, match="owned by uid"):
        FileKeyring(path).get_password("ursa", "openai")


@_POSIX_TEST
def test_file_keyring_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    FileKeyring(target).set_password("ursa", "openai", "token")
    path = tmp_path / "keyring.json"
    path.symlink_to(target)

    with pytest.raises(KeyringError, match="Could not open"):
        FileKeyring(path).get_password("ursa", "openai")


def test_file_keyring_locks_updates_across_processes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.json"
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    process = context.Process(
        target=_set_password_in_process, args=(path, started)
    )

    try:
        with FileLock(FileKeyring(path).lock_path, mode=0o600):
            process.start()
            assert started.wait(timeout=10)
            process.join(timeout=0.5)
            assert process.is_alive()

        process.join(timeout=10)
    finally:
        if process.is_alive():
            process.terminate()
            process.join()

    assert process.exitcode == 0
    assert FileKeyring(path).get_password("ursa", "openai") == "token"


def test_file_keyring_delete_removes_password(tmp_path: Path) -> None:
    keyring = FileKeyring(tmp_path / "keyring.json")
    keyring.set_password("ursa", "openai", "token")

    keyring.delete_password("ursa", "openai")

    assert keyring.get_password("ursa", "openai") is None
    with pytest.raises(PasswordDeleteError):
        keyring.delete_password("ursa", "openai")


@pytest.mark.parametrize(
    ("contents", "error"),
    [
        ('{"ursa":', "not valid JSON"),
        ("[]", "invalid entry"),
        ('{"ursa":{"openai":1}}', "invalid entry"),
    ],
)
def test_file_keyring_rejects_invalid_contents(
    tmp_path: Path, contents: str, error: str
) -> None:
    path = tmp_path / "keyring.json"
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(KeyringError, match=error):
        FileKeyring(path).get_password("ursa", "openai")


def test_fallback_keyring_prefers_working_system_keyring(
    tmp_path: Path,
) -> None:
    fallback_path = tmp_path / "fallback.json"
    keyring = FallbackKeyring(
        FileKeyring(tmp_path / "system.json"), FileKeyring(fallback_path)
    )

    keyring.set_password("ursa", "openai", "system-token")

    assert keyring.get_password("ursa", "openai") == "system-token"
    assert not fallback_path.exists()


def test_fallback_keyring_does_not_override_empty_system_result(
    tmp_path: Path,
) -> None:
    fallback = FileKeyring(tmp_path / "keyring.json")
    fallback.set_password("ursa", "openai", "file-token")
    keyring = FallbackKeyring(NullSystemKeyring(), fallback)

    assert keyring.get_password("ursa", "openai") is None


def test_fallback_keyring_uses_file_when_system_operations_fail(
    tmp_path: Path,
) -> None:
    fallback = FileKeyring(tmp_path / "keyring.json")
    keyring = FallbackKeyring(FailingSystemKeyring(), fallback)

    keyring.set_password("ursa", "openai", "file-token")

    assert keyring.get_password("ursa", "openai") == "file-token"
    keyring.delete_password("ursa", "openai")
    assert fallback.get_password("ursa", "openai") is None


def test_get_keyring_wraps_without_replacing_system_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import keyring

    system = NullSystemKeyring()
    monkeypatch.setattr("keyring.core._keyring_backend", system)
    get_keyring.cache_clear()

    try:
        configured = get_keyring()

        assert keyring.get_keyring() is system
        assert configured.system is system
        assert get_keyring() is configured
    finally:
        get_keyring.cache_clear()
