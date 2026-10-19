from __future__ import annotations

import stat
from pathlib import Path

import pytest
import yaml
from keyring.backend import KeyringBackend
from keyring.errors import KeyringError, PasswordDeleteError

from ursa.util.keyring import (
    FallbackKeyring,
    FileKeyring,
    InsecureKeyringPermissionsError,
    get_keyring,
)


class WorkingKeyring(KeyringBackend):
    priority = 0

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        del self.values[(service, username)]


class FailingKeyring(KeyringBackend):
    priority = 0

    @staticmethod
    def _fail() -> None:
        raise RuntimeError("system keyring unavailable")

    def get_password(self, service: str, username: str) -> str | None:
        self._fail()

    def set_password(self, service: str, username: str, password: str) -> None:
        self._fail()

    def delete_password(self, service: str, username: str) -> None:
        self._fail()


def file_mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_file_keyring_creates_yaml_with_mode_600(tmp_path: Path) -> None:
    path = tmp_path / "config" / "ursa" / "keyring.yml"
    keyring = FileKeyring(path)

    keyring.set_password("ursa", "openai", "token")
    keyring.set_password("ursa-ai", "dashboard:default:llm", "other")

    assert file_mode(path) == 0o600
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {
        "ursa": {"openai": "token"},
        "ursa-ai": {"dashboard:default:llm": "other"},
    }
    assert keyring.get_password("ursa", "openai") == "token"
    assert keyring.get_password("ursa", "missing") is None
    assert isinstance(keyring, KeyringBackend)


def test_file_keyring_uses_ursa_config_path_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    keyring = FileKeyring()
    keyring.set_password("ursa", "openai", "token")

    assert keyring.path == tmp_path / ".config" / "ursa" / "keyring.yml"
    assert keyring.path.exists()


def test_file_keyring_accepts_more_restrictive_permissions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.yml"
    path.write_text("ursa:\n  openai: token\n", encoding="utf-8")
    path.chmod(0o400)

    assert FileKeyring(path).get_password("ursa", "openai") == "token"


@pytest.mark.parametrize("mode", [0o700, 0o640, 0o604, 0o666])
def test_file_keyring_rejects_more_permissive_permissions(
    tmp_path: Path, mode: int
) -> None:
    path = tmp_path / "keyring.yml"
    path.write_text("ursa:\n  openai: token\n", encoding="utf-8")
    path.chmod(mode)
    keyring = FileKeyring(path)

    with pytest.raises(
        InsecureKeyringPermissionsError,
        match=f"permissions {mode:03o}",
    ):
        keyring.get_password("ursa", "openai")
    with pytest.raises(InsecureKeyringPermissionsError):
        keyring.set_password("ursa", "anthropic", "token")
    with pytest.raises(InsecureKeyringPermissionsError):
        keyring.delete_password("ursa", "openai")


def test_file_keyring_preserves_restrictive_mode_when_updating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.yml"
    keyring = FileKeyring(path)
    keyring.set_password("ursa", "openai", "old")
    path.chmod(0o400)

    keyring.set_password("ursa", "openai", "new")

    assert keyring.get_password("ursa", "openai") == "new"
    assert file_mode(path) == 0o400


def test_file_keyring_delete_removes_password(tmp_path: Path) -> None:
    keyring = FileKeyring(tmp_path / "keyring.yml")
    keyring.set_password("ursa", "openai", "token")

    keyring.delete_password("ursa", "openai")

    assert keyring.get_password("ursa", "openai") is None
    with pytest.raises(PasswordDeleteError):
        keyring.delete_password("ursa", "openai")


def test_file_keyring_rejects_invalid_yaml(tmp_path: Path) -> None:
    path = tmp_path / "keyring.yml"
    path.write_text("ursa: [", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(KeyringError, match="not valid YAML"):
        FileKeyring(path).get_password("ursa", "openai")


def test_fallback_keyring_prefers_working_system_keyring(
    tmp_path: Path,
) -> None:
    system = WorkingKeyring()
    fallback_path = tmp_path / "keyring.yml"
    keyring = FallbackKeyring(system, FileKeyring(fallback_path))

    keyring.set_password("ursa", "openai", "system-token")

    assert keyring.get_password("ursa", "openai") == "system-token"
    assert not fallback_path.exists()
    assert isinstance(keyring, KeyringBackend)


def test_fallback_keyring_does_not_override_empty_system_result(
    tmp_path: Path,
) -> None:
    fallback = FileKeyring(tmp_path / "keyring.yml")
    fallback.set_password("ursa", "openai", "file-token")
    keyring = FallbackKeyring(WorkingKeyring(), fallback)

    assert keyring.get_password("ursa", "openai") is None


def test_fallback_keyring_uses_file_when_system_operations_fail(
    tmp_path: Path,
) -> None:
    fallback = FileKeyring(tmp_path / "keyring.yml")
    keyring = FallbackKeyring(FailingKeyring(), fallback)

    keyring.set_password("ursa", "openai", "file-token")

    assert keyring.get_password("ursa", "openai") == "file-token"
    keyring.delete_password("ursa", "openai")
    assert fallback.get_password("ursa", "openai") is None


def test_fallback_keyring_rejects_insecure_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "keyring.yml"
    path.write_text("ursa:\n  openai: token\n", encoding="utf-8")
    path.chmod(0o644)
    keyring = FallbackKeyring(FailingKeyring(), FileKeyring(path))

    with pytest.raises(InsecureKeyringPermissionsError):
        keyring.get_password("ursa", "openai")


def test_get_keyring_wraps_without_replacing_system_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import keyring

    system = WorkingKeyring()
    monkeypatch.setattr("keyring.core._keyring_backend", system)

    configured = get_keyring()

    assert keyring.get_keyring() is system
    assert configured.system is system
    assert get_keyring() is configured
