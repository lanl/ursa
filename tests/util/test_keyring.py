import os
import stat
from pathlib import Path

import pytest
from keyring.backends.fail import Keyring as FailingKeyring
from keyring.backends.null import Keyring as NullKeyring
from keyring.errors import KeyringError

from ursa.util.secrets import FallbackKeyring, FileKeyring


def test_file_keyring_crud_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "config/ursa/keyring.json"
    keyring = FileKeyring(path)
    keyring.set_password("ursa", "openai", "token")
    assert keyring.get_password("ursa", "openai") == "token"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    keyring.delete_password("ursa", "openai")
    assert keyring.get_password("ursa", "openai") is None

    path.chmod(0o600)
    cases = [('{"ursa":', "not valid JSON"), ("[]", "invalid entry")]
    for contents, error in cases:
        path.write_text(contents)
        path.chmod(0o600)
        with pytest.raises(KeyringError, match=error):
            keyring.get_password("ursa", "openai")


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX file security")
def test_file_security(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.json"
    FileKeyring(target).set_password("ursa", "openai", "token")
    target.chmod(0o700)
    with pytest.raises(KeyringError, match="permissions 700"):
        FileKeyring(target).get_password("ursa", "openai")

    target.chmod(0o600)
    with monkeypatch.context() as patch:
        patch.setattr(os, "getuid", lambda: target.stat().st_uid + 1)
        with pytest.raises(KeyringError, match="owned by uid"):
            FileKeyring(target).get_password("ursa", "openai")

    link = tmp_path / "keyring.json"
    link.symlink_to(target)
    with pytest.raises(KeyringError, match="could not be opened"):
        FileKeyring(link).get_password("ursa", "openai")


def test_fallback_selection(tmp_path: Path) -> None:
    system = FileKeyring(tmp_path / "system.json")
    fallback = FileKeyring(tmp_path / "unused-fallback.json")
    system.set_password("ursa", "native", "native-token")
    fallback.set_password("ursa", "file", "file-token")
    native_first = FallbackKeyring(system, fallback)
    assert native_first.get_password("ursa", "native") == "native-token"

    for index, unavailable in enumerate((NullKeyring(), FailingKeyring())):
        file = FileKeyring(tmp_path / f"fallback-{index}.json")
        keyring = FallbackKeyring(unavailable, file)
        keyring.set_password("ursa", "openai", "file-token")
        assert keyring.get_password("ursa", "openai") == "file-token"
    system.set_password("ursa", "openai", "stale-system-token")
    assert (
        FallbackKeyring(system, file).get_password("ursa", "openai")
        == "file-token"
    )
