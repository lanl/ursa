"""System keyring access with a permission-checked file fallback."""

from __future__ import annotations

import os
import stat
import tempfile
import threading
from collections.abc import Mapping
from pathlib import Path

import keyring
import yaml
from keyring.backend import KeyringBackend
from keyring.errors import KeyringError, PasswordDeleteError

_DEFAULT_KEYRING_PATH = Path("~/.config/ursa/keyring.yml")
_ALLOWED_MODE = 0o600
_file_lock = threading.RLock()


class InsecureKeyringPermissionsError(KeyringError):
    """Raised when the fallback file grants access beyond its owner."""


class FileKeyring(KeyringBackend):
    """Store keyring entries in a YAML file readable only by its owner."""

    priority = 0

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self._path = path

    @property
    def path(self) -> Path:
        """Return the configured path, expanding the current user's home."""
        return (
            self._path
            if self._path is not None
            else _DEFAULT_KEYRING_PATH.expanduser()
        )

    @staticmethod
    def _validate_stat(path: Path, result: os.stat_result) -> int:
        mode = stat.S_IMODE(result.st_mode)
        if not stat.S_ISREG(result.st_mode):
            raise KeyringError(
                f"Fallback keyring path is not a regular file: {path}"
            )
        if mode & ~_ALLOWED_MODE:
            raise InsecureKeyringPermissionsError(
                f"Fallback keyring file {path} has permissions {mode:03o}; "
                "permissions must be 600 or more restrictive"
            )
        return mode

    def _read(self) -> tuple[dict[str, dict[str, str]], int]:
        path = self.path
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except FileNotFoundError:
            return {}, _ALLOWED_MODE
        except OSError as e:
            raise KeyringError(
                f"Could not open fallback keyring file {path}: {e}"
            ) from e

        try:
            mode = self._validate_stat(path, os.fstat(descriptor))
            with os.fdopen(descriptor, encoding="utf-8") as stream:
                descriptor = -1
                try:
                    raw = yaml.safe_load(stream)
                except yaml.YAMLError as e:
                    raise KeyringError(
                        f"Fallback keyring file is not valid YAML: {path}"
                    ) from e
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        if raw is None:
            return {}, mode
        if not isinstance(raw, Mapping):
            raise KeyringError(
                f"Fallback keyring file must contain a mapping: {path}"
            )

        data: dict[str, dict[str, str]] = {}
        for service, passwords in raw.items():
            if not isinstance(service, str) or not isinstance(
                passwords, Mapping
            ):
                raise KeyringError(
                    f"Fallback keyring file has an invalid entry: {path}"
                )
            entries: dict[str, str] = {}
            for username, password in passwords.items():
                if not isinstance(username, str) or not isinstance(
                    password, str
                ):
                    raise KeyringError(
                        f"Fallback keyring file has an invalid entry: {path}"
                    )
                entries[username] = password
            data[service] = entries
        return data, mode

    def _write(self, data: dict[str, dict[str, str]], mode: int) -> None:
        path = self.path
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
        )
        temporary_path = Path(temporary_name)
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                yaml.safe_dump(data, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)

    def get_password(self, service: str, username: str) -> str | None:
        """Return an entry from the fallback keyring, if present."""
        with _file_lock:
            data, _ = self._read()
            return data.get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        """Set an entry, creating the fallback file with mode 0600."""
        with _file_lock:
            data, mode = self._read()
            data.setdefault(service, {})[username] = password
            self._write(data, mode)

    def delete_password(self, service: str, username: str) -> None:
        """Delete an entry from the fallback keyring."""
        with _file_lock:
            data, mode = self._read()
            passwords = data.get(service)
            if passwords is None or username not in passwords:
                raise PasswordDeleteError("Password not found")
            del passwords[username]
            if not passwords:
                del data[service]
            self._write(data, mode)


class FallbackKeyring(KeyringBackend):
    """Use a native keyring first and a file when the native operation fails."""

    priority = 0

    def __init__(
        self,
        system: KeyringBackend,
        fallback: KeyringBackend | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.fallback = FileKeyring() if fallback is None else fallback

    def get_password(self, service: str, username: str) -> str | None:
        try:
            return self.system.get_password(service, username)
        except Exception:
            return self.fallback.get_password(service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        try:
            self.system.set_password(service, username, password)
        except Exception:
            self.fallback.set_password(service, username, password)

    def delete_password(self, service: str, username: str) -> None:
        try:
            self.system.delete_password(service, username)
        except Exception:
            self.fallback.delete_password(service, username)


_cached_keyring: FallbackKeyring | None = None


def get_keyring() -> FallbackKeyring:
    """Return URSA's native-first keyring without changing global state."""
    system = keyring.get_keyring()
    if isinstance(system, FallbackKeyring):
        return system

    global _cached_keyring
    if _cached_keyring is None or _cached_keyring.system is not system:
        _cached_keyring = FallbackKeyring(system)
    return _cached_keyring
