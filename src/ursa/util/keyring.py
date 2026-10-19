"""System keyring access with a permission-checked file fallback."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from functools import cache
from pathlib import Path

import keyring
from filelock import FileLock
from keyring.backend import KeyringBackend
from keyring.errors import KeyringError, PasswordDeleteError

_ALLOWED_MODE = 0o600
_HAS_POSIX_FILE_MODES = os.name == "posix"


class InsecureKeyringPermissionsError(KeyringError):
    """Raised when the fallback file grants access beyond its owner."""


class FileKeyring(KeyringBackend):
    """Store keyring entries in a permission-checked JSON file."""

    priority = 0

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = path or Path.home() / ".config" / "ursa" / "keyring.json"

    @property
    def lock_path(self) -> Path:
        """Return the sidecar path used to coordinate fallback updates."""
        return self.path.with_name(f"{self.path.name}.lock")

    @staticmethod
    def _validate_stat(path: Path, result: os.stat_result) -> int:
        mode = stat.S_IMODE(result.st_mode)
        if not stat.S_ISREG(result.st_mode):
            raise KeyringError(
                f"Fallback keyring path is not a regular file: {path}"
            )
        if _HAS_POSIX_FILE_MODES and mode & ~_ALLOWED_MODE:
            raise InsecureKeyringPermissionsError(
                f"Fallback keyring file {path} has permissions {mode:03o}; "
                "permissions must be 600 or more restrictive"
            )
        return mode

    def _write_lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        return FileLock(self.lock_path, mode=_ALLOWED_MODE)

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

        with os.fdopen(descriptor, encoding="utf-8") as stream:
            mode = self._validate_stat(path, os.fstat(stream.fileno()))
            try:
                data = json.load(stream)
            except json.JSONDecodeError as e:
                raise KeyringError(
                    f"Fallback keyring file is not valid JSON: {path}"
                ) from e

        if not isinstance(data, dict) or any(
            not isinstance(passwords, dict)
            or any(
                not isinstance(password, str) for password in passwords.values()
            )
            for passwords in data.values()
        ):
            raise KeyringError(
                f"Fallback keyring file has an invalid entry: {path}"
            )
        return data, mode

    def _write(self, data: dict[str, dict[str, str]], mode: int) -> None:
        path = self.path
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
        )
        temporary_path = Path(temporary_name)
        try:
            if _HAS_POSIX_FILE_MODES:
                os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                descriptor = -1
                json.dump(data, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary_path.unlink(missing_ok=True)

    def get_password(self, service: str, username: str) -> str | None:
        """Return an entry from the fallback keyring, if present."""
        data, _ = self._read()
        return data.get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        """Set an entry in the fallback file."""
        with self._write_lock():
            data, mode = self._read()
            data.setdefault(service, {})[username] = password
            self._write(data, mode)

    def delete_password(self, service: str, username: str) -> None:
        """Delete an entry from the fallback keyring."""
        with self._write_lock():
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


@cache
def get_keyring() -> FallbackKeyring:
    """Return URSA's native-first keyring without changing global state."""
    system = keyring.get_keyring()
    if isinstance(system, FallbackKeyring):
        return system
    return FallbackKeyring(system)
