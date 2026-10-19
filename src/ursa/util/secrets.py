"""Secret references and keyring access."""

import json
import os
import stat
import tempfile
from functools import cache
from os import environ
from pathlib import Path
from typing import Annotated, Any

import keyring
from filelock import FileLock
from keyring.backend import KeyringBackend
from keyring.errors import KeyringError, PasswordDeleteError
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    SecretStr,
    ValidationError,
    model_validator,
)

from ursa.util.crossplatform import user_config_paths

_ALLOWED_MODE = 0o600


class InsecureKeyringPermissionsError(KeyringError):
    """Raised when the fallback file grants access beyond its owner."""


class FileKeyring(KeyringBackend):
    """Store keyring entries in a permission-checked JSON file."""

    priority = 0

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = path or user_config_paths()[-1].with_name("keyring.json")

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
        if os.name == "posix" and mode & ~_ALLOWED_MODE:
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
            if os.name == "posix":
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


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("secret reference values cannot be blank")
    return value


class SecretReference(BaseModel):
    """A reference to a secret in the environment or system keyring."""

    model_config = ConfigDict(extra="forbid")

    env: Annotated[str | None, AfterValidator(_non_blank)] = None
    keyring: bool | Annotated[str, AfterValidator(_non_blank)] | None = None

    @classmethod
    def maybe_validate(cls, value: Any, **kwargs) -> Any:
        """Type a secret mapping while leaving unrelated values unchanged."""
        try:
            return cls.model_validate(value, **kwargs)
        except ValidationError:
            return value

    @model_validator(mode="after")
    def _validate_source(self):
        sources = int(self.env is not None) + int(
            self.keyring not in (None, False)
        )
        if sources != 1:
            raise ValueError("a secret reference requires exactly one source")
        return self

    def resolve(
        self, default_keyring_username: str | None = None
    ) -> SecretStr | None:
        """Resolve the reference, returning ``None`` for a missing env value."""
        if self.env is not None:
            value = environ.get(self.env)
            return SecretStr(value) if value is not None else None

        username = (
            default_keyring_username if self.keyring is True else self.keyring
        )
        if not username:
            raise ValueError("keyring=true requires a default username")
        value = get_keyring().get_password("ursa", username)
        if value is None:
            raise ValueError(
                f"No secret found in the system keyring for '{username}'"
            )
        return SecretStr(value)

    def get_secret_value(
        self, default_keyring_username: str | None = None
    ) -> str | None:
        """Resolve and unwrap the referenced secret value."""
        secret = self.resolve(default_keyring_username)
        return secret.get_secret_value() if secret is not None else None


class SecretTemplate(SecretReference):
    """A secret reference rendered into a string at point of use."""

    template: str = "%s"

    @model_validator(mode="after")
    def _validate_template(self):
        if self.template.count("%s") != 1:
            raise ValueError("secret template must contain exactly one '%s'")
        return self

    def get_secret_value(
        self, default_keyring_username: str | None = None
    ) -> str | None:
        secret = super().get_secret_value(default_keyring_username)
        if secret is None:
            return None
        return self.template % secret
