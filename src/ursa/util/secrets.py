"""Secret references and keyring access."""

import json
import os
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from functools import cache
from os import environ
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, TextIO

import keyring
from filelock import FileLock
from keyring.backend import KeyringBackend
from keyring.backends.null import Keyring as NullKeyring
from keyring.errors import (
    InitError,
    KeyringError,
    NoKeyringError,
    PasswordDeleteError,
)
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    SecretStr,
    ValidationError,
    model_validator,
)

from ursa.util.crossplatform import default_user_config_path
from ursa.util.fileio import atomic_write_text

_ALLOWED_MODE = 0o600
_SECRET_NAME_SECTIONS = {("inference_providers",), ("mcp_servers",)}
KEYRING_SERVICE = "ursa"


class FileKeyring(KeyringBackend):
    """Store keyring entries in a permission-checked JSON file."""

    priority = 0

    def __init__(self, path: Path | None = None) -> None:
        super().__init__()
        self.path = path or default_user_config_path().with_name("keyring.json")

    def _write_lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        return FileLock(f"{self.path}.lock", mode=_ALLOWED_MODE)

    @property
    def selection_path(self) -> Path:
        return self.path.with_suffix(f"{self.path.suffix}.selected")

    def _selection_lock(self) -> FileLock:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        return FileLock(f"{self.selection_path}.lock", mode=_ALLOWED_MODE)

    def mark_selected(self) -> None:
        """Persist that URSA selected the file backend."""
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        descriptor = os.open(
            self.selection_path,
            os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            _ALLOWED_MODE,
        )
        os.close(descriptor)

    def _error(self, message: str) -> KeyringError:
        return KeyringError(f"Keyring file {self.path} {message}")

    def _open(self) -> TextIO:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags)
        except FileNotFoundError:
            raise
        except OSError as e:
            raise self._error(f"could not be opened: {e}") from e
        try:
            result = os.fstat(descriptor)
            mode = stat.S_IMODE(result.st_mode)
            if not stat.S_ISREG(result.st_mode):
                raise self._error("is not a regular file")
            if hasattr(os, "getuid"):
                current_uid = os.getuid()
                if result.st_uid != current_uid:
                    raise self._error(
                        "is owned by uid "
                        f"{result.st_uid}; current uid is {current_uid}"
                    )
            if os.name == "posix" and mode & ~_ALLOWED_MODE:
                raise self._error(
                    "has permissions "
                    f"{mode:03o}; permissions must be 600 or more restrictive"
                )
            stream = os.fdopen(descriptor, "r", encoding="utf-8")
        except BaseException:
            os.close(descriptor)
            raise
        return stream

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            with self._open() as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            raise self._error("is not valid JSON") from e

        valid = isinstance(data, dict) and all(
            isinstance(passwords, dict)
            and all(
                isinstance(password, str) for password in passwords.values()
            )
            for passwords in data.values()
        )
        if not valid:
            raise self._error("has an invalid entry")
        return data

    def _write(self, data: dict[str, dict[str, str]]) -> None:
        mode = _ALLOWED_MODE
        with suppress(FileNotFoundError):
            with self._open() as current:
                mode = stat.S_IMODE(os.fstat(current.fileno()).st_mode)

        text = f"{json.dumps(data, indent=2, sort_keys=True)}\n"
        atomic_write_text(self.path, text, mode)

    @contextmanager
    def _edit(self) -> Iterator[dict[str, dict[str, str]]]:
        with self._write_lock():
            data = self._read()
            yield data
            self._write(data)

    def get_password(self, service: str, username: str) -> str | None:
        """Return an entry from the keyring, if present."""
        data = self._read()
        return data.get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        """Set an entry in the keyring file."""
        with self._edit() as data:
            data.setdefault(service, {})[username] = password

    def delete_password(self, service: str, username: str) -> None:
        """Delete an entry from the keyring file."""
        with self._edit() as data:
            passwords = data.get(service)
            if passwords is None or username not in passwords:
                raise PasswordDeleteError("Password not found")
            del passwords[username]
            if not passwords:
                del data[service]


class FallbackKeyring(KeyringBackend):
    """Use one native keyring, falling back permanently when unavailable."""

    priority = 0

    def __init__(
        self,
        system: KeyringBackend,
        fallback: KeyringBackend | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.fallback = FileKeyring() if fallback is None else fallback
        self._backend = (
            self.fallback
            if isinstance(system, NullKeyring)
            or isinstance(self.fallback, FileKeyring)
            and self.fallback.selection_path.exists()
            else self.system
        )
        self._lock = Lock()

    @contextmanager
    def _operation_lock(self) -> Iterator[None]:
        with self._lock:
            if isinstance(self.fallback, FileKeyring):
                with self.fallback._selection_lock():
                    yield
            else:
                yield

    def _call(self, method: str, *args: str) -> Any:
        with self._operation_lock():
            if (
                isinstance(self.fallback, FileKeyring)
                and self.fallback.selection_path.exists()
            ):
                self._backend = self.fallback
            backend = self._backend
            if backend is self.fallback and isinstance(
                self.fallback, FileKeyring
            ):
                self.fallback.mark_selected()
            try:
                return getattr(backend, method)(*args)
            except (InitError, NoKeyringError):
                if backend is self.fallback:
                    raise
                self._backend = self.fallback
                if isinstance(self.fallback, FileKeyring):
                    self.fallback.mark_selected()
                return getattr(self.fallback, method)(*args)

    def get_password(self, service: str, username: str) -> str | None:
        return self._call("get_password", service, username)

    def set_password(self, service: str, username: str, password: str) -> None:
        self._call("set_password", service, username, password)

    def delete_password(self, service: str, username: str) -> None:
        self._call("delete_password", service, username)


@cache
def get_keyring() -> FallbackKeyring:
    """Return URSA's native-first keyring without changing global state."""
    return FallbackKeyring(keyring.get_keyring())


def _non_blank(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("secret reference values cannot be blank")
    return value


class SecretReference(BaseModel):
    """A reference to a secret in the environment or keyring."""

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

        username = self.keyring_username(default_keyring_username)
        if not username:
            raise ValueError("keyring=true requires a default username")
        value = get_keyring().get_password(KEYRING_SERVICE, username)
        if value is None:
            raise ValueError(f"No secret found in the keyring for '{username}'")
        return SecretStr(value)

    def keyring_username(
        self, default_username: str | None = None
    ) -> str | None:
        """Return the effective keyring username, if this source uses one."""
        if self.keyring in (None, False):
            return None
        if isinstance(self.keyring, str):
            return self.keyring
        return default_username

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


def iter_secret_references(
    value: Any,
    path: tuple[str, ...] = (),
    default_username: str | None = None,
) -> Iterator[tuple[tuple[str, ...], SecretReference, str | None]]:
    """Yield secret references found in a loaded configuration value."""
    value = SecretTemplate.maybe_validate(value)
    if isinstance(value, SecretReference):
        yield path, value, default_username
    elif isinstance(value, BaseModel):
        fields = {
            name: getattr(value, name) for name in type(value).model_fields
        }
        fields.update(value.model_extra or {})
        yield from iter_secret_references(fields, path, default_username)
    elif isinstance(value, Mapping):
        if isinstance(value.get("inference_provider"), str):
            default_username = value["inference_provider"]
        elif isinstance(value.get("model"), str) and ":" in value["model"]:
            default_username = value["model"].split(":", 1)[0]
        for name, item in value.items():
            if name == "api_key_env" and isinstance(item, str):
                reference = SecretReference(env=item)
                yield (*path, "api_key"), reference, default_username
                continue
            child_default = default_username
            if path in _SECRET_NAME_SECTIONS or child_default is None:
                child_default = str(name)
            yield from iter_secret_references(
                item, (*path, str(name)), child_default
            )
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, item in enumerate(value):
            yield from iter_secret_references(
                item, (*path, str(index)), default_username
            )


def keyring_usernames(value: Any) -> list[str]:
    """Return the unique keyring usernames referenced by a config value."""
    return sorted({
        username
        for _, reference, default in iter_secret_references(value)
        if (username := reference.keyring_username(default)) is not None
    })
