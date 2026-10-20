"""Secret references and keyring access."""

import json
import os
import stat
from collections.abc import Generator, Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext, suppress
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
        self.selection_path = Path(f"{self.path}.selected")

    def _lock(self, selection: bool = False) -> FileLock:
        self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        path = self.selection_path if selection else self.path
        return FileLock(f"{path}.lock", mode=_ALLOWED_MODE)

    def mark_selected(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        os.close(os.open(self.selection_path, flags, _ALLOWED_MODE))

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
                problem = "is not a regular file"
            elif hasattr(os, "getuid") and result.st_uid != os.getuid():
                problem = f"is owned by uid {result.st_uid}; current uid is {os.getuid()}"
            elif os.name == "posix" and mode & ~_ALLOWED_MODE:
                problem = f"has permissions {mode:03o}; permissions must be 600 or more restrictive"
            else:
                return os.fdopen(descriptor, "r", encoding="utf-8")
            raise self._error(problem)
        except BaseException:
            os.close(descriptor)
            raise

    def _read(self) -> dict[str, dict[str, str]]:
        try:
            with self._open() as stream:
                data = json.load(stream)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as e:
            raise self._error("is not valid JSON") from e

        if not isinstance(data, dict) or any(
            not isinstance(passwords, dict)
            or any(not isinstance(value, str) for value in passwords.values())
            for passwords in data.values()
        ):
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
    def _edit(self) -> Generator[dict[str, dict[str, str]], None, None]:
        with self._lock():
            data = self._read()
            yield data
            self._write(data)

    def get_password(self, service: str, username: str) -> str | None:
        return self._read().get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        with self._edit() as data:
            data.setdefault(service, {})[username] = password

    def delete_password(self, service: str, username: str) -> None:
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
        self._file = (
            self.fallback if isinstance(self.fallback, FileKeyring) else None
        )
        selected = self._file is not None and self._file.selection_path.exists()
        self._backend = (
            self.fallback
            if isinstance(system, NullKeyring) or selected
            else system
        )
        self._lock = Lock()

    def _call(self, method: str, *args: str) -> Any:
        selection_lock = (
            self._file._lock(selection=True) if self._file else nullcontext()
        )
        with self._lock, selection_lock:
            if self._file and self._file.selection_path.exists():
                self._backend = self.fallback
            backend = self._backend
            if backend is self.fallback and self._file:
                self._file.mark_selected()
            try:
                return getattr(backend, method)(*args)
            except (InitError, NoKeyringError):
                if backend is self.fallback:
                    raise
                self._backend = self.fallback
                if self._file:
                    self._file.mark_selected()
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
        if isinstance(self.keyring, str):
            return self.keyring
        return default_username if self.keyring else None

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
        yield from iter_secret_references(dict(value), path, default_username)
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
            child_default = (
                str(name) if path in _SECRET_NAME_SECTIONS else default_username
            )
            child_default = child_default or str(name)
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
