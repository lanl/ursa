"""References to secrets stored outside configuration files."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from os import environ
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    SecretStr,
    ValidationError,
    model_validator,
)


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

        import keyring

        username = (
            default_keyring_username if self.keyring is True else self.keyring
        )
        if not username:
            raise ValueError("keyring=true requires a default username")
        value = keyring.get_password("ursa", username)
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


SecretPath = tuple[str, ...]
SecretTransform = Callable[[SecretReference, SecretPath, str | None], Any]
LiteralSecretTransform = Callable[[SecretStr, SecretPath], Any]


def transform_secret_references(
    value: Any,
    transform: SecretTransform,
    *,
    transform_literal: LiteralSecretTransform | None = None,
    path: SecretPath = (),
    default_username: str | None = None,
) -> Any:
    """Transform every secret in a loaded config using URSA's naming rules."""
    reference = (
        SecretTemplate.maybe_validate(value)
        if isinstance(value, SecretTemplate)
        or isinstance(value, Mapping)
        and "template" in value
        else SecretReference.maybe_validate(value)
    )
    if isinstance(reference, SecretReference):
        return transform(reference, path, default_username)

    if isinstance(value, SecretStr):
        return transform_literal(value, path) if transform_literal else value

    if isinstance(value, Mapping):
        if isinstance(value.get("inference_provider"), str):
            default_username = value["inference_provider"]
        elif isinstance(value.get("model"), str) and ":" in value["model"]:
            default_username = value["model"].split(":", 1)[0]

        transformed = {}
        for name, child in value.items():
            child_username = default_username
            if path in {("inference_providers",), ("mcp_servers",)}:
                child_username = str(name)
            elif child_username is None:
                child_username = str(name)
            if name == "api_key_env" and isinstance(child, str):
                transformed["api_key"] = transform(
                    SecretReference(env=child),
                    (*path, "api_key"),
                    child_username,
                )
            else:
                transformed[name] = transform_secret_references(
                    child,
                    transform,
                    transform_literal=transform_literal,
                    path=(*path, str(name)),
                    default_username=child_username,
                )
        return transformed

    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [
            transform_secret_references(
                child,
                transform,
                transform_literal=transform_literal,
                path=(*path, str(index)),
                default_username=default_username,
            )
            for index, child in enumerate(value)
        ]
    return value


def iter_secret_references(
    value: Any,
) -> Iterator[tuple[SecretPath, SecretReference, str | None]]:
    """Yield every external secret reference in a loaded URSA config."""
    references: list[tuple[SecretPath, SecretReference, str | None]] = []

    def collect(
        reference: SecretReference,
        path: SecretPath,
        default_username: str | None,
    ) -> SecretReference:
        references.append((path, reference, default_username))
        return reference

    transform_secret_references(value, collect)
    yield from references


def externalize_secret_references(
    value: Any,
    default_username: str | None = None,
) -> tuple[Any, dict[str, str]]:
    """Resolve secrets here and replace them with isolated environment refs."""
    secret_env: dict[str, str] = {}

    def externalize(secret: str, template: str | None = None) -> dict[str, str]:
        name = f"URSA_HARBOR_SECRET_{len(secret_env)}"
        secret_env[name] = secret
        reference = {"env": name}
        if template is not None:
            reference["template"] = template
        return reference

    def resolve_reference(
        reference: SecretReference,
        path: SecretPath,
        username: str | None,
    ) -> dict[str, str]:
        resolved = reference.resolve(username)
        if resolved is None:
            assert reference.env is not None
            location = ".".join(path) or "config"
            raise ValueError(
                f"Secret at {location} uses unset environment variable "
                f"'{reference.env}'"
            )
        template = (
            reference.template
            if isinstance(reference, SecretTemplate)
            else None
        )
        return externalize(resolved.get_secret_value(), template)

    projected = transform_secret_references(
        value,
        resolve_reference,
        transform_literal=lambda secret, _path: externalize(
            secret.get_secret_value()
        ),
        default_username=default_username,
    )
    return projected, secret_env
