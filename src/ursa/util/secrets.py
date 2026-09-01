"""References to secrets stored outside configuration files."""

from collections.abc import Mapping, Sequence
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


def externalize_secret_references(
    value: Any, default_username: str | None = None
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

    def resolve(item: Any, path: tuple[str, ...], username: str | None) -> Any:
        is_template = isinstance(item, SecretTemplate) or (
            isinstance(item, Mapping) and "template" in item
        )
        reference = (
            SecretTemplate.maybe_validate(item)
            if is_template
            else SecretReference.maybe_validate(item)
        )
        if isinstance(reference, SecretReference):
            resolved = reference.resolve(username)
            if resolved is None:
                assert reference.env is not None
                raise ValueError(
                    f"Secret at {'.'.join(path)} uses unset environment "
                    f"variable '{reference.env}'"
                )
            template = reference.template if is_template else None
            return externalize(resolved.get_secret_value(), template)

        if isinstance(item, SecretStr):
            return externalize(item.get_secret_value())

        if isinstance(item, Mapping):
            if isinstance(item.get("inference_provider"), str):
                username = item["inference_provider"]
            elif isinstance(item.get("model"), str) and ":" in item["model"]:
                username = item["model"].split(":", 1)[0]

            resolved = {}
            for name, child in item.items():
                child_username = username
                if path in {("inference_providers",), ("mcp_servers",)}:
                    child_username = str(name)
                elif child_username is None:
                    child_username = str(name)
                if name == "api_key_env" and isinstance(child, str):
                    resolved["api_key"] = resolve(
                        {"env": child}, (*path, "api_key"), child_username
                    )
                else:
                    resolved[name] = resolve(
                        child, (*path, str(name)), child_username
                    )
            return resolved

        if isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            return [
                resolve(child, (*path, str(index)), username)
                for index, child in enumerate(item)
            ]
        return item

    return resolve(value, (), default_username), secret_env
