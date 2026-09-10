"""References to secrets stored outside configuration files."""

from os import environ
from typing import Annotated, Any, Self
from warnings import warn

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
    """A secret reference rendered into a string at point of use."""

    model_config = ConfigDict(extra="forbid")

    env: Annotated[str | None, AfterValidator(_non_blank)] = None
    keyring: bool | Annotated[str, AfterValidator(_non_blank)] | None = None
    template: str = "%s"

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

    @model_validator(mode="after")
    def _validate_template(self):
        if self.template.count("%s") != 1:
            raise ValueError("secret template must contain exactly one '%s'")
        return self

    def bind_keyring_name(self, name: str) -> Self:
        """Replace the contextual ``keyring: true`` shorthand with a name."""
        if self.keyring is not True:
            return self
        normalized = _non_blank(name)
        assert normalized is not None
        return self.model_copy(update={"keyring": normalized})

    def resolve(self) -> SecretStr | None:
        """Resolve and render the secret, or return ``None`` for a missing env."""
        if self.env is not None:
            value = environ.get(self.env)
            return (
                SecretStr(self.template % value) if value is not None else None
            )

        import keyring

        username = self.keyring
        if not isinstance(username, str):
            raise ValueError(
                "keyring=true must be bound by UrsaConfig.resolve() before "
                "the secret is used"
            )
        value = keyring.get_password("ursa", username)
        if value is None:
            raise ValueError(
                f"No secret found in the system keyring for '{username}'"
            )
        return SecretStr(self.template % value)

    def get_secret_value(self) -> str | None:
        """Resolve and unwrap the referenced secret value."""
        secret = self.resolve()
        return secret.get_secret_value() if secret is not None else None


class SecretTemplate(SecretReference):
    """Deprecated compatibility name for :class:`SecretReference`."""

    def __init__(self, **data: Any) -> None:
        warn(
            "SecretTemplate is deprecated; use SecretReference instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)
