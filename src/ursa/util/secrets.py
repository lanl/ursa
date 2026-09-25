"""References to secrets stored outside configuration files."""

from copy import deepcopy
from os import environ
from typing import Annotated, Any, Self

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

    def model_merge(self, other: Self | dict[str, Any]) -> Self:
        """Merge a higher-priority reference without combining its sources."""
        if isinstance(other, SecretReference):
            updates = other.model_dump(mode="python", exclude_unset=True)
            update_fields = set(other.model_fields_set)
        else:
            updates = deepcopy(other)
            update_fields = set(updates)

        superseded_field = None
        has_env = updates.get("env") is not None
        has_keyring = updates.get("keyring") not in (None, False)
        if has_env and not has_keyring:
            updates["keyring"] = None
            superseded_field = "keyring"
        elif has_keyring and not has_env:
            updates["env"] = None
            superseded_field = "env"

        merged = type(self).model_validate({
            **self.model_dump(mode="python"),
            **updates,
        })
        fields_set = self.model_fields_set | update_fields
        if superseded_field is not None:
            fields_set.discard(superseded_field)
        merged.__pydantic_fields_set__ = fields_set
        return merged

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
