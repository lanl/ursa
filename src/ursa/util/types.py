import re
from typing import Annotated

from pydantic import StringConstraints

AsciiStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, strict=True, pattern=r"^[\x20-\x7E\t\n\r\f\v]+$"
    ),
]
""" Limit strings to "text" ASCII characters (letters, digits, symbols, whitespace) """

_ALLOWED_ASCII_PATTERN = re.compile(r"^[\x20-\x7E\t\n\r\f\v]*$")
_INVALID_ASCII_PATTERN = re.compile(r"[^\x20-\x7E\t\n\r\f\v]")
_ALLOWED_ASCII_DESCRIPTION = (
    "printable ASCII characters plus tab, newline, carriage return, "
    "form feed, and vertical tab"
)


class AsciiValidationError(ValueError):
    """Raised when a tool string contains unsupported non-ASCII characters."""


def _invalid_ascii_description(value: str) -> str:
    details = []
    for match in _INVALID_ASCII_PATTERN.finditer(value):
        char = match.group()
        details.append(f"position {match.start()} (U+{ord(char):04X})")
    return ", ".join(details)


def validate_ascii(value: str | None) -> str | None:
    """Validate tool strings against URSA's ASCII text-string constraint.

    This intentionally rejects invalid characters instead of cleaning them. Silent
    cleaning can change file paths or commands into different valid strings, which
    risks operating on the wrong target.
    """
    if value is None:
        return value

    if type(value) is not str:
        raise AsciiValidationError(
            f"Value must be a strict string, got {type(value).__name__}."
        )

    if not _ALLOWED_ASCII_PATTERN.fullmatch(value):
        invalid = _invalid_ascii_description(value)
        raise AsciiValidationError(
            f"Value must contain only {_ALLOWED_ASCII_DESCRIPTION}; "
            f"invalid character(s) at {invalid}."
        )

    return value.strip()


def ascii_validation_message(field_name: str, exc: AsciiValidationError) -> str:
    """Format an ASCII validation failure for return to an LLM tool caller."""
    return (
        f"Invalid {field_name}: {exc} Please provide a corrected ASCII string."
    )
