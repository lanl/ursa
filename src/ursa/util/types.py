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


def validate_ascii(value: str) -> str:
    # 0. Allow pass through if None for optional args
    if value is None:
        return value

    # 1. Enforce strict type checking
    if type(value) is not str:
        raise TypeError("Value must be a strict string")

    # 2. Define the inverse regex pattern
    # The '^' inside '[^...]' removes anything NOT in your allowed ASCII set
    invalid_chars_pattern = re.compile(r"[^\x20-\x7E\t\n\r\f\v]")

    # 3. Silently drop invalid characters
    cleaned_value = invalid_chars_pattern.sub("", value)

    # 4. Strip leading and trailing whitespace
    return cleaned_value.strip()
