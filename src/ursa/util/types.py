from typing import Annotated

from pydantic import StringConstraints

AsciiStr = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, strict=True, pattern=r"^[\x20-\x7E\t\n\r\f\v]+$"
    ),
]
""" Limit strings to "text" ASCII characters (letters, digits, symbols, whitespace) """
