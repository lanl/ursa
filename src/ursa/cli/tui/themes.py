"""URSA themes for the Textual CLI."""

from textual.theme import BUILTIN_THEMES, Theme

URSA_DARK = Theme(
    name="ursa-dark",
    primary="#0178D4",
    secondary="#004578",
    accent="#ffa62b",
    warning="#ffa62b",
    error="#ba3c5b",
    success="#4EBF71",
    foreground="#d5dbe0",
    background="#101317",
    surface="#14191d",
    panel="#202830",
)


URSA_LIGHT = Theme(
    name="ursa-light",
    primary="#496f91",
    secondary="#5d8068",
    accent="#496f91",
    warning="#9b6b22",
    error="#a83f50",
    success="#39734a",
    foreground="#24313c",
    background="#f5f7f9",
    surface="#ffffff",
    panel="#e7edf2",
    dark=False,
)


AVAILABLE_THEMES = (URSA_DARK, URSA_LIGHT, *BUILTIN_THEMES.values())
