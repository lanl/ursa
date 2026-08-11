"""Short hints displayed in the Textual welcome banner."""

import random

TIPS = (
    "# opens the fuzzy agent picker and routes the prompt.",
    "@ finds workspace files and directories without leaving the app.",
    "/ opens commands for agents, status, and the complete keymap.",
    "Esc closes a picker without changing your prompt.",
    "Shift+Enter adds a newline; Enter submits the prompt.",
    "Ctrl+C clears the prompt; Up restores it from history.",
    "Ctrl+T toggles the complete event transcript.",
    "Ctrl+O expands or collapses full command output.",
    "Alt+Up and Alt+Down move between turn markers.",
    "The prompt grows automatically, up to ten lines.",
)


def random_tip() -> str:
    """Choose one welcome hint for the current application session."""
    return random.choice(TIPS)
