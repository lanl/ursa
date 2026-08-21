"""Short hints displayed in the Textual welcome banner."""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import TYPE_CHECKING

from textual.binding import Binding

if TYPE_CHECKING:
    from textual.app import App

TIPS = (
    "{agent_macro} opens the fuzzy agent picker and routes the prompt.",
    "{file_macro} finds workspace files and directories without leaving the app.",
    "{command_macro} opens commands for managing agents, viewing status, and displaying the complete keymap.",
    "{cancel} closes a picker without changing your prompt.",
    "{insert_newline} adds a newline; {submit_prompt} submits the prompt.",
    "{clear_prompt} clears the prompt; {history_up} restores it from history.",
    "Tool and agent outputs are truncated by default; {toggle_card_details} toggles the full output.",
    "{previous_turn_marker} and {next_turn_marker} jump to the previous or next turn marker.",
    "{agent_macro}agent routes your next prompt to that agent without changing the default.",
    "Named agents preserve state between sessions. Start URSA with `--name` to resume one.",
    "MCP tools are attached only to agents that support tools.",
    "{quit} waits for the active turn before quitting; {hard_quit} quits immediately.",
    "Use {command_macro}agents to explore available agents and their tools.",
    "Use {command_macro}keymap to see all available keyboard shortcuts.",
    "Use {command_macro}theme to change the color theme.",
    "Found a problem? Let us know: https://github.com/lanl/ursa/issues",
    "Unsure about something? Check out our docs: https://lanl.github.io/ursa",
)

BEAR_FACTS = (
    "Despite their name, black bears can be black, cinnamon, brown, blond, and even white.",  # https://www.nps.gov/subjects/bears/black-bears.htm
    "Polar bears can smell a carcass from nearly 20 miles away.",  # https://www.nps.gov/subjects/bears/polar-bears.htm
    "A Kodiak brown bear can be up to 10 feet tall when standing upright.",  # https://www.fws.gov/species/kodiak-brown-bear-ursus-arctos-middendorffi
    "A black bear can run as fast as 35 miles per hour.",  # https://www.nps.gov/glac/learn/nature/bears.htm
    "The Andean bear, also known as the spectacled bear, is the only bear native to South America.",  # https://nationalzoo.si.edu/animals/andean-bear
)


def _effective_bindings(owner: type[object]) -> Iterable[Binding]:
    """Yield an owner's bindings with runtime subclass overrides applied."""
    bindings: dict[str, Binding] = {}
    for base in reversed(owner.__mro__):
        for binding in Binding.make_bindings(base.__dict__.get("BINDINGS", ())):
            bindings[binding.key] = binding
    return bindings.values()


def runtime_keymap(
    app: App[object], owners: Iterable[type[object]]
) -> dict[str, str]:
    """Map binding actions to their current, terminal-friendly key labels."""
    keymap: dict[str, list[str]] = {}
    newline_key = getattr(app, "preferred_newline_key", "ctrl+j")
    for owner in owners:
        for binding in _effective_bindings(owner):
            if (
                binding.action == "insert_newline"
                and binding.key != newline_key
            ):
                continue
            keymap.setdefault(binding.action, []).append(
                app.get_key_display(binding)
            )
    return {action: " / ".join(keys) for action, keys in keymap.items()}


def random_tip(app: App[object], owners: Iterable[type[object]]) -> str:
    """Choose one welcome hint for the current application session."""
    if random.random() <= (1 / (len(TIPS) + 1)):
        return random.choice(BEAR_FACTS)
    return random.choice(TIPS).format_map(runtime_keymap(app, owners))
