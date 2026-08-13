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
    "{command_macro} opens commands for agents, status, and the complete keymap.",
    "{cancel} closes a picker without changing your prompt.",
    "{insert_newline} adds a newline; {submit_prompt} submits the prompt.",
    "{clear_prompt} clears the prompt; {history_up} restores it from history.",
    "{toggle_transcript} toggles the complete event transcript.",
    "Tool and agent output is truncated by default; {toggle_card_details} shows or hides the full output.",
    "{previous_turn_marker} and {next_turn_marker} move between turn markers.",
    "{agent_macro}agent routes the next prompt without changing the default agent.",
    "Named agents preserve state between sessions. Start URSA with `--name` to load one.",
    "MCP tools are attached only to agents that support tool use.",
    "{quit} waits for the active turn before quitting; {hard_quit} quits immediately.",
    "Use {command_macro}agents to explore available agents and their tools.",
    "Use {command_macro}keymap to see all available keyboard shortcuts.",
    "Use {command_macro}theme to change the color theme.",
    "Something broken? Let us know: https://github.com/lanl/ursa/issues",
    "Unsure about something? Check out our docs: https://lanl.github.io/ursa",
)

BEAR_FACTS = (
    "Despite their name Black bears can be black, cinnamon, brown, blond and even white",  # https://www.nps.gov/subjects/bears/black-bears.htm
    "Polar bears can smell a carcass from nearly 20 miles away.",  # https://www.nps.gov/subjects/bears/polar-bears.htm
    "A Kodiak brown bear can be up to 10 feet tall when standing upright",  # https://www.fws.gov/species/kodiak-brown-bear-ursus-arctos-middendorffi
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
    for owner in owners:
        for binding in _effective_bindings(owner):
            keymap.setdefault(binding.action, []).append(
                app.get_key_display(binding)
            )
    return {action: " / ".join(keys) for action, keys in keymap.items()}


def random_tip(app: App[object], owners: Iterable[type[object]]) -> str:
    """Choose one welcome hint for the current application session."""
    if random.random() <= 0.1:
        return random.choice(BEAR_FACTS)
    return random.choice(TIPS).format_map(runtime_keymap(app, owners))
