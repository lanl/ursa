"""LangChain tools that give any URSA agent model-card skills.

Attach them to a tool-using agent, for example::

    from ursa.agents import ExecutionAgent
    from ursa.model_cards.tools import model_card_tools

    agent = ExecutionAgent(llm=llm, extra_tools=model_card_tools())

The tools read and write the same catalog as ``ModelCardAgent``: the
directory in ``config["configurable"]["model_catalog_dir"]`` when set,
otherwise the group catalog in the URSA cache.
"""

from __future__ import annotations

import json
from typing import Annotated

from langchain.tools import ToolRuntime
from langchain_core.tools import BaseTool, tool

from ursa.agents.base import AgentContext
from ursa.model_cards.catalog import ModelCatalog
from ursa.model_cards.extract import extract_model_card
from ursa.model_cards.matching import rank_models
from ursa.model_cards.report import (
    render_card,
    render_recommendations,
    render_requirements,
)
from ursa.model_cards.requirements import heuristic_requirements
from ursa.model_cards.running import (
    choose_interface,
    parse_json_arguments,
    validate_arguments,
)
from ursa.model_cards.sources import load_document
from ursa.util.events import ToolEvents


def catalog_from_runtime(
    runtime: ToolRuntime[AgentContext] | None,
) -> ModelCatalog:
    configurable = (
        ((runtime.config or {}).get("configurable") or {}) if runtime else {}
    )
    directory = configurable.get("model_catalog_dir")
    if directory:
        return ModelCatalog(directory)
    group = (
        getattr(getattr(runtime, "context", None), "group", None) or "default"
    )
    return ModelCatalog.for_group(group)


@tool
def parse_model_card(
    source: Annotated[
        str,
        "Model card location: hf://org/name, a Hugging Face URL, a URL to a Markdown/YAML/JSON card, "
        "or a workspace path to a card file or model directory.",
    ],
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Read a foundation-model card, normalize it, and save it to the model catalog.

    Returns the recognized metadata, input and output formats, compute needs, and
    card-quality notes.
    """
    events = ToolEvents.from_runtime("parse_model_card", runtime)
    workspace = getattr(runtime.context, "workspace", None)
    with events.range(
        "ingest", "Reading model card", done="Model card read", source=source
    ):
        try:
            card = extract_model_card(
                load_document(source, workspace=workspace)
            )
        except (OSError, ValueError) as exc:
            return f"Could not read a model card from {source!r}: {exc}"
        catalog_from_runtime(runtime).add(card)
    return render_card(card)


@tool
def search_model_cards(
    query: Annotated[str, "Keywords describing the physics, task, or data."],
    runtime: ToolRuntime[AgentContext],
    k: Annotated[int, "Maximum number of models to return."] = 5,
) -> str:
    """Search the model catalog by keywords; returns matching models with brief facts."""
    results = catalog_from_runtime(runtime).search(query, k=max(1, k))
    if not results:
        return "No catalog models match the query."
    return "\n".join(
        f"- {card.name} (`{card.id}`), score {score:.2f}: {', '.join(card.domains[:2])} / "
        f"{', '.join(card.tasks[:2])}; {(card.summary or '')[:160]}"
        for card, score in results
    )


@tool
def describe_model_card(
    model: Annotated[str, "Catalog model id, name, alias, or repository id."],
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Show the full normalized card of a catalog model, including parameter formats."""
    card = catalog_from_runtime(runtime).get(model)
    if card is None:
        return f"No catalog model matches {model!r}. Use parse_model_card to add it first."
    return render_card(card)


@tool
def recommend_models(
    request: Annotated[
        str, "What the user needs: physics, task, data, constraints."
    ],
    runtime: ToolRuntime[AgentContext],
    k: Annotated[int, "Number of recommendations."] = 3,
) -> str:
    """Rank catalog models for a request and explain each ranking."""
    catalog = catalog_from_runtime(runtime)
    cards = catalog.all()
    if not cards:
        return "The model catalog is empty. Use parse_model_card to add model cards first."
    workspace = getattr(runtime.context, "workspace", None)
    req = heuristic_requirements(request, workspace=workspace)
    recs = rank_models(req, cards, top_k=max(1, k))
    return f"Requirements: {render_requirements(req)}\n\n{render_recommendations(recs, limit=k)}"


@tool
def check_model_inputs(
    model: Annotated[str, "Catalog model id, name, or alias."],
    arguments_json: Annotated[
        str, "Candidate tool arguments as a JSON object."
    ],
    runtime: ToolRuntime[AgentContext],
) -> str:
    """Validate candidate arguments against a model's documented input schema."""
    card = catalog_from_runtime(runtime).get(model)
    if card is None:
        return f"No catalog model matches {model!r}."
    arguments = parse_json_arguments(arguments_json)
    if arguments is None:
        return "arguments_json must contain a JSON object."
    interface = choose_interface(card)
    if interface is not None and interface.input_schema:
        errors = validate_arguments(interface.input_schema, arguments)
        return (
            "Arguments are valid."
            if not errors
            else "Invalid arguments:\n" + "\n".join(f"- {e}" for e in errors)
        )
    names = {spec.name for spec in card.inputs}
    unknown = sorted(set(arguments) - names)
    missing = sorted(
        spec.name
        for spec in card.inputs
        if spec.required and spec.name not in arguments
    )
    report = {
        "documented_inputs": sorted(names),
        "unknown": unknown,
        "missing": missing,
    }
    return (
        "The card has no machine-readable schema; compared names only:\n"
        + json.dumps(report, indent=2)
    )


def model_card_tools() -> list[BaseTool]:
    """All model-card tools, for ``extra_tools=`` on tool-using agents."""
    return [
        parse_model_card,
        search_model_cards,
        describe_model_card,
        recommend_models,
        check_model_inputs,
    ]
