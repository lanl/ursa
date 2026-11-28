"""Model-card recognition, requirement matching, and model execution.

The package turns scientific foundation-model cards from many sources into
one normalized :class:`ModelCard`, matches cards to user requirements, and
plans model runs. :class:`~ursa.agents.ModelCardAgent` drives these pieces
agentically; the functions below are also usable on their own::

    from ursa.model_cards import ModelCatalog, extract_model_card, load_document

    card = extract_model_card(load_document("hf://camlab-ethz/Poseidon-B"))
    print(card.inputs, card.architecture.parameter_count)
"""

import importlib
from typing import Any

_lazy_attrs: dict[str, tuple[str, str]] = {
    "CardDocument": (".extract", "CardDocument"),
    "DOMAINS": (".taxonomy", "DOMAINS"),
    "ModelCard": (".schema", "ModelCard"),
    "ModelCatalog": (".catalog", "ModelCatalog"),
    "Recommendation": (".schema", "Recommendation"),
    "RunPlan": (".running", "RunPlan"),
    "TASKS": (".taxonomy", "TASKS"),
    "UserRequirements": (".schema", "UserRequirements"),
    "Vocabulary": (".taxonomy", "Vocabulary"),
    "build_run_plan": (".running", "build_run_plan"),
    "extract_model_card": (".extract", "extract_model_card"),
    "heuristic_requirements": (".requirements", "heuristic_requirements"),
    "load_document": (".sources", "load_document"),
    "model_card_tools": (".tools", "model_card_tools"),
    "parse_requirements": (".requirements", "parse_requirements"),
    "rank_models": (".matching", "rank_models"),
    "refine_card": (".llm_extract", "refine_card"),
    "render_card": (".report", "render_card"),
}

__all__ = list(_lazy_attrs)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _lazy_attrs[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(importlib.import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + __all__)
