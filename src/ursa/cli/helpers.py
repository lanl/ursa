"""Pure helpers and constants shared by the Textual CLI."""

from collections.abc import Mapping
from typing import Any

from ursa.cli.runtime import HITL

SUMMARY_GROUP_GRACE_SECONDS = 3.0
FILE_TOOLS = {
    "read_file": "Reading",
    "write_code": "Editing",
    "write_code_with_repo": "Editing",
    "edit_code": "Editing",
}

SEARCH_TOOLS = {
    "run_arxiv_search",
    "run_osti_search",
    "run_web_search",
}

AGENT_LABELS = {
    "ExecutionAgent": ("⚙️", "Execute"),
    "DeepReviewAgent": ("🔎", "Deep Review"),
    "HypothesizerAgent": ("💡", "Hypothesize"),
    "LammpsAgent": ("⚛️", "LAMMPS"),
    "PlanningAgent": ("🗺️", "Plan"),
    "executor": ("⚙️", "Execute"),
    "deep_review": ("🔎", "Deep Review"),
    "hypothesizer": ("💡", "Hypothesize"),
    "lammps": ("⚛️", "LAMMPS"),
    "planner": ("🗺️", "Plan"),
}

COMMAND_CHOICES = {
    "agents": "Configured agents, descriptions, and options",
    "status": "Tokens, models, endpoints, group, and MCP servers",
    "keymap": "Complete keyboard map",
}


def _endpoint(value: Any) -> str:
    """Return a concise endpoint label for a model-like object."""
    if value is None:
        return "none"
    for attribute in ("base_url", "api_base", "openai_api_base"):
        if endpoint := getattr(value, attribute, None):
            return str(endpoint)
    return "default"


def _embedding_name(hitl: HITL) -> str:
    embedding = getattr(hitl, "embedding", None)
    if embedding is None:
        return "none"
    for attribute in ("model_name", "model"):
        if value := getattr(embedding, attribute, None):
            return str(value)
    return type(embedding).__name__


def _route_prompt(hitl: HITL, prompt: str) -> tuple[str, str]:
    """Route a leading ``#agent`` macro, defaulting to chat."""
    first, separator, rest = prompt.partition(" ")
    if first.startswith("#") and first[1:] in hitl.agents:
        return first[1:], rest if separator else ""
    return "chat", prompt


def _fuzzy_match(query: str, candidate: str) -> bool:
    """Return whether all query characters occur in order in candidate."""
    return _fuzzy_score(query, candidate) is not None


def _field_fuzzy_score(query: str, value: str) -> int | None:
    """Score a fuzzy subsequence, favoring compact and early matches."""
    query = query.casefold()
    value = value.casefold()
    if not query:
        return 0
    positions: list[int] = []
    start = 0
    for character in query:
        position = value.find(character, start)
        if position < 0:
            return None
        positions.append(position)
        start = position + 1
    span = positions[-1] - positions[0] + 1
    score = 1000 - positions[0] * 4 - (span - len(query)) * 8
    if query == value:
        score += 3000
    elif value.startswith(query):
        score += 2000
    elif query in value:
        score += 1000
    return score


def _fuzzy_score(query: str, candidate: str) -> int | None:
    """Rank matches, strongly preferring a picker's primary name field."""
    primary, separator, description = candidate.partition(" — ")
    primary_score = _field_fuzzy_score(query, primary)
    description_score = (
        _field_fuzzy_score(query, description) if separator else None
    )
    scores = []
    if primary_score is not None:
        scores.append(10_000 + primary_score)
    if description_score is not None:
        scores.append(description_score)
    return max(scores) if scores else None


def _token_usage(value: Any) -> int:
    """Extract total token usage from common LangChain response shapes."""
    seen: set[int] = set()

    def visit(item: Any) -> int:
        if item is None or id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, dict):
            for key in ("total_tokens", "total_token_count"):
                count = item.get(key)
                if isinstance(count, int):
                    return count
            return max((visit(child) for child in item.values()), default=0)
        if isinstance(item, (list, tuple)):
            return max((visit(child) for child in item), default=0)
        for attribute in (
            "llm_output",
            "usage_metadata",
            "response_metadata",
            "generations",
            "message",
        ):
            if hasattr(item, attribute):
                count = visit(getattr(item, attribute))
                if count:
                    return count
        return 0

    return visit(value)


def _model_name(hitl: HITL) -> str:
    model = hitl.model
    for attribute in ("model_name", "model"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    return type(model).__name__


def _reasoning_trace(chunk: Any) -> str | None:
    """Extract provider-published reasoning summaries from an LLM chunk."""

    def text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            return " ".join(
                text(value.get(key))
                for key in (
                    "text",
                    "content",
                    "summary",
                    "reasoning",
                    "thinking",
                )
                if value.get(key)
            )
        if isinstance(value, (list, tuple)):
            return " ".join(filter(None, (text(item) for item in value)))
        return ""

    values = [chunk, getattr(chunk, "message", None)]
    for value in values:
        mappings = [value] if isinstance(value, Mapping) else []
        for attribute in ("additional_kwargs", "response_metadata"):
            mapping = getattr(value, attribute, None)
            if isinstance(mapping, Mapping):
                mappings.append(mapping)
        for mapping in mappings:
            for key in (
                "reasoning_content",
                "reasoning_summary",
                "reasoning",
                "thinking",
            ):
                if trace := " ".join(text(mapping.get(key)).split()):
                    return trace[-500:]

        content = getattr(value, "content", None)
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if str(block.get("type", "")).casefold() in {
                    "reasoning",
                    "reasoning_summary",
                    "thinking",
                }:
                    if trace := " ".join(text(block).split()):
                        return trace[-500:]
    return None
