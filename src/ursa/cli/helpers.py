"""Pure helpers and constants shared by the Textual CLI."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rich.cells import cell_len, chop_cells  # noqa: TID251

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
    "agents": "Configured agents, descriptions, options, and tools",
    "status": "Tokens, models, endpoints, group, and MCP servers",
    "keymap": "Complete keyboard map",
    "theme": "Choose the application color theme",
}


def _plan_step_text(index: int, step: Any) -> str:
    """Normalize one plan step into numbered display text."""
    if not isinstance(step, Mapping):
        dump = getattr(step, "model_dump", None)
        step = dump() if callable(dump) else {"name": str(step)}
    name = str(step.get("name") or f"Step {index}")
    description = " ".join(str(step.get("description") or "").split())
    return f"{index}. {name}" + (f": {description}" if description else "")


def _truncate_middle(text: str, width: int) -> str:
    """Fit text to a terminal-cell width while preserving both ends."""
    if cell_len(text) <= width:
        return text
    marker = " … truncated … "
    available = width - cell_len(marker)
    if available < 0:
        if width <= 0:
            return ""
        if width == 1:
            return "…"
        return f"{chop_cells(text, width - 1)[0]}…"
    left = (available + 1) // 2
    right = available // 2
    prefix = chop_cells(text, left)[0]
    suffix = chop_cells(text[::-1], right)[0][::-1]
    return f"{prefix} _… truncated …_ {suffix}"


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
    match = re.match(
        r"^#(?P<name>\S+)(?:\s(?P<prompt>.*))?$", prompt, re.DOTALL
    )
    if match and match["name"] in hitl.agents:
        return match["name"], match["prompt"] or ""
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


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token counts from one model response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0


def _token_usage_breakdown(value: Any) -> TokenUsage:
    """Extract token counts from common LangChain/provider response shapes."""
    seen: set[int] = set()
    candidates: list[TokenUsage] = []

    def count(item: Any) -> int:
        return (
            item if isinstance(item, int) and not isinstance(item, bool) else 0
        )

    def mapping_usage(item: Mapping[str, Any]) -> TokenUsage:
        input_tokens = max(
            count(item.get(key))
            for key in ("input_tokens", "prompt_tokens", "input_token_count")
        )
        output_tokens = max(
            count(item.get(key))
            for key in (
                "output_tokens",
                "completion_tokens",
                "output_token_count",
            )
        )
        cached_tokens = max(
            count(item.get(key))
            for key in (
                "cached_tokens",
                "cached_input_tokens",
                "cache_read_input_tokens",
                "prompt_cache_hits",
            )
        )
        for details_key in (
            "input_token_details",
            "input_tokens_details",
            "prompt_tokens_details",
        ):
            details = item.get(details_key)
            if isinstance(details, Mapping):
                cached_tokens = max(
                    cached_tokens,
                    count(details.get("cached_tokens")),
                    count(details.get("cache_read")),
                )
        total_tokens = max(
            count(item.get("total_tokens")),
            count(item.get("total_token_count")),
            input_tokens + output_tokens,
        )
        return TokenUsage(
            input_tokens,
            output_tokens,
            cached_tokens,
            total_tokens,
        )

    def visit(item: Any) -> None:
        if item is None or id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, Mapping):
            usage = mapping_usage(item)
            if usage != TokenUsage():
                candidates.append(usage)
            for child in item.values():
                visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        for attribute in (
            "llm_output",
            "usage_metadata",
            "response_metadata",
            "generations",
            "message",
        ):
            if hasattr(item, attribute):
                visit(getattr(item, attribute))

    visit(value)
    if not candidates:
        return TokenUsage()
    best = max(
        candidates,
        key=lambda usage: (
            bool(usage.input_tokens) + bool(usage.output_tokens),
            usage.total_tokens,
        ),
    )
    return TokenUsage(
        best.input_tokens,
        best.output_tokens,
        max(usage.cached_tokens for usage in candidates),
        best.total_tokens,
    )


def _token_usage(value: Any) -> int:
    """Extract total token usage from common LangChain response shapes."""
    return _token_usage_breakdown(value).total_tokens


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
