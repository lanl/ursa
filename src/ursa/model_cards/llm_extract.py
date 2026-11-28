"""Evidence-checked LLM refinement of extracted model cards.

Deterministic extraction is precise but misses facts written as free prose,
such as "expects density, velocity, and pressure on a 128 x 128 grid". This
module asks an LLM to fill the gaps under one strict rule: every value must
come with a quote from the card, and the quote is checked against the card
text before the value is accepted. The check ignores case, whitespace, quote
styles, and Markdown markup, and it accepts "..." elisions. A quote of six or
more words may differ from the card in up to 10% of its word trigrams, but
never in a trigram that holds a number. Numbers in a value must also appear
in its quote. Values that fail are dropped and counted in a
``llm_unverified`` issue. The LLM sees the card without HTML comments and
template text, the same text the deterministic extractor reads.

LLM values fill empty fields and add list items, such as domains, tasks,
input and output specs, resolutions, and license restrictions. They never
override structured sources such as ``config.json``, front matter, or MCP
schemas. The one exception narrows use: a verified non-commercial
restriction sets commercial use to false.
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ursa.model_cards import patterns as P
from ursa.model_cards.extract import assess_completeness
from ursa.model_cards.markdown import evidence_text, parse_card_text, prose
from ursa.model_cards.schema import (
    CardIssue,
    ChannelSpec,
    Evidence,
    IOSpec,
    ModelCard,
)
from ursa.model_cards.taxonomy import DOMAINS, TASKS, Vocabulary
from ursa.util.structured_output import invoke_structured


class Quoted(BaseModel):
    value: str = Field(description="The fact, stated concisely.")
    evidence: str = Field(
        description="Exact sentence or phrase copied from the card that states the fact."
    )


class QuotedIO(BaseModel):
    role: Literal["input", "output"]
    name: str = Field(description="Variable, field, or argument name.")
    description: str | None = None
    kind: str | None = Field(
        default=None,
        description=(
            "field, tensor, scalar, vector, time_series, image, molecule, "
            "structure, graph, point_cloud, sequence, text, table, or file"
        ),
    )
    representation: str | None = Field(
        default=None,
        description="Encoding such as SMILES, NetCDF, or NumPy array.",
    )
    shape: str | None = Field(
        default=None, description="Shape as written, e.g. '(B, C, H, W)'."
    )
    dtype: str | None = None
    units: str | None = None
    normalization: str | None = None
    semantics: str | None = Field(
        default=None,
        description="Meaning of output values: absolute state, delta, regression, logits, probabilities, embedding.",
    )
    channels: list[str] = Field(
        default_factory=list,
        description="Named channels or variables, in order.",
    )
    evidence: str = Field(
        description="Exact quote from the card describing this input or output."
    )


class CardExtraction(BaseModel):
    """Facts read from a model card, each with a verbatim quote."""

    summary: Quoted | None = None
    domains: list[Quoted] = Field(
        default_factory=list,
        description="Physics domain ids from the provided list.",
    )
    tasks: list[Quoted] = Field(
        default_factory=list, description="Task ids from the provided list."
    )
    io: list[QuotedIO] = Field(default_factory=list)
    parameter_count: Quoted | None = None
    spatial_resolution: list[Quoted] = Field(default_factory=list)
    n_spatial_dims: Quoted | None = None
    time_step: Quoted | None = None
    horizon: Quoted | None = None
    context_steps: Quoted | None = None
    min_gpu_memory_gb: Quoted | None = None
    cpu_supported: Quoted | None = None
    intended_uses: list[Quoted] = Field(default_factory=list)
    limitations: list[Quoted] = Field(default_factory=list)
    out_of_scope: list[Quoted] = Field(default_factory=list)
    license_restrictions: list[Quoted] = Field(default_factory=list)


EXTRACTION_PROMPT = """You read scientific foundation-model cards and record facts.

Rules:
- Record only facts the card states. Never infer, guess, or use outside knowledge.
- Every value needs an "evidence" field: copy the supporting sentence or phrase from the card exactly, character for character.
- Leave a field empty when the card does not state it. Empty is better than wrong.
- domains and tasks must use ids from the lists below.
- For inputs and outputs, record names, shapes, units, normalization, and value meaning only as written.

Physics domain ids:
{domains}

Task ids:
{tasks}
"""


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )
    text = re.sub(r"[`*_>#|]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_source(
    quote: str, source: str, *, normalized_source: str | None = None
) -> bool:
    """True when ``quote`` appears in ``source`` (whitespace and markup tolerant)."""
    q = _normalize(quote)
    s = (
        normalized_source
        if normalized_source is not None
        else _normalize(source)
    )
    if len(q) < 6:
        return False
    if q in s:
        return True
    parts = [
        p.strip(" .")
        for p in re.split(r"\.\.\.|…|\[\.\.\.\]", q)
        if len(p.strip(" .")) >= 8
    ]
    if len(parts) > 1 and all(part in s for part in parts):
        return True
    words = q.split()
    if len(words) < 6:
        return False
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    present = [gram in s for gram in grams]
    if any(not ok and re.search(r"\d", g) for g, ok in zip(grams, present)):
        # The tolerance absorbs paraphrase, not changed numbers: without this,
        # "a 96 x 96 grid" quoted as "a 128 x 96 grid" passed.
        return False
    return sum(present) / len(grams) >= 0.9


def _numbers(text: str | None) -> set[str]:
    return {
        n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", text or "")
    }


def numbers_supported(value: str, quote: str) -> bool:
    """Every number in ``value`` must also appear in ``quote``."""
    return _numbers(value) <= _numbers(quote)


def build_messages(
    card: ModelCard,
    text: str,
    *,
    domains: Vocabulary,
    tasks: Vocabulary,
    max_chars: int,
):
    body = text if len(text) <= max_chars else _focus(text, max_chars)
    known = json.dumps(card.brief(), indent=1, default=str)[:4000]
    return [
        SystemMessage(
            content=EXTRACTION_PROMPT.format(
                domains=domains.describe(), tasks=tasks.describe()
            )
        ),
        HumanMessage(
            content=(
                f"Model: {card.name} ({card.id})\n\n"
                f"Facts already extracted (fill gaps; do not repeat them):\n{known}\n\n"
                f"<model_card>\n{body}\n</model_card>\n\n"
                "Record the facts the card states, each with an exact quote."
            )
        ),
    ]


def _focus(text: str, max_chars: int) -> str:
    """Keep the most informative sections when a card is too long."""
    card = parse_card_text(text)
    priority = re.compile(
        r"input|output|variable|description|overview|architecture|use|limitation|"
        r"hardware|requirement|data|resolution|license",
        re.I,
    )
    chosen, others = [], []
    for section in card.sections:
        block = f"## {section.title}\n{prose(section.own_text)}"
        (chosen if priority.search(section.title) else others).append(block)
    head = (
        prose("\n".join(card.clean_body.splitlines()[: card.sections[0].line]))
        if card.sections
        else ""
    )
    combined = "\n\n".join([head, *chosen, *others])
    return combined[:max_chars]


def refine_card(
    card: ModelCard,
    text: str,
    llm: Any,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
    max_chars: int = 24_000,
    config: Any = None,
) -> ModelCard:
    """Fill gaps in ``card`` with quote-verified facts read by ``llm``."""
    text = evidence_text(text)
    if not text.strip():
        return card
    extraction = invoke_structured(
        llm,
        CardExtraction,
        build_messages(
            card, text, domains=domains, tasks=tasks, max_chars=max_chars
        ),
        config=config,
        context="model card extraction",
        fallback=CardExtraction(),
        repair=1,
    )
    return merge_extraction(
        card, extraction, text, domains=domains, tasks=tasks
    )


def merge_extraction(
    card: ModelCard,
    extraction: CardExtraction,
    text: str,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> ModelCard:
    """Merge verified LLM facts into a copy of ``card``."""
    card = card.model_copy(deep=True)
    source = _normalize(text)
    accepted, rejected = 0, 0

    def ok(item: Quoted | QuotedIO | None, value: str | None = None) -> bool:
        nonlocal rejected
        if item is None:
            return False
        valid = quote_in_source(
            item.evidence, text, normalized_source=source
        ) and numbers_supported(
            value if value is not None else getattr(item, "value", ""),
            item.evidence,
        )
        if not valid:
            rejected += 1
        return valid

    def record(path: str, item: Quoted | QuotedIO) -> None:
        nonlocal accepted
        accepted += 1
        card.provenance.setdefault(
            path,
            Evidence(
                source="llm",
                method="llm",
                quote=item.evidence[:300],
                confidence=0.6,
            ),
        )

    if not card.summary and ok(extraction.summary):
        card.summary = extraction.summary.value[:600]
        record("summary", extraction.summary)
    for key, vocab, target in (
        ("domains", domains, card.domains),
        ("tasks", tasks, card.tasks),
    ):
        for item in getattr(extraction, key):
            concept = vocab.normalize(item.value)
            # The id is a label, not a quoted fact; only the quote is checked.
            if concept and concept not in target and ok(item, value=""):
                target.append(concept)
                record(f"{key}.{concept}", item)
    for item in extraction.io:
        if not ok(
            item,
            value=" ".join(filter(None, [item.shape or "", item.units or ""])),
        ):
            continue
        role_specs = card.inputs if item.role == "input" else card.outputs
        spec = next(
            (s for s in role_specs if s.name.lower() == item.name.lower()), None
        )
        if (
            spec is None
            and len(role_specs) == 1
            and role_specs[0].name in {"input", "output", "fields"}
        ):
            spec = role_specs[0]
        if spec is None:
            spec = IOSpec(name=item.name)
            role_specs.append(spec)
        spec.description = spec.description or item.description
        spec.kind = spec.kind or _kind(item.kind)
        spec.representation = spec.representation or item.representation
        if not spec.shape and item.shape:
            spec.shape = P.parse_shape(item.shape) or [item.shape]
        for key in ("dtype", "units", "normalization"):
            if not getattr(spec, key) and getattr(item, key):
                setattr(spec, key, getattr(item, key))
        if item.role == "output" and not spec.semantics:
            spec.semantics = item.semantics
        if not spec.channels and item.channels:
            spec.channels = [
                ChannelSpec(name=name) for name in item.channels if name
            ]
        record(f"{item.role}s.{spec.name}", item)
    if card.architecture.parameter_count is None and ok(
        extraction.parameter_count
    ):
        match = re.search(
            rf"(?P<num>{P.NUMBER})\s*(?P<unit>[KMBT]|thousand|million|billion)?",
            extraction.parameter_count.value,
            re.I,
        )
        count = (
            P.parse_count(match.group("num"), match.group("unit"))
            if match
            else None
        )
        if count:
            card.architecture.parameter_count = count
            card.architecture.parameter_count_text = P.format_count(count)
            record("architecture.parameter_count", extraction.parameter_count)
    for item in extraction.spatial_resolution:
        if item.value not in card.spatial.resolution and ok(item):
            card.spatial.resolution.append(item.value)
            record("spatial.resolution", item)
    if card.spatial.n_dims is None and ok(extraction.n_spatial_dims):
        digits = re.search(r"[123]", extraction.n_spatial_dims.value)
        if digits:
            card.spatial.n_dims = int(digits.group(0))
            record("spatial.n_dims", extraction.n_spatial_dims)
    for key in ("time_step", "horizon"):
        item = getattr(extraction, key)
        if getattr(card.temporal, key) is None and ok(item):
            setattr(card.temporal, key, item.value[:160])
            record(f"temporal.{key}", item)
    if card.temporal.context_steps is None and ok(extraction.context_steps):
        digits = re.search(r"\d+", extraction.context_steps.value)
        if digits:
            card.temporal.context_steps = int(digits.group(0))
            record("temporal.context_steps", extraction.context_steps)
    if card.compute.min_gpu_memory_gb is None and ok(
        extraction.min_gpu_memory_gb
    ):
        gigabytes = re.search(
            r"\d+(?:\.\d+)?", extraction.min_gpu_memory_gb.value
        )
        if gigabytes:
            card.compute.min_gpu_memory_gb = float(gigabytes.group(0))
            record("compute.min_gpu_memory_gb", extraction.min_gpu_memory_gb)
    if card.compute.cpu_supported is None and ok(
        extraction.cpu_supported, value=""
    ):
        verdict = extraction.cpu_supported.value.strip().lower()
        if verdict in {"true", "yes", "false", "no"}:
            card.compute.cpu_supported = verdict in {"true", "yes"}
            record("compute.cpu_supported", extraction.cpu_supported)
    for key in ("intended_uses", "limitations", "out_of_scope"):
        target = getattr(card, key)
        if target:
            continue
        for item in getattr(extraction, key)[:6]:
            if ok(item):
                target.append(item.value[:300])
                record(key, item)
    for item in extraction.license_restrictions:
        if item.value not in card.license.restrictions and ok(item):
            card.license.restrictions.append(item.value[:200])
            record("license.restrictions", item)
            if P.NONCOMMERCIAL.search(item.evidence) and not P.BOTH_USES.search(
                item.evidence
            ):
                card.license.commercial_use = False
    if rejected:
        card.issues.append(
            CardIssue(
                severity="info",
                code="llm_unverified",
                message=f"Dropped {rejected} LLM-proposed value(s) whose quotes or numbers were not found in the card.",
            )
        )
    if accepted:
        card.source.files = list(
            dict.fromkeys([*card.source.files, "llm_extraction"])
        )
    card.completeness = assess_completeness(card)
    return card


def _kind(value: str | None) -> str | None:
    allowed = {
        "field",
        "tensor",
        "scalar",
        "vector",
        "time_series",
        "image",
        "molecule",
        "structure",
        "graph",
        "point_cloud",
        "sequence",
        "text",
        "table",
        "file",
        "other",
    }
    value = (value or "").strip().lower().replace(" ", "_")
    return value if value in allowed else None
