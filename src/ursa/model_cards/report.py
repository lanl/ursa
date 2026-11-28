"""Markdown rendering for cards, catalogs, requirements, and rankings."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from ursa.model_cards.schema import (
    IOSpec,
    ModelCard,
    Recommendation,
    UserRequirements,
)
from ursa.model_cards.taxonomy import DOMAINS, TASKS

if TYPE_CHECKING:
    from ursa.model_cards.compat import Assessment
    from ursa.model_cards.data import DataProfile
    from ursa.model_cards.matching import CatalogMatch

_SOURCE_TAGS = {
    "mcp_schema": "MCP",
    "config.json": "config",
    "hub_api": "Hub",
    "hub_api.safetensors": "Hub",
    "front_matter": "front matter",
    "key_value": "card",
    "text": "card text",
    "name": "name",
    "llm": "LLM-read",
}


def _label(vocab, concept_id: str) -> str:
    concept = vocab.get(concept_id)
    return concept.label if concept else concept_id


def _source(card: ModelCard, path: str) -> str:
    evidence = card.provenance.get(path)
    if evidence is None:
        return ""
    tag = _SOURCE_TAGS.get(evidence.source)
    if tag is None:
        tag = "LLM-read" if evidence.method == "llm" else "card"
    return f" _({tag})_"


def _io_markdown(
    specs: Sequence[IOSpec], card: ModelCard, role: str
) -> list[str]:
    lines = []
    for spec in specs:
        facts = [
            spec.kind,
            spec.representation,
            f"shape ({', '.join(spec.shape)})" if spec.shape else None,
            spec.dtype,
            f"units {spec.units}" if spec.units else None,
            spec.semantics,
            ", ".join(spec.file_formats) if spec.file_formats else None,
        ]
        text = ", ".join(f for f in facts if f) or "format not stated"
        line = f"- `{spec.name}`: {text}{_source(card, f'{role}.{spec.name}')}"
        if spec.channels:
            shown = ", ".join(
                c.name + (f" [{c.units}]" if c.units else "")
                for c in spec.channels[:16]
            )
            more = (
                f" (+{len(spec.channels) - 16} more)"
                if len(spec.channels) > 16
                else ""
            )
            line += f"\n  - channels ({len(spec.channels)}): {shown}{more}"
        if spec.normalization:
            line += f"\n  - normalization: {spec.normalization}"
        lines.append(line)
    return lines or ["- not specified by the card"]


def render_card(card: ModelCard, *, detail: bool = True) -> str:
    """Readable digest of one normalized card."""
    a = card.architecture
    lines = [
        f"### {card.name}",
        f"`{card.id}` · dialect: {card.source.dialect or 'unknown'}",
    ]
    if card.summary:
        lines += ["", card.summary]
    lines += [
        "",
        f"- **Domains:** {', '.join(_label(DOMAINS, d) for d in card.domains) or 'not stated'}",
        f"- **Tasks:** {', '.join(_label(TASKS, t) for t in card.tasks) or 'not stated'}",
        f"- **Architecture:** {a.name or a.family or 'not stated'}"
        + (f" ({a.family})" if a.family and a.name else "")
        + (
            f", {a.parameter_count_text} parameters{_source(card, 'architecture.parameter_count')}"
            if a.parameter_count
            else ""
        ),
        f"- **License:** {card.license.spdx or card.license.name or 'not stated'}"
        + (
            ""
            if card.license.commercial_use is None
            else f" (commercial use {'allowed' if card.license.commercial_use else 'not allowed'})"
        ),
    ]
    if card.spatial.model_dump(exclude_defaults=True):
        spatial = card.spatial
        parts = [
            f"{spatial.n_dims}D" if spatial.n_dims else None,
            spatial.grid,
            "; ".join(spatial.resolution) if spatial.resolution else None,
        ]
        lines.append(f"- **Spatial:** {', '.join(p for p in parts if p)}")
    temporal = card.temporal.model_dump(exclude_defaults=True)
    if temporal:
        lines.append(
            "- **Temporal:** "
            + ", ".join(
                f"{k.replace('_', ' ')} {v}" for k, v in temporal.items()
            )
        )
    compute = card.compute
    hardware = [
        f"inference: {compute.inference_hardware}"
        if compute.inference_hardware
        else None,
        f"min GPU memory {compute.min_gpu_memory_gb:g} GB"
        if compute.min_gpu_memory_gb
        else None,
        f"CPU {'supported' if compute.cpu_supported else 'not supported'}"
        if compute.cpu_supported is not None
        else None,
        f"weights ≈ {compute.estimated_weights_gb:g} GB"
        if compute.estimated_weights_gb
        else None,
    ]
    if any(hardware):
        lines.append("- **Compute:** " + "; ".join(h for h in hardware if h))
    if card.interfaces:
        how = []
        for interface in card.interfaces:
            target = (
                interface.name
                or interface.repo_id
                or interface.entry_point
                or ""
            )
            extra = f" via {interface.loader}" if interface.loader else ""
            how.append(f"{interface.kind} `{target}`{extra}")
        lines.append("- **Run via:** " + "; ".join(how))
    lines += [
        "",
        "**Inputs**",
        *_io_markdown(card.inputs, card, "inputs"),
        "",
        "**Outputs**",
        *_io_markdown(card.outputs, card, "outputs"),
    ]
    if detail:
        if card.training_data:
            lines += [
                "",
                "**Training data:** "
                + ", ".join(d.name for d in card.training_data[:8]),
            ]
        if card.evaluation:
            metrics = "; ".join(
                f"{m.metric} {m.value}"
                + (f" ({m.dataset})" if m.dataset else "")
                for m in card.evaluation[:6]
            )
            lines.append(f"**Reported metrics:** {metrics}")
        if card.intended_uses:
            lines += [
                "",
                "**Intended uses:** " + " | ".join(card.intended_uses[:4]),
            ]
        if card.limitations:
            lines.append("**Limitations:** " + " | ".join(card.limitations[:4]))
        if card.issues:
            lines += ["", "**Card quality notes**"]
            lines += [
                f"- {issue.severity}: {issue.message}" for issue in card.issues
            ]
        if card.completeness:
            c = card.completeness
            lines.append(
                f"\nCompleteness {c.score:.0%}; ready to recommend: {'yes' if c.recommend_ready else 'no'}; "
                f"ready to run: {'yes' if c.run_ready else 'no'}"
                + (f"; missing: {', '.join(c.missing)}" if c.missing else "")
            )
    return "\n".join(lines)


def render_catalog(cards: Sequence[ModelCard]) -> str:
    if not cards:
        return "The model catalog is empty."
    lines = [
        "| Model | Domains | Tasks | Params | License | Inputs → Outputs |",
        "|---|---|---|---|---|---|",
    ]
    for card in cards:
        ins = ", ".join(s.name for s in card.inputs[:3]) or "?"
        outs = _offers(card) or "?"
        lines.append(
            f"| {card.name} | {', '.join(_label(DOMAINS, d) for d in card.domains[:2]) or '?'} "
            f"| {', '.join(_label(TASKS, t) for t in card.tasks[:2]) or '?'} | "
            f"{card.architecture.parameter_count_text or '?'} | {card.license.spdx or '?'} | {ins} → {outs} |"
        )
    return "\n".join(lines)


def render_requirements(req: UserRequirements) -> str:
    facts = [
        f"domains: {', '.join(_label(DOMAINS, d) for d in req.domains)}"
        if req.domains
        else None,
        f"tasks: {', '.join(_label(TASKS, t) for t in req.tasks)}"
        if req.tasks
        else None,
        f"data available: {', '.join(i.name for i in req.inputs[:4])}"
        if req.inputs
        else None,
        f"wanted: {', '.join(o.name for o in req.outputs[:4])}"
        if req.outputs
        else None,
        f"{req.n_spatial_dims}D" if req.n_spatial_dims else None,
        f"resolution {req.resolution}" if req.resolution else None,
        f"GPU memory {req.gpu_memory_gb:g} GB" if req.gpu_memory_gb else None,
        "CPU only" if req.cpu_only else None,
        f"≤ {req.max_parameters:,} parameters" if req.max_parameters else None,
        "commercial use" if req.commercial_use else None,
        "open license" if req.open_license_required else None,
        "uncertainty estimates" if req.needs_uncertainty else None,
        f"data files: {', '.join(req.data_paths)}" if req.data_paths else None,
    ]
    return "; ".join(f for f in facts if f) or "no specific constraints"


def render_recommendations(
    recs: Sequence[Recommendation], *, limit: int = 5
) -> str:
    if not recs:
        return "No models in the catalog match the request."
    lines = []
    for rec in list(recs)[:limit]:
        lines.append(
            f"{rec.rank}. **{rec.name}** (`{rec.model_id}`): {rec.fit} fit, score {rec.score:.2f}"
        )
        if rec.rationale:
            lines.append(f"   - Why: {rec.rationale}")
        elif rec.strengths:
            lines.append(f"   - Why: {' '.join(rec.strengths[:3])}")
        scores = ", ".join(
            f"{c.criterion} {c.score:.2f}" for c in rec.criteria if c.weight > 0
        )
        lines.append(f"   - Criteria: {scores}")
        for blocker in rec.blockers:
            lines.append(f"   - Blocker: {blocker}")
        for gap in rec.gaps[:4]:
            lines.append(f"   - Check: {gap}")
    return "\n".join(lines)


# Traceable answers ---------------------------------------------------------------

_FIELD_LABELS = {
    "spatial.resolution": "spatial resolution",
    "spatial.n_dims": "spatial dimensions",
    "spatial.grid": "grid",
    "temporal.time_step": "time step",
    "license.spdx": "license",
    "compute.min_gpu_memory_gb": "GPU memory",
    "compute.cpu_supported": "CPU support",
    "architecture.parameter_count": "parameter count",
    "training_scope.examples": "training set size",
    "architecture.base_model": "base model",
}


def _field_label(key: str) -> str:
    if key in _FIELD_LABELS:
        return _FIELD_LABELS[key]
    group, _, name = key.partition(".")
    if group in {"inputs", "outputs"}:
        if name.endswith(".channels"):
            return f"channels of {group[:-1]} '{name[: -len('.channels')]}'"
        return f"{group[:-1]} '{name}'"
    if group == "training_scope":
        return f"training range ({name})"
    if group == "domains":
        return f"domain {_label(DOMAINS, name)}"
    if group == "tasks":
        return f"task {_label(TASKS, name)}"
    return key.replace("_", " ")


_CITED_SOURCES = {
    "mcp_schema": "MCP tool schema",
    "front_matter": "front matter",
    "hub_api": "Hugging Face Hub API",
    "hub_api.safetensors": "Hugging Face Hub API",
    "text": "card text",
    "key_value": "card text",
}


class Citations:
    """Numbered sources for the claims in one answer.

    Each claim about a model cites its card: the field, the source the
    extractor recorded for it (a section, ``config.json``, an MCP schema, or
    the Hub), the supporting quote when one exists, and the card's location.
    Claims about the user's data cite the file and the reader that opened it.
    """

    def __init__(self, workspace: str | Path | None = None) -> None:
        self._items: list[str] = []
        self._index: dict[tuple, int] = {}
        # Data files inside the workspace are cited by their workspace path,
        # so a shared report does not carry the user's home folders.
        self._workspace = Path(workspace).resolve() if workspace else None

    def _add(self, ident: tuple, item: str) -> str:
        if ident not in self._index:
            self._items.append(item)
            self._index[ident] = len(self._items)
        return f"[{self._index[ident]}]"

    def card(
        self, card: ModelCard, key: str | None = None, quote: str | None = None
    ) -> str:
        evidence = card.provenance.get(key) if key else None
        quote = quote or (evidence.quote if evidence else None)
        text = f"{card.name} card"
        if key:
            text += f", {_field_label(key)}"
        if quote:
            text += f": “{quote.strip()[:200]}”"
        where = [card.source.uri]
        if evidence is not None and evidence.source not in where:
            where.insert(
                0, _CITED_SOURCES.get(evidence.source, evidence.source)
            )
        elif key and evidence is None and not quote:
            where.append("exact line not recorded")
        return self._add(
            ("card", card.id, key, quote), f"{text} ({'; '.join(where)})"
        )

    def data(self, profile: DataProfile, ref: str | None = None) -> str:
        label = ref or profile.name
        where = profile.path
        if self._workspace is not None:
            try:
                relative = (
                    Path(profile.path).resolve().relative_to(self._workspace)
                )
                where = f"{relative.as_posix()} in the workspace"
            except ValueError:
                pass
        return self._add(
            ("data", profile.path, label),
            f"{label}, read with {profile.reader} from {where}",
        )

    def render(self) -> str:
        if not self._items:
            return ""
        return "## Sources\n\n" + "\n".join(
            f"[{i}] {item}" for i, item in enumerate(self._items, start=1)
        )


def render_data_profile(profile: DataProfile, citations: Citations) -> str:
    mark = citations.data(profile)
    lines = [f"**`{profile.name}`**: {profile.summary()} {mark}"]
    if profile.variables:
        shown = []
        for variable in profile.variables[:12]:
            fact = variable.name
            if variable.units:
                fact += f" [{variable.units}]"
            if variable.shape:
                fact += f" {'×'.join(str(n) for n in variable.shape)}"
            shown.append(fact)
        more = (
            f" (+{len(profile.variables) - 12} more)"
            if len(profile.variables) > 12
            else ""
        )
        lines.append(
            f"- Variables ({len(profile.variables)}): {', '.join(shown)}{more}"
        )
    if profile.examples:
        lines.append(f"- Examples: {', '.join(profile.examples)}")
    if profile.labels:
        lines.append(
            f"- Columns that could serve as fine-tuning targets: {', '.join(profile.labels[:8])}"
        )
    lines += [f"- {note}" for note in profile.notes]
    return "\n".join(lines)


_STATUS_WORDS = {
    "ok": "ok",
    "convert": "convert",
    "mismatch": "mismatch",
    "unknown": "to confirm",
}


def _offers(card: ModelCard) -> str:
    """What a model predicts, from its stated outputs."""
    parts = []
    for spec in card.outputs[:3]:
        text = spec.name
        if spec.units:
            text += f" [{spec.units}]"
        if spec.channels:
            names = ", ".join(c.name for c in spec.channels[:6])
            more = ", …" if len(spec.channels) > 6 else ""
            text += f" ({len(spec.channels)}: {names}{more})"
        parts.append(text)
    return "; ".join(parts)


_USABLE = {"inference", "inference_after_conversion"}


def render_verdict(
    assessments: Sequence[Assessment],
    cards: dict[str, ModelCard],
    citations: Citations,
    *,
    wanted: Sequence[str] = (),
    implied: Sequence[str] = (),
    profile: DataProfile | None = None,
) -> str:
    """The direct answer: which pathway the best model takes, with its source."""
    lines = []
    if implied and profile is not None:
        lines.append(
            "You did not name the quantity you want, so I take it to be the "
            f"target column(s) in `{profile.name}`: {', '.join(implied)}."
        )
    # A target column such as "refractive_index" reads as "refractive index".
    targets = ", ".join(wanted or [name.replace("_", " ") for name in implied])
    usable = [a for a in assessments if a.pathway in _USABLE]
    direct = [a for a in usable if a.pathway == "inference"]
    tune = [a for a in assessments if a.pathway == "fine_tune"]

    def cite(assessment: Assessment, aspect: str) -> str:
        card = cards.get(assessment.model_id)
        check = next((c for c in assessment.checks if c.aspect == aspect), None)
        if card is None or check is None:
            return ""
        return " " + citations.card(card, check.card_key, check.quote)

    def caution(assessment: Assessment) -> str:
        check = next(
            (
                c
                for c in assessment.checks
                if c.aspect == "training range" and c.status == "unknown"
            ),
            None,
        )
        return (
            f" Caution: {check.detail[0].lower()}{check.detail[1:]}{cite(assessment, 'training range')}"
            if check
            else ""
        )

    def shared_base(items: Sequence[Assessment]) -> str:
        """Point to the base model that several candidates were fine-tuned from."""
        bases = Counter(
            cards[a.model_id].architecture.base_model
            for a in items
            if a.model_id in cards and cards[a.model_id].architecture.base_model
        )
        if not bases:
            return ""
        base, count = bases.most_common(1)[0]
        if count < 2:
            return ""
        source = next(
            cards[a.model_id]
            for a in items
            if a.model_id in cards
            and cards[a.model_id].architecture.base_model == base
        )
        share = "All" if count == len(items) else f"{count} of"
        return (
            f" {share} {len(items)} candidates were fine-tuned from `{base}`"
            f"{' ' + citations.card(source, 'architecture.base_model')}; "
            "start from that base model and train a new head on your labeled data."
        )

    def names(items: Sequence[Assessment]) -> str:
        shown = [f"**{a.name}**" for a in items[:3]]
        more = f" (and {len(items) - 3} more)" if len(items) > 3 else ""
        return ", ".join(shown) + more

    if targets:
        if direct and profile is None:
            lines.append(
                f"**Answer: no fine-tuning is needed for {targets}.** {names(direct)} "
                f"predicts it directly.{cite(direct[0], 'outputs')} Name your data "
                "file to check that the model takes it."
            )
        elif direct:
            lines.append(
                f"**Answer: run inference as is.** {names(direct)} predicts {targets} "
                f"from this data directly, so you do not need fine-tuning.{cite(direct[0], 'outputs')}"
                + caution(direct[0])
            )
        elif usable:
            steps = len(usable[0].conversions)
            lines.append(
                f"**Answer: run inference after converting the data.** {names(usable)} "
                f"predicts {targets} once the data is converted ({steps} step(s), listed below)."
                f"{cite(usable[0], 'outputs')}"
            )
        elif tune:
            lines.append(
                f"**Answer: fine-tune.** No catalog model's card states that it predicts {targets} "
                f"from this kind of data. {names(tune)} can take the data and be fine-tuned for it."
                + shared_base(tune)
            )
        else:
            lines.append(
                f"**Answer: no catalog model fits.** No model takes this data and predicts {targets}."
            )
    elif len(usable) == 1:
        only = usable[0]
        card = cards.get(only.model_id)
        steps = len(only.conversions)
        how = (
            "as is"
            if only.pathway == "inference"
            else f"after {steps} conversion step{'s' if steps != 1 else ''} (listed below)"
        )
        offers = (
            f" and predicts {_offers(card)}" if card and card.outputs else ""
        )
        lines.append(
            f"**Answer: {only.label.lower()}.** **{only.name}** takes this data {how}{offers}."
            + caution(only)
        )
    elif usable:
        kind = (
            "as is"
            if len(direct) == len(usable)
            else "as is or after a conversion"
        )
        lines.append(
            f"**Answer: {len(usable)} models can take this data {kind}.** "
            "They predict different quantities, listed below; name the quantity you need to narrow the choice."
        )
    elif tune:
        lines.append(
            f"**Answer: fine-tune.** No model takes this data as it is; {names(tune)} could be adapted to it."
        )
    elif assessments and all(a.pathway == "not_suitable" for a in assessments):
        lines.append("**Answer: no catalog model fits this data.**")
    return "\n\n".join(lines)


def render_assessments(
    assessments: Sequence[Assessment],
    cards: dict[str, ModelCard],
    citations: Citations,
    profile: DataProfile | None = None,
    *,
    limit: int = 8,
) -> str:
    if not assessments:
        return "No catalog model could be checked against the data."
    usable = [a for a in assessments if a.pathway in _USABLE]
    # When a model already gives the wanted quantity, models that would need
    # fine-tuning are alternatives; one line each keeps them in view.
    brief = bool(usable) and any(
        c.aspect == "outputs" for a in usable for c in a.checks
    )
    shown = [
        a
        for a in assessments
        if a.pathway != "not_suitable"
        and not (brief and a.pathway == "fine_tune")
    ]
    alternatives = [
        a for a in assessments if brief and a.pathway == "fine_tune"
    ]
    unsuitable = [a for a in assessments if a.pathway == "not_suitable"]
    # Notes that every listed fine-tuning candidate shares appear once.
    tuned = [a for a in shown[:limit] if a.pathway == "fine_tune"]
    common = (
        [
            n
            for n in tuned[0].fine_tune_notes
            if all(n in a.fine_tune_notes for a in tuned)
        ]
        if len(tuned) > 1
        else []
    )
    lines = []
    for rank, assessment in enumerate(shown[:limit], start=1):
        card = cards.get(assessment.model_id)
        lines.append(
            f"{rank}. **{assessment.name}** (`{assessment.model_id}`): "
            f"**{assessment.label}.** {assessment.summary}"
        )
        outputs_mismatch = any(
            c.aspect == "outputs" and c.status == "mismatch"
            for c in assessment.checks
        )
        if card is not None and card.outputs and not outputs_mismatch:
            mark = citations.card(card, f"outputs.{card.outputs[0].name}")
            lines.append(f"   - predicts: {_offers(card)} {mark}")
        for check in assessment.checks:
            marks = ""
            if card is not None and (check.card_key or check.quote):
                marks += citations.card(card, check.card_key, check.quote)
            if profile is not None and check.data_ref:
                marks += citations.data(profile, check.data_ref)
            lines.append(
                f"   - {check.aspect}, {_STATUS_WORDS[check.status]}: {check.detail} {marks}".rstrip()
            )
        for note in assessment.fine_tune_notes:
            if note not in common:
                lines.append(f"   - fine-tuning: {note}")
    if common:
        lines.append("")
        lines.append(f"Fine-tuning notes for all {len(tuned)} candidates:")
        lines += [f"- {note}" for note in common]
    if len(shown) > limit:
        lines.append(
            f"\n{len(shown) - limit} more model(s) follow with the same or a later pathway."
        )
    if alternatives:
        lines.append("")
        lines.append(
            f"Fine-tune before use ({len(alternatives)}), since their cards state other outputs: "
            + "; ".join(
                f"{a.name} ({_offers(cards[a.model_id]) if a.model_id in cards else 'outputs not stated'})"
                for a in alternatives[:6]
            )
            + (
                f"; +{len(alternatives) - 6} more"
                if len(alternatives) > 6
                else ""
            )
            + "."
        )
    # Unsuitable models share a few reasons; one line per reason keeps the
    # answer short while still saying why each model was ruled out.
    reasons: dict[str, list[Assessment]] = {}
    for assessment in unsuitable:
        reasons.setdefault(_generic_reason(assessment.summary), []).append(
            assessment
        )
    if reasons:
        lines.append("")
    for reason, group in reasons.items():
        card = cards.get(group[0].model_id)
        check = next(
            (c for c in group[0].checks if c.status == "mismatch"), None
        )
        mark = (
            citations.card(card, check.card_key)
            if card is not None and check is not None and check.card_key
            else ""
        )
        names = ", ".join(a.name for a in group[:6])
        more = f" (+{len(group) - 6} more)" if len(group) > 6 else ""
        lines.append(
            f"Not suitable ({len(group)}): {names}{more}. {reason} {mark}".rstrip()
        )
    return "\n".join(lines)


def _generic_reason(summary: str) -> str:
    """A reason without the model-specific input name, for grouping."""
    return re.sub(r" \(input '[^']+'\)", "", summary)


def render_batch_table(rows: Sequence[dict], limit: int = 12) -> str:
    """Markdown table of batch results: one row per item, one column per value."""
    from ursa.model_cards.running import flatten_result

    table = []
    for row in rows[:limit]:
        values = flatten_result(row.get("result")) if "result" in row else {}
        item = next(iter(row.get("arguments", {}).values()), "")
        table.append((item, row.get("name"), values, row.get("error")))
    columns = list(
        dict.fromkeys(k for _, _, values, _ in table for k in values)
    )
    named = any(name for _, name, _, _ in table)
    header = ["Item"] + (["Name"] if named else []) + columns
    if any(error for *_, error in table):
        header.append("Error")
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for item, name, values, error in table:
        cells = [f"`{item}`"] + ([name or ""] if named else [])
        cells += [_cell(values.get(column)) for column in columns]
        if "Error" in header:
            cells.append(error or "")
        lines.append("| " + " | ".join(cells) + " |")
    if len(rows) > limit:
        lines.append(
            f"\n{len(rows) - limit} more row(s) are in the saved table."
        )
    return "\n".join(lines)


def _cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return "" if value is None else str(value)


_SOURCE_KIND_LABELS = {
    "huggingface": "from Hugging Face",
    "mcp": "on MCP servers",
    "nomad": "on Nomad servers",
    "file": "from local files",
    "url": "from URLs",
    "native": "from URSA cards",
}


def render_catalog_answer(
    cards: Sequence[ModelCard],
    matches: Sequence[CatalogMatch],
    maybes: Sequence[CatalogMatch],
    filters: Sequence[str],
    citations: Citations,
) -> str:
    origin = Counter(card.source.kind for card in cards)
    where = ", ".join(
        f"{count} {_SOURCE_KIND_LABELS.get(kind, kind)}"
        for kind, count in origin.most_common()
    )
    lines = [
        f"You have **{len(cards)}** model(s) in the catalog"
        + (f": {where}." if where else ".")
    ]
    if not filters:
        lines.append("")
        lines.append(render_catalog(cards))
        return "\n".join(lines)
    lines.append(
        f"With your filters ({'; '.join(filters)}), **{len(matches)}** match"
        + (
            f", and {len(maybes)} more might: their cards do not state every filtered fact."
            if maybes
            else "."
        )
    )
    if matches:
        lines += ["", "| Model | Why it matches | Source |", "|---|---|---|"]
        for match in matches:
            mark = citations.card(match.card)
            lines.append(
                f"| {match.card.name} | {'; '.join(match.reasons) or 'no filter excludes it'} | {mark} |"
            )
    if maybes:
        lines += [
            "",
            "| Might match | Not stated in the card | Source |",
            "|---|---|---|",
        ]
        for match in maybes:
            mark = citations.card(match.card)
            lines.append(
                f"| {match.card.name} | {', '.join(match.unknown)} | {mark} |"
            )
    return "\n".join(lines)
