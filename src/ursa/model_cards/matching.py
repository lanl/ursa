"""Score and rank model cards against user requirements.

Every recommendation is explainable: each criterion reports a score, a
weight, and a sentence, and the final score is their weighted mean. Hard
blockers (license conflicts, explicit hardware limits, size limits, user
exclusions) mark a model ``incompatible`` regardless of its score; gaps
list what the user must still check or do.

Criteria
--------
domain       physics-domain similarity (taxonomy-aware, partial credit)
task         task similarity
inputs       overlap between the data the user has and the model inputs
outputs      overlap between wanted quantities and model outputs
relevance    TF-IDF similarity between the request and the card text; this
             keeps matching useful for fields outside the vocabularies
practicality hardware fit, license fit, access, and how to invoke the model
evidence     how complete the card is (unknowns lower confidence)
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ursa.model_cards.catalog import lexical_scores, tokenize
from ursa.model_cards.requirements import requirements_text
from ursa.model_cards.schema import (
    CriterionScore,
    IOSpec,
    ModelCard,
    Recommendation,
    UserRequirements,
)
from ursa.model_cards.taxonomy import (
    DOMAINS,
    TASKS,
    Vocabulary,
    best_similarity,
)

DEFAULT_WEIGHTS = {
    "domain": 0.26,
    "task": 0.20,
    "inputs": 0.14,
    "outputs": 0.10,
    "relevance": 0.14,
    "practicality": 0.10,
    "evidence": 0.06,
}
_OPEN_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd-3-clause",
    "bsd-2-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "cc0-1.0",
    "cdla-permissive-2.0",
    "mpl-2.0",
    "gpl-3.0",
    "lgpl-3.0",
    "agpl-3.0",
}
_SYNONYMS = {
    "u": "velocity",
    "v": "velocity",
    "vx": "velocity",
    "vy": "velocity",
    "vz": "velocity",
    "rho": "density",
    "p": "pressure",
    "t": "temperature",
    "temp": "temperature",
    "smiles": "molecule",
    "selfies": "molecule",
    "molecular": "molecule",
    "molecules": "molecule",
    "seismograms": "seismogram",
    "waveforms": "waveform",
    "fields": "field",
    "images": "image",
    "velocities": "velocity",
    "wind": "velocity",
}


def _io_tokens(specs: Sequence[IOSpec]) -> set[str]:
    tokens: set[str] = set()
    for spec in specs:
        text = " ".join(
            [
                spec.name,
                spec.description or "",
                spec.representation or "",
                spec.kind or "",
            ]
            + [channel.name for channel in spec.channels]
        )
        tokens.update(_SYNONYMS.get(t, t) for t in tokenize(text))
    return tokens - {"input", "output", "inputs", "outputs", "model", "data"}


def _overlap(wanted: set[str], offered: set[str]) -> float:
    if not wanted or not offered:
        return 0.0
    return len(wanted & offered) / len(wanted)


def _concept_score(
    vocab: Vocabulary, wanted: list[str], offered: list[str], label: str
) -> CriterionScore:
    if not wanted:
        return CriterionScore(
            criterion=label,
            score=0.5,
            weight=0.0,
            detail=f"No {label} stated in the request.",
        )
    if not offered:
        return CriterionScore(
            criterion=label,
            score=0.25,
            weight=0.0,
            detail=f"The card does not state a {label}.",
        )
    known = [w for w in wanted if vocab.get(w)]
    free = [w for w in wanted if not vocab.get(w)]
    score, want, offer = best_similarity(
        vocab, known, [o for o in offered if vocab.get(o)]
    )
    for item in free:
        if item in offered:
            score, want, offer = 1.0, item, item

    def name(concept_id: str | None) -> str:
        concept = vocab.get(concept_id) if concept_id else None
        return concept.label if concept else str(concept_id)

    if score >= 1.0:
        detail = f"Matches {label} '{name(offer)}'."
    elif score > 0:
        detail = f"Related {label}: wanted '{name(want)}', card states '{name(offer)}'."
    else:
        detail = (
            f"Card {label}s ({', '.join(name(o) for o in offered[:3])}) differ from the request "
            f"({', '.join(name(w) for w in wanted[:3])})."
        )
    return CriterionScore(
        criterion=label, score=round(score, 3), weight=0.0, detail=detail
    )


def score_model(
    req: UserRequirements,
    card: ModelCard,
    *,
    relevance: float = 0.0,
    weights: dict[str, float] | None = None,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> Recommendation:
    """Score one card against the requirements."""
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    blockers: list[str] = []
    gaps: list[str] = []
    strengths: list[str] = []
    criteria: list[CriterionScore] = []

    domain = _concept_score(domains, req.domains, card.domains, "domain")
    task = _concept_score(tasks, req.tasks, card.tasks, "task")
    criteria += [domain, task]
    if domain.score >= 1.0:
        strengths.append(domain.detail)
    if task.score >= 1.0:
        strengths.append(task.detail)

    have = _io_tokens([
        IOSpec(name=i.name, description=i.description, kind=i.kind)
        for i in req.inputs
    ])
    want = _io_tokens([
        IOSpec(name=o.name, description=o.description, kind=o.kind)
        for o in req.outputs
    ])
    model_in, model_out = _io_tokens(card.inputs), _io_tokens(card.outputs)
    if not req.inputs:
        inputs = CriterionScore(
            criterion="inputs",
            score=0.5,
            weight=0.0,
            detail="The request does not describe available data.",
        )
    elif not card.inputs:
        inputs = CriterionScore(
            criterion="inputs",
            score=0.3,
            weight=0.0,
            detail="The card does not specify its inputs.",
        )
        gaps.append("Confirm the input format; the card does not specify it.")
    else:
        value = _overlap(have, model_in)
        kinds_user = {i.kind for i in req.inputs if i.kind}
        kinds_model = {s.kind for s in card.inputs if s.kind}
        if kinds_user & kinds_model:
            value = max(value, 0.6)
        inputs = CriterionScore(
            criterion="inputs",
            score=round(min(1.0, value), 3),
            weight=0.0,
            detail=(
                f"Shared input terms: {', '.join(sorted(have & model_in)[:6])}."
                if have & model_in
                else "Model inputs do not obviously match the data described."
            ),
        )
    if not req.outputs:
        outputs = CriterionScore(
            criterion="outputs",
            score=0.5,
            weight=0.0,
            detail="The request does not name wanted outputs.",
        )
    elif not card.outputs:
        outputs = CriterionScore(
            criterion="outputs",
            score=0.3,
            weight=0.0,
            detail="The card does not specify its outputs.",
        )
    else:
        value = _overlap(want, model_out)
        outputs = CriterionScore(
            criterion="outputs",
            score=round(value, 3),
            weight=0.0,
            detail=(
                f"Produces requested quantities: {', '.join(sorted(want & model_out)[:6])}."
                if want & model_out
                else "Model outputs do not obviously match the requested quantities."
            ),
        )
    criteria += [inputs, outputs]
    if inputs.score >= 0.6 and req.inputs and card.inputs:
        strengths.append(inputs.detail)
    if outputs.score >= 0.6 and req.outputs and card.outputs:
        strengths.append(outputs.detail)

    criteria.append(
        CriterionScore(
            criterion="relevance",
            score=round(min(1.0, max(0.0, relevance)), 3),
            weight=0.0,
            detail="Text similarity between the request and the card.",
        )
    )

    practical: list[float] = []
    # License
    lic = card.license
    if req.commercial_use:
        if lic.commercial_use is False:
            blockers.append(
                f"License '{lic.spdx or lic.name or 'unknown'}' does not permit commercial use"
                + (f" ({lic.restrictions[0]})" if lic.restrictions else "")
                + "."
            )
            practical.append(0.0)
        elif lic.commercial_use is None:
            gaps.append(
                "Verify commercial-use rights; the card's license is unclear."
            )
            practical.append(0.4)
        else:
            practical.append(1.0)
    if req.open_license_required:
        if lic.spdx in _OPEN_LICENSES and lic.commercial_use is not False:
            practical.append(1.0)
        elif lic.spdx is None:
            gaps.append("The card states no license; confirm it is open.")
            practical.append(0.3)
        else:
            blockers.append(
                f"License '{lic.spdx or lic.name}' is not an open license."
            )
            practical.append(0.0)
    # Hardware
    compute = card.compute
    if req.cpu_only:
        if compute.cpu_supported is False or compute.multi_gpu_required:
            blockers.append("The card states the model needs a GPU.")
            practical.append(0.0)
        elif compute.cpu_supported is None:
            gaps.append("CPU execution is not documented.")
            practical.append(0.5)
        else:
            strengths.append("Runs on CPU according to the card.")
            practical.append(1.0)
        if (card.architecture.parameter_count or 0) > 1_000_000_000:
            gaps.append("Over 1B parameters; CPU inference will be slow.")
    if req.gpu_memory_gb:
        stated = compute.min_gpu_memory_gb
        estimate = (
            (compute.estimated_weights_gb or 0) * 1.5 + 1.0
            if compute.estimated_weights_gb
            else None
        )
        if stated and stated > req.gpu_memory_gb:
            blockers.append(
                f"The card requires {stated:g} GB of GPU memory; {req.gpu_memory_gb:g} GB is available."
            )
            practical.append(0.0)
        elif estimate and estimate > req.gpu_memory_gb:
            gaps.append(
                f"Estimated inference memory ~{estimate:.1f} GB (from parameter count) exceeds {req.gpu_memory_gb:g} GB; "
                "reduced precision or smaller batches may be needed."
            )
            practical.append(0.3)
        elif stated or estimate:
            practical.append(1.0)
        else:
            gaps.append("Memory needs are not documented.")
            practical.append(0.6)
    # Size
    count = card.architecture.parameter_count
    if req.max_parameters:
        if count and count > req.max_parameters:
            blockers.append(
                f"{card.architecture.parameter_count_text} parameters exceed the limit of {req.max_parameters:,}."
            )
            practical.append(0.0)
        elif count is None:
            gaps.append("Parameter count is not stated.")
    # Access and invocation
    if card.gated:
        gaps.append("Weights are gated; request access first.")
        practical.append(0.6)
    kinds = {interface.kind for interface in card.interfaces}
    if "mcp_tool" in kinds:
        strengths.append("Callable now as an MCP tool.")
        practical.append(1.0)
    elif any(i.loader or i.snippet for i in card.interfaces):
        practical.append(0.8)
    elif kinds & {"cli", "onnx", "python"}:
        practical.append(0.7)
    elif kinds:
        gaps.append(
            "The card gives no loading code; consult the linked repository."
        )
        practical.append(0.5)
    else:
        practical.append(0.2)
    criteria.append(
        CriterionScore(
            criterion="practicality",
            score=round(sum(practical) / len(practical), 3)
            if practical
            else 0.5,
            weight=0.0,
            detail="Hardware, license, access, and invocation fit.",
        )
    )
    completeness = card.completeness.score if card.completeness else 0.5
    criteria.append(
        CriterionScore(
            criterion="evidence",
            score=round(completeness, 3),
            weight=0.0,
            detail=(
                "Card is well documented."
                if completeness >= 0.8
                else f"Card leaves gaps: {', '.join((card.completeness.missing if card.completeness else [])[:4])}."
            ),
        )
    )

    # Spatial and temporal fit
    dims_mismatch = bool(
        req.n_spatial_dims
        and card.spatial.n_dims
        and req.n_spatial_dims != card.spatial.n_dims
    )
    resolution_match = bool(
        req.resolution
        and any(
            req.resolution.replace(" ", "").lower()
            in r.replace(" ", "").lower()
            for r in card.spatial.resolution
        )
    )
    if resolution_match:
        strengths.append(
            f"Trained at the requested resolution ({req.resolution})."
        )
    if dims_mismatch:
        gaps.append(
            f"The model works on {card.spatial.n_dims}D data; the request describes {req.n_spatial_dims}D data."
        )
        for criterion in criteria:
            if criterion.criterion == "inputs":
                criterion.score = round(criterion.score * 0.5, 3)
    if req.needs_uncertainty:
        has_uq = "uncertainty_quantification" in card.tasks or any(
            o.semantics == "ensemble mean and spread" for o in card.outputs
        )
        if has_uq:
            strengths.append("Provides uncertainty estimates.")
        else:
            gaps.append("The card does not describe uncertainty estimates.")
    names = {
        card.id.lower(),
        card.name.lower(),
        *(a.lower() for a in card.aliases),
    }
    for excluded in req.excluded_models:
        if excluded.lower() in names or excluded.lower() in card.name.lower():
            blockers.append(f"Excluded by request ({excluded}).")
    preferred = any(
        p.lower() in names or p.lower() in card.name.lower()
        for p in req.preferred_models
    )
    if preferred:
        strengths.append("Preferred by the request.")
    for issue in card.issues:
        if issue.code in {
            "license_conflict",
            "card_inaccessible",
            "license_missing",
        }:
            gaps.append(issue.message)

    # Weighted score; criteria without information carry less weight.
    total, weight_sum = 0.0, 0.0
    for criterion in criteria:
        weight = weights.get(criterion.criterion, 0.0)
        if criterion.criterion in {
            "domain",
            "task",
            "inputs",
            "outputs",
        } and criterion.detail.startswith(("No ", "The request does not")):
            weight *= 0.3
        criterion.weight = round(weight, 3)
        total += weight * criterion.score
        weight_sum += weight
    score = total / weight_sum if weight_sum else 0.0
    if dims_mismatch:
        # A 3D surrogate does not run on 2D data (or the reverse) without
        # retraining, so a stated mismatch weighs heavily.
        score *= 0.75
    if resolution_match:
        score = min(1.0, score + 0.03)
    if preferred:
        score = min(1.0, score + 0.1)
    if blockers:
        fit = "incompatible"
    elif score >= 0.62:
        fit = "strong"
    elif score >= 0.45:
        fit = "moderate"
    else:
        fit = "weak"
    return Recommendation(
        model_id=card.id,
        name=card.name,
        score=round(score, 3),
        fit=fit,
        criteria=criteria,
        blockers=list(dict.fromkeys(blockers)),
        gaps=list(dict.fromkeys(gaps)),
        strengths=list(dict.fromkeys(strengths)),
        interfaces=[
            i.kind + (f":{i.name}" if i.name else "") for i in card.interfaces
        ],
    )


def rank_models(
    req: UserRequirements,
    cards: Sequence[ModelCard],
    *,
    top_k: int | None = None,
    weights: dict[str, float] | None = None,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> list[Recommendation]:
    """Rank cards; incompatible models sort after every compatible one."""
    if not cards:
        return []
    raw = lexical_scores(requirements_text(req), list(cards))
    top = max(raw.values()) if raw else 0.0
    recs = [
        score_model(
            req,
            card,
            relevance=(raw.get(card.id, 0.0) / top) if top > 0 else 0.0,
            weights=weights,
            domains=domains,
            tasks=tasks,
        )
        for card in cards
    ]
    recs.sort(key=lambda r: (r.fit == "incompatible", -r.score, r.name.lower()))
    for rank, rec in enumerate(recs, start=1):
        rec.rank = rank
    return recs[:top_k] if top_k else recs


_FIT_VALUE = {"strong": 1.0, "moderate": 0.65, "weak": 0.3, "incompatible": 0.0}


def blend_review(
    recs: list[Recommendation],
    review: dict[str, tuple[str, str, list[str]]],
    *,
    weight: float = 0.35,
) -> list[Recommendation]:
    """Blend an LLM fit assessment into deterministic scores.

    ``review`` maps model id to (fit, rationale, caveats). Deterministic
    blockers always win: a blocked model stays incompatible.
    """
    for rec in recs:
        if rec.model_id not in review:
            continue
        fit, rationale, caveats = review[rec.model_id]
        rec.rationale = rationale.strip() or rec.rationale
        rec.gaps = list(dict.fromkeys(rec.gaps + [c for c in caveats if c]))
        if rec.blockers:
            continue
        rec.score = round(
            (1 - weight) * rec.score + weight * _FIT_VALUE.get(fit, rec.score),
            3,
        )
        rec.fit = (
            "strong"
            if rec.score >= 0.62
            else "moderate"
            if rec.score >= 0.45
            else "weak"
        )
    recs.sort(key=lambda r: (r.fit == "incompatible", -r.score, r.name.lower()))
    for rank, rec in enumerate(recs, start=1):
        rec.rank = rank
    return recs


def explain(rec: Recommendation) -> str:
    """One-paragraph plain-language explanation of a recommendation."""
    if rec.rationale:
        return rec.rationale
    parts = []
    if rec.strengths:
        parts.append(" ".join(rec.strengths[:3]))
    weakest = sorted(
        (c for c in rec.criteria if c.weight > 0.05), key=lambda c: c.score
    )[:1]
    for criterion in weakest:
        if criterion.score < 0.5:
            parts.append(criterion.detail)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


# Catalog questions ------------------------------------------------------------

_SERVED = re.compile(
    r"\b(?:served|mcp|nomad|runnable|run (?:right )?now|can (?:i|we) (?:run|call))\b",
    re.I,
)


class CatalogMatch:
    """One catalog model and why it matches, or might match, a filter."""

    def __init__(self, card: ModelCard, reasons: list[str], unknown: list[str]):
        self.card = card
        self.reasons = reasons
        self.unknown = unknown


def catalog_filters(
    req: UserRequirements,
    request: str = "",
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> list[str]:
    """Plain-language names of the filters a request applies."""

    def label(vocab: Vocabulary, concept_id: str) -> str:
        concept = vocab.get(concept_id)
        return concept.label if concept else concept_id

    names = []
    names += [f"domain {label(domains, d)}" for d in req.domains]
    names += [f"task {label(tasks, t)}" for t in req.tasks]
    names += [
        f"takes {i.name if len(i.name.split()) <= 4 else f'{i.kind} data'}"
        for i in req.inputs
        if i.kind and not i.path
    ]
    if req.cpu_only:
        names.append("runs on a CPU")
    if req.gpu_memory_gb:
        names.append(f"fits {req.gpu_memory_gb:g} GB of GPU memory")
    if req.commercial_use:
        names.append("allows commercial use")
    if req.open_license_required:
        names.append("open license")
    if req.max_parameters:
        names.append(f"at most {req.max_parameters / 1e6:g}M parameters")
    if _SERVED.search(request):
        names.append("callable now as an MCP tool")
    return names


def filter_catalog(
    req: UserRequirements,
    cards: Sequence[ModelCard],
    request: str = "",
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> tuple[list[CatalogMatch], list[CatalogMatch]]:
    """Split cards into those that match every filter and those that might.

    A card that contradicts a filter is dropped. A card that leaves a filtered
    fact unstated is a "might" match, with the unstated facts listed, so a
    count never silently includes or excludes a model.
    """
    matches: list[CatalogMatch] = []
    maybes: list[CatalogMatch] = []
    want_kinds = {i.kind for i in req.inputs if i.kind and not i.path}
    served = bool(_SERVED.search(request))
    for card in cards:
        reasons: list[str] = []
        unknown: list[str] = []
        ok = True
        server = next(
            (
                i.server or "MCP"
                for i in card.interfaces
                if i.kind == "mcp_tool"
            ),
            None,
        )
        # A served model runs on its server, so the user's hardware does not
        # limit it; say so next to a hardware fact the card leaves out.
        remote = f" (it runs on the {server} server)" if server else ""
        if req.domains:
            score, _, offer = best_similarity(
                domains, req.domains, card.domains
            )
            if score >= 0.7:
                label = (
                    domains.get(offer).label if domains.get(offer) else offer
                )
                reasons.append(f"domain {label}")
            elif card.domains:
                ok = False
            else:
                unknown.append("domain")
        if req.tasks:
            score, _, offer = best_similarity(tasks, req.tasks, card.tasks)
            if score >= 0.7:
                label = tasks.get(offer).label if tasks.get(offer) else offer
                reasons.append(f"task {label}")
            elif card.tasks:
                ok = False
            else:
                unknown.append("task")
        if want_kinds:
            kinds = {s.kind for s in card.inputs if s.kind}
            if kinds & want_kinds:
                reasons.append(f"takes {', '.join(sorted(kinds & want_kinds))}")
            elif kinds:
                ok = False
            else:
                unknown.append("input kind")
        if req.cpu_only:
            if card.compute.cpu_supported:
                reasons.append("runs on a CPU")
            elif (
                card.compute.cpu_supported is False
                or card.compute.multi_gpu_required
            ):
                ok = False
            else:
                unknown.append("CPU support" + remote)
        if req.gpu_memory_gb:
            need = card.compute.min_gpu_memory_gb
            if need is not None:
                if need <= req.gpu_memory_gb:
                    reasons.append(f"needs {need:g} GB of GPU memory")
                else:
                    ok = False
            else:
                unknown.append("GPU memory" + remote)
        if req.commercial_use:
            if card.license.commercial_use:
                reasons.append("allows commercial use")
            elif card.license.commercial_use is False:
                ok = False
            else:
                unknown.append("commercial use")
        if req.open_license_required:
            if (
                card.license.spdx in _OPEN_LICENSES
                and card.license.commercial_use is not False
            ):
                reasons.append(f"license {card.license.spdx}")
            elif card.license.spdx:
                ok = False
            else:
                unknown.append("license")
        if req.max_parameters:
            count = card.architecture.parameter_count
            if count is not None:
                if count <= req.max_parameters:
                    reasons.append(f"{count / 1e6:.3g}M parameters")
                else:
                    ok = False
            else:
                unknown.append("parameter count")
        if served:
            if server:
                reasons.append(f"served on {server}")
            else:
                ok = False
        if not ok:
            continue
        (maybes if unknown else matches).append(
            CatalogMatch(card, reasons, unknown)
        )
    return matches, maybes
