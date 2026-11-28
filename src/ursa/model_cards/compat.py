"""Check a user's data against model cards and choose a pathway.

Two questions drive this module. Can the data go into the model as it is, or
only after a stated conversion? And can the user run inference directly, or
must they fine-tune first? Each check pairs one fact from the card with one
fact from the data. A check keeps the provenance key of its card fact, so an
answer can cite where the card states it.

The logic is deterministic. It never invents a model property: a fact the
card leaves out makes a check ``unknown``, and the answer lists it as a point
to confirm.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from ursa.model_cards.catalog import tokenize
from ursa.model_cards.data import DataProfile
from ursa.model_cards.schema import IOSpec, ModelCard, UserRequirements
from ursa.model_cards.taxonomy import DOMAINS, Vocabulary, best_similarity

CheckStatus = Literal["ok", "convert", "mismatch", "unknown"]
Pathway = Literal[
    "inference",
    "inference_after_conversion",
    "fine_tune",
    "not_suitable",
    "unknown",
]
PATHWAY_LABELS = {
    "inference": "Run inference as is",
    "inference_after_conversion": "Run inference after converting the data",
    "fine_tune": "Fine-tune before use",
    "not_suitable": "Not suitable for this data",
    "unknown": "Not enough information to decide",
}
_PATHWAY_ORDER = {
    "inference": 0,
    "inference_after_conversion": 1,
    "fine_tune": 2,
    "unknown": 3,
    "not_suitable": 4,
}


class Check(BaseModel):
    """One comparison between a card fact and a data fact."""

    aspect: str
    status: CheckStatus
    detail: str
    card_key: str | None = Field(
        default=None,
        description="Provenance key of the card fact, such as 'inputs.smi'.",
    )
    data_ref: str | None = Field(
        default=None, description="Where the data fact comes from."
    )
    quote: str | None = Field(
        default=None,
        description="Card line that supports the check, when the card text was searched.",
    )


class Assessment(BaseModel):
    """Whether one model can use the data, and which pathway fits."""

    model_id: str
    name: str
    pathway: Pathway
    summary: str
    checks: list[Check] = Field(default_factory=list)
    conversions: list[str] = Field(default_factory=list)
    fine_tune_notes: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        return PATHWAY_LABELS[self.pathway]


# Data kinds ------------------------------------------------------------------

_FAMILY = {
    "field": "gridded",
    "tensor": "gridded",
    "image": "gridded",
    "time_series": "series",
    "molecule": "molecule",
    "sequence": "sequence",
    "text": "sequence",
    "structure": "structure",
    "graph": "structure",
    "point_cloud": "structure",
    "table": "table",
    "scalar": "parameter",
    "vector": "parameter",
}
_FAMILY_LABEL = {
    "gridded": "gridded fields or arrays",
    "series": "time series",
    "molecule": "molecules",
    "sequence": "sequences",
    "structure": "atomic or molecular structures",
    "table": "tables",
}
_REPRESENTATION_FAMILY = (
    (re.compile(r"smiles|selfies|inchi|molecul", re.I), "molecule"),
    (re.compile(r"protein|amino|fasta|nucleotide|dna|rna", re.I), "sequence"),
    (
        re.compile(r"well|tensor|array|grid|field|image|netcdf|hdf5", re.I),
        "gridded",
    ),
    (
        re.compile(r"ase|atoms|crystal|cif|xyz|structure|graph", re.I),
        "structure",
    ),
)


def _data_spec_family(spec: IOSpec) -> str | None:
    if spec.representation:
        for pattern, family in _REPRESENTATION_FAMILY:
            if pattern.search(spec.representation):
                return family
    return _FAMILY.get(spec.kind or "")


def _data_inputs(card: ModelCard) -> list[IOSpec]:
    """Inputs that carry data, as opposed to scalar settings such as a time."""
    specs = [s for s in card.inputs if _FAMILY.get(s.kind or "") != "parameter"]
    return specs or [s for s in card.inputs if s.kind is None]


def _profile_family(profile: DataProfile) -> str | None:
    if profile.representation:
        for pattern, family in _REPRESENTATION_FAMILY:
            if pattern.search(profile.representation):
                return family
    return _FAMILY.get(profile.kind)


def _kind_check(
    card: ModelCard, profile: DataProfile
) -> tuple[Check, IOSpec | None]:
    data_family = _profile_family(profile)
    specs = _data_inputs(card)
    families = [(spec, _data_spec_family(spec)) for spec in specs]
    known = [(spec, family) for spec, family in families if family]
    if not known or data_family is None:
        return (
            Check(
                aspect="data kind",
                status="unknown",
                detail=(
                    "The card does not state what kind of data the model takes."
                    if not known
                    else "The kind of data in the file could not be determined."
                ),
                data_ref=profile.name,
            ),
            specs[0] if specs else None,
        )
    for spec, family in known:
        if family == data_family:
            return (
                Check(
                    aspect="data kind",
                    status="ok",
                    detail=(
                        f"The model takes {_FAMILY_LABEL.get(family, family)} "
                        f"(input '{spec.name}'), and the file holds {_FAMILY_LABEL.get(data_family, data_family)}."
                    ),
                    card_key=f"inputs.{spec.name}",
                    data_ref=profile.name,
                ),
                spec,
            )
    # Tables of numbers can feed array models after reshaping.
    spec, family = known[0]
    status: CheckStatus = (
        "convert"
        if {family, data_family} == {"table", "gridded"}
        else "mismatch"
    )
    return (
        Check(
            aspect="data kind",
            status=status,
            detail=(
                f"The model takes {_FAMILY_LABEL.get(family, family)} (input '{spec.name}'), "
                f"but the file holds {_FAMILY_LABEL.get(data_family, data_family)}."
            ),
            card_key=f"inputs.{spec.name}",
            data_ref=profile.name,
        ),
        spec,
    )


# Variables ---------------------------------------------------------------------

# Names that mean the same variable in common conventions (ERA5 short names,
# CF standard names, and plain words). Matching first compares normalized
# names exactly, then these groups.
_ALIAS_GROUPS = (
    {"t2m", "2t", "2mtemperature", "tas", "temperature2m", "2metretemperature"},
    {"u10", "u10m", "10u", "10muwind", "10mucomponentofwind"},
    {"v10", "v10m", "10v", "10mvwind", "10mvcomponentofwind"},
    {"u100", "u100m", "100u", "100mucomponentofwind"},
    {"v100", "v100m", "100v", "100mvcomponentofwind"},
    {"msl", "mslp", "meansealevelpressure", "pmsl", "psl"},
    {"sp", "surfacepressure", "ps"},
    {"tp", "totalprecipitation", "pr", "precipitation"},
    {"tcwv", "totalcolumnwatervapour", "totalcolumnwatervapor", "prw"},
    {"sst", "seasurfacetemperature", "tos"},
    {"z", "geopotential", "zg"},
    {"q", "specifichumidity", "hus"},
    {"t", "temperature", "ta", "airtemperature"},
    {"u", "ucomponentofwind", "ua", "uvelocity", "velocityx", "vx"},
    {"v", "vcomponentofwind", "va", "vvelocity", "velocityy", "vy"},
    {"w", "verticalvelocity", "wvelocity", "velocityz", "vz"},
    {"rho", "density", "avdensity"},
    {"p", "pressure", "avpressure"},
    {"e", "energy", "avenergy"},
)
_ALIASES = {name: group for group in _ALIAS_GROUPS for name in group}
_LEVEL_NAME = re.compile(r"^([a-z]+?)_?(\d{2,4})$")
_GENERIC_INPUTS = {
    "input",
    "inputs",
    "x",
    "data",
    "fields",
    "state",
    "states",
    "initial_state",
    "tensor",
    "array",
    "image",
    "images",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _match_variable(wanted: str, profile: DataProfile) -> str | None:
    """Return the data variable that provides the card channel ``wanted``."""
    by_norm = {_norm(v.name): v for v in profile.variables}
    key = _norm(wanted)
    if key in by_norm:
        return by_norm[key].name
    group = _ALIASES.get(key, set())
    for name in group:
        if name in by_norm:
            return by_norm[name].name
    for variable in profile.variables:
        if variable.long_name and _norm(variable.long_name) in group | {key}:
            return variable.name
    # "z500" or "u_850": a variable on pressure levels that includes the level.
    level_match = _LEVEL_NAME.match(wanted.lower())
    if level_match:
        base, level = level_match.group(1), float(level_match.group(2))
        for variable in profile.variables:
            names = {_norm(variable.name)} | _ALIASES.get(
                _norm(variable.name), set()
            )
            if base in names and level in variable.levels:
                return f"{variable.name} at level {level:g}"
    return None


def _expected_variables(card: ModelCard) -> list[tuple[str, IOSpec]]:
    """Card channels, or named field inputs, that the data must provide."""
    expected: list[tuple[str, IOSpec]] = []
    for spec in _data_inputs(card):
        if spec.channels:
            expected += [(channel.name, spec) for channel in spec.channels]
        elif (
            _FAMILY.get(spec.kind or "") in {"gridded", "series"}
            and spec.name.lower() not in _GENERIC_INPUTS
        ):
            expected.append((spec.name, spec))
    return expected


def _variable_checks(card: ModelCard, profile: DataProfile) -> list[Check]:
    expected = _expected_variables(card)
    if not expected:
        return []
    if not profile.variables:
        return [
            Check(
                aspect="variables",
                status="unknown",
                detail=(
                    f"The card lists {len(expected)} input variable(s); "
                    "the file format does not name its variables."
                ),
                card_key=f"inputs.{expected[0][1].name}",
                data_ref=profile.name,
            )
        ]
    found = [
        (name, spec, _match_variable(name, profile)) for name, spec in expected
    ]
    missing = [name for name, _, match in found if match is None]
    matched = [(name, match) for name, _, match in found if match is not None]
    spec = expected[0][1]
    # A field named in the card text ("Input field: `av_density`") has its
    # own line; cite it rather than the container input that holds it.
    key = next(
        (
            f"inputs.{name}"
            for name, _ in expected
            if f"inputs.{name}" in card.provenance
        ),
        f"inputs.{spec.name}",
    )
    if not missing:
        detail = f"All {len(expected)} input variable(s) the card lists are in the file"
        renamed = [f"{n} = {m}" for n, m in matched if _norm(n) != _norm(m)]
        if renamed:
            detail += f" ({', '.join(renamed[:6])})"
        return [
            Check(
                aspect="variables",
                status="ok",
                detail=detail + ".",
                card_key=key,
                data_ref=profile.name,
            )
        ]
    shown = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
    return [
        Check(
            aspect="variables",
            status="mismatch",
            detail=(
                f"{len(missing)} of {len(expected)} input variables the card lists "
                f"are missing from the file: {shown}."
            ),
            card_key=key,
            data_ref=profile.name,
        )
    ]


# Space and time -------------------------------------------------------------------

_DEG = re.compile(r"(\d+(?:\.\d+)?)\s*(?:deg|°|degrees?)", re.I)
_GRID = re.compile(r"(\d+)\s*[x×]\s*(\d+)(?:\s*[x×]\s*(\d+))?")


def _dims_check(card: ModelCard, profile: DataProfile) -> Check | None:
    card_dims, data_dims = card.spatial.n_dims, profile.spatial_dims
    if card_dims is None or data_dims is None:
        return None
    return Check(
        aspect="dimensions",
        status="ok" if card_dims == data_dims else "mismatch",
        detail=(
            f"The model works in {card_dims}D, and the data is {data_dims}D."
        ),
        card_key="spatial.n_dims",
        data_ref=profile.name,
    )


def _grid_check(card: ModelCard, profile: DataProfile) -> Check | None:
    stated = card.spatial.resolution
    if not stated:
        return None
    for text in stated:
        degree = _DEG.search(text)
        data_degree = _DEG.search(profile.resolution or "")
        if degree and data_degree:
            same = (
                abs(float(degree.group(1)) - float(data_degree.group(1))) < 1e-6
            )
            return Check(
                aspect="grid",
                status="ok" if same else "convert",
                detail=(
                    f"The card states a {text} grid; the data is on a {profile.resolution} grid"
                    + ("." if same else f", so regrid it to {text}.")
                ),
                card_key="spatial.resolution",
                data_ref=profile.name,
            )
    for text in stated:
        grid = _GRID.search(text)
        if grid and profile.grid:
            sizes = [int(g) for g in grid.groups() if g]
            same = (
                sizes == list(profile.grid) or sizes == list(profile.grid)[::-1]
            )
            data_grid = " x ".join(str(n) for n in profile.grid)
            return Check(
                aspect="grid",
                status="ok" if same else "convert",
                detail=(
                    f"The card states a {text} grid; the data grid is {data_grid}"
                    + ("." if same else f", so interpolate it to {text}.")
                ),
                card_key="spatial.resolution",
                data_ref=profile.name,
            )
    return None


_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*-?\s*(seconds?|secs?|s|minutes?|mins?|hours?|hrs?|h|days?|d)\b",
    re.I,
)
_TO_HOURS = {"s": 1 / 3600, "m": 1 / 60, "h": 1.0, "d": 24.0}


def _hours(text: str | None) -> float | None:
    match = _DURATION.search(text or "")
    if not match:
        return None
    unit = match.group(2).lower()
    key = "m" if unit.startswith("min") else unit[0]
    return float(match.group(1)) * _TO_HOURS[key]


def _time_check(card: ModelCard, profile: DataProfile) -> Check | None:
    card_step, data_step = (
        _hours(card.temporal.time_step),
        _hours(profile.time_step),
    )
    if card_step is None or data_step is None:
        return None
    same = abs(card_step - data_step) < 1e-9
    return Check(
        aspect="time step",
        status="ok" if same else "convert",
        detail=(
            f"The model steps by {card.temporal.time_step}; the data is spaced every {profile.time_step}"
            + ("." if same else ", so resample the data to the model's step.")
        ),
        card_key="temporal.time_step",
        data_ref=profile.name,
    )


# Units, normalization, and formats -------------------------------------------------

_UNIT_FAMILIES = (
    {"k", "kelvin", "degc", "°c", "celsius", "c"},
    {"pa", "hpa", "kpa", "mbar", "bar", "atm"},
    {"m", "km", "cm", "mm"},
    {"ms1", "m/s", "kmh", "kmh1", "m s-1"},
)


def _unit_norm(units: str) -> str:
    return re.sub(r"[\s^*]", "", units.lower())


def _units_checks(card: ModelCard, profile: DataProfile) -> list[Check]:
    checks: list[Check] = []
    by_name = {v.name: v for v in profile.variables}
    for spec in _data_inputs(card):
        pairs = [(c.name, c.units) for c in spec.channels] or [
            (spec.name, spec.units)
        ]
        for name, units in pairs:
            match = _match_variable(name, profile)
            variable = by_name.get((match or "").split(" at level ")[0])
            if not (units and variable and variable.units):
                continue
            a, b = _unit_norm(units), _unit_norm(variable.units)
            if a == b:
                continue
            convertible = any(
                a in family and b in family for family in _UNIT_FAMILIES
            )
            checks.append(
                Check(
                    aspect="units",
                    status="convert",
                    detail=(
                        f"'{variable.name}' is in {variable.units}; the model expects {units}"
                        + (
                            "; convert it."
                            if convertible
                            else "; confirm how to convert it."
                        )
                    ),
                    card_key=f"inputs.{spec.name}",
                    data_ref=f"{profile.name} › {variable.name}",
                )
            )
    return checks


def _normalization_checks(card: ModelCard) -> list[Check]:
    return [
        Check(
            aspect="normalization",
            status="convert",
            detail=f"Apply the documented input normalization: {spec.normalization}",
            card_key=f"inputs.{spec.name}",
        )
        for spec in _data_inputs(card)
        if spec.normalization
    ]


def _format_check(spec: IOSpec | None, profile: DataProfile) -> Check | None:
    if spec is None or not spec.representation:
        return None
    rep = spec.representation.lower()
    data_rep = (profile.representation or profile.format).lower()
    if rep == data_rep or rep in data_rep or data_rep in rep:
        return None
    if "well" in rep and "well" in data_rep:
        return Check(
            aspect="format",
            status="convert",
            detail=(
                "The model takes a WellFormat state; load the file with "
                "nomad.well_format.WellFormat.from_file."
            ),
            card_key=f"inputs.{spec.name}",
            data_ref=profile.name,
        )
    if "smiles" in rep and profile.kind == "molecule":
        return None
    return Check(
        aspect="format",
        status="convert",
        detail=f"The model takes {spec.representation}; the file is {profile.representation or profile.format}.",
        card_key=f"inputs.{spec.name}",
        data_ref=profile.name,
    )


def _range_checks(card: ModelCard, profile: DataProfile) -> list[Check]:
    """Whether the items in the file lie inside the range the model learned.

    A card that says its training molecules have "up to 9 heavy atoms" makes
    a prediction for a 14-atom molecule an extrapolation. The model still
    runs, so the check asks the user to confirm rather than ruling it out.
    """
    checks = []
    for domain in card.training_scope.ranges:
        sizes = profile.item_sizes.get(domain.quantity)
        if not sizes:
            continue
        outside = [n for n in sizes if not domain.contains(n)]
        noun = "molecules" if domain.quantity == "heavy atoms" else "items"
        if not outside:
            detail = (
                f"All {len(sizes)} {noun} have {min(sizes)} to {max(sizes)} {domain.quantity}, "
                f"inside the training range ({domain.text})."
            )
            status: CheckStatus = "ok"
        else:
            verb, whose = (
                ("lies", "its prediction extrapolates")
                if len(outside) == 1
                else ("lie", "their predictions extrapolate")
            )
            detail = (
                f"{len(outside)} of {len(sizes)} {noun} {verb} outside the training range "
                f"({domain.text}; the largest has {max(sizes)}), so {whose}."
            )
            status = "unknown"
        checks.append(
            Check(
                aspect="training range",
                status=status,
                detail=detail,
                card_key=f"training_scope.{domain.quantity}",
                data_ref=profile.name,
            )
        )
    return checks


# Outputs and tasks ------------------------------------------------------------------

# Words that carry no quantity: "the HOMO-LUMO gap of these molecules" asks
# for a gap; "molecules" only names what the gap belongs to.
_STOP = {
    "my",
    "your",
    "their",
    "its",
    "our",
    "these",
    "those",
    "value",
    "values",
    "data",
    "dataset",
    "file",
    "files",
    "predict",
    "prediction",
    "predictions",
    "property",
    "properties",
    "molecule",
    "molecules",
    "compound",
    "compounds",
    "sample",
    "samples",
    "system",
    "systems",
    "structure",
    "structures",
    "each",
    "all",
}
_OUTPUT_CUE = re.compile(
    r"\b(?:predict\w*|outputs?|returns?|propert\w*|targets?|regression|estimat\w*)\b",
    re.I,
)


def _stem(word: str) -> str:
    """Singular form for matching: "points" -> "point", "properties" -> "property"."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _terms(text: str) -> set[str]:
    """Singular words of ``text``, with hyphenated words also split ("HOMO-LUMO")."""
    words = set()
    for token in tokenize(text):
        words.add(_stem(token))
        words.update(
            _stem(part) for part in re.split(r"[-.]", token) if len(part) > 1
        )
    return words


def _quantity_terms(phrase: str) -> list[set[str]]:
    """Term sets for each quantity named in a wanted-output phrase."""
    from ursa.model_cards.requirements import QUANTITIES

    named = [m.group(0) for m in QUANTITIES.finditer(phrase)]
    stop = {_stem(word) for word in _STOP}
    sets = [{t for t in _terms(q) if "-" not in t} - stop for q in named] or [
        {t for t in _terms(phrase) if "-" not in t} - stop
    ]
    return [terms for terms in sets if terms]


def card_line(
    text: str | None, terms: set[str], *, require_cue: bool = False
) -> str | None:
    """The card line that mentions every term, preferring lines about outputs.

    With ``require_cue``, only a line that speaks about outputs counts, so a
    sentence such as "we do not predict the gap" in a limitations section
    never reads as an output statement.
    """
    if not text or not terms:
        return None
    fallback = None
    for raw in text.splitlines():
        line = raw.strip().lstrip("#-*>| ").strip()
        if not line or not terms <= _terms(line):
            continue
        if _OUTPUT_CUE.search(line) and not _NEGATED.search(line):
            return line[:240]
        fallback = fallback or line[:240]
    return None if require_cue else fallback


_NEGATED = re.compile(r"\b(?:not|never|cannot)\b", re.I)


def _offered_text(card: ModelCard) -> str:
    parts = [card.summary or ""] + card.intended_uses
    for spec in card.outputs:
        parts += [spec.name, spec.description or "", spec.semantics or ""]
        parts += [c.name for c in spec.channels]
    return " ".join(parts)


def _offered_by(card: ModelCard, terms: set[str]) -> str | None:
    """Provenance key of the output that states ``terms``, if any."""
    for spec in card.outputs:
        if terms <= _terms(f"{spec.name} {spec.description or ''}"):
            return f"outputs.{spec.name}"
        if terms <= _terms(" ".join(c.name for c in spec.channels)):
            channels = f"outputs.{spec.name}.channels"
            return (
                channels
                if channels in card.provenance
                else f"outputs.{spec.name}"
            )
    return None


def _outputs_check(
    card: ModelCard, req: UserRequirements | None, card_text: str | None
) -> Check | None:
    wanted = [o.name for o in (req.outputs if req else [])]
    if not wanted:
        return None
    offered = _terms(_offered_text(card))
    missing, quotes, keys = [], [], []
    for item in wanted:
        for terms in _quantity_terms(item):
            if terms <= offered:
                keys.append(_offered_by(card, terms))
                continue
            line = card_line(card_text, terms, require_cue=True)
            if line:
                quotes.append(line)
            else:
                missing.append(item)
                break
    key = next((k for k in keys if k), None) or (
        f"outputs.{card.outputs[0].name}" if card.outputs else None
    )
    if not missing:
        return Check(
            aspect="outputs",
            status="ok",
            detail=f"The card states the model provides {', '.join(wanted)}.",
            card_key=key,
            quote=None if any(keys) else (quotes[0] if quotes else None),
        )
    return Check(
        aspect="outputs",
        status="mismatch",
        detail=(
            f"The card does not say the model predicts {', '.join(missing)}"
            + (
                f"; its outputs are {', '.join(s.name for s in card.outputs[:4])}."
                if card.outputs
                else "."
            )
        ),
        card_key=f"outputs.{card.outputs[0].name}" if card.outputs else None,
    )


def _embedding_check(
    card: ModelCard, req: UserRequirements | None
) -> Check | None:
    # Only a model that returns embeddings and nothing else needs a head; a
    # fine-tuned predictor built on an encoder already has one.
    predicts = any(s.semantics in _PREDICTIONS for s in card.outputs)
    embedding = "embedding" in card.tasks or any(
        s.semantics == "embedding" for s in card.outputs
    )
    if not embedding or predicts or set(card.tasks) - {"embedding"}:
        return None
    wants_prediction = bool(
        req and (req.outputs or set(req.tasks) - {"embedding"})
    )
    if not wants_prediction:
        return None
    return Check(
        aspect="task",
        status="mismatch",
        detail=(
            "The model produces embeddings, not predictions; a prediction "
            "needs a head trained on labeled data."
        ),
        card_key="tasks.embedding",
    )


_PREDICTIONS = {"regression", "class labels", "logits", "probabilities"}


# Pathway ----------------------------------------------------------------------------

_GB_PER_PARAM_FINETUNE = 16e-9  # fp32 weights, gradients, and two Adam moments
_FEW_LABELS = 100


def _fine_tune_notes(card: ModelCard, profile: DataProfile | None) -> list[str]:
    notes = []
    if profile is not None:
        if profile.labels:
            notes.append(
                f"The data has candidate target column(s) for fine-tuning: {', '.join(profile.labels[:5])}."
            )
            if (
                profile.n_samples is not None
                and profile.n_samples < _FEW_LABELS
            ):
                notes.append(
                    f"The file has only {profile.n_samples} labeled "
                    f"example{'' if profile.n_samples == 1 else 's'}, "
                    "which is likely too few for reliable fine-tuning."
                )
        else:
            notes.append(
                "Fine-tuning needs labeled examples; the file has no column or array that looks like a target."
            )
    if card.training_scope.examples:
        notes.append(
            f"For scale, the card reports {card.training_scope.examples:,} examples "
            "in the data this model was trained or fine-tuned on."
        )
    if card.architecture.base_model:
        notes.append(
            f"The card names its base model: {card.architecture.base_model}."
        )
    repos = [i.repo_id for i in card.interfaces if i.repo_id]
    if repos:
        notes.append(f"Weights are published at {repos[0]}.")
    elif all(i.kind == "mcp_tool" for i in card.interfaces) and card.interfaces:
        notes.append(
            "The model is served only as an inference tool; fine-tuning needs its weights from the model owner."
        )
    if card.license.commercial_use is False:
        notes.append(
            "The license forbids commercial use, so check it before distributing a fine-tuned model."
        )
    if card.architecture.parameter_count:
        need = card.architecture.parameter_count * _GB_PER_PARAM_FINETUNE
        notes.append(
            f"Full fine-tuning needs roughly {need:.1f} GB of GPU memory for "
            f"{card.architecture.parameter_count / 1e6:.1f}M parameters, plus activations."
        )
    return notes


def assess_model(
    card: ModelCard,
    profile: DataProfile | None,
    req: UserRequirements | None = None,
    *,
    card_text: str | None = None,
    domains: Vocabulary = DOMAINS,
) -> Assessment:
    """Decide whether ``card`` can run on the data as is, after conversion, or after fine-tuning."""
    checks: list[Check] = []
    spec = None
    if profile is not None:
        kind, spec = _kind_check(card, profile)
        checks.append(kind)
        if kind.status != "mismatch":
            checks += _variable_checks(card, profile)
            for check in (
                _dims_check(card, profile),
                _grid_check(card, profile),
                _time_check(card, profile),
                _format_check(spec, profile),
            ):
                if check is not None:
                    checks.append(check)
            checks += _units_checks(card, profile)
            checks += _normalization_checks(card)
            checks += _range_checks(card, profile)
    for check in (
        _outputs_check(card, req, card_text),
        _embedding_check(card, req),
    ):
        if check is not None:
            checks.append(check)

    status = {c.aspect: c.status for c in checks}
    mismatched = {c.aspect for c in checks if c.status == "mismatch"}
    conversions = [c.detail for c in checks if c.status == "convert"]
    related = status.get("data kind") == "ok" or bool(
        req is not None
        and req.domains
        and best_similarity(domains, req.domains, card.domains)[0] >= 0.5
    )

    if mismatched & {"data kind", "dimensions"}:
        pathway: Pathway = "not_suitable"
        summary = next(
            c.detail
            for c in checks
            if c.aspect in mismatched & {"data kind", "dimensions"}
        )
    elif mismatched & {"outputs", "task", "variables"}:
        reason = "; ".join(
            c.detail.rstrip(".") for c in checks if c.aspect in mismatched
        )
        if related:
            remedy = (
                "Supply the missing variables, or fine-tune the model to the ones you have."
                if mismatched == {"variables"}
                else "Fine-tuning can adapt the model."
            )
            pathway, summary = "fine_tune", f"{reason}. {remedy}"
        else:
            pathway, summary = "not_suitable", reason + "."
    elif profile is None and not checks:
        pathway = "unknown"
        summary = "Name a data file to check it against the card."
    elif status.get("data kind") == "unknown":
        pathway = "unknown"
        summary = next(c.detail for c in checks if c.aspect == "data kind")
    elif conversions:
        pathway = "inference_after_conversion"
        steps = "step" if len(conversions) == 1 else "steps"
        summary = f"The data fits after {len(conversions)} conversion {steps}."
    else:
        pathway = "inference"
        summary = (
            "The data matches what the card states the model takes."
            if profile is not None
            else "The card states the model provides what you want; "
            "name a data file to check that it takes your data."
        )
    notes = _fine_tune_notes(card, profile) if pathway == "fine_tune" else []
    return Assessment(
        model_id=card.id,
        name=card.name,
        pathway=pathway,
        summary=summary,
        checks=checks,
        conversions=conversions,
        fine_tune_notes=notes,
    )


def assess_models(
    cards: Sequence[ModelCard],
    profile: DataProfile | None,
    req: UserRequirements | None = None,
    *,
    texts: dict[str, str | None] | None = None,
    domains: Vocabulary = DOMAINS,
) -> list[Assessment]:
    """Assess each card; the best pathways come first, ties keep the given order."""
    texts = texts or {}
    assessed = [
        (
            index,
            assess_model(
                card,
                profile,
                req,
                card_text=texts.get(card.id),
                domains=domains,
            ),
        )
        for index, card in enumerate(cards)
    ]
    assessed.sort(key=lambda item: (_PATHWAY_ORDER[item[1].pathway], item[0]))
    return [assessment for _, assessment in assessed]
