"""Turn a user's request into structured model requirements.

Two paths produce :class:`~ursa.model_cards.schema.UserRequirements`:

- ``heuristic_requirements`` reads the request with the domain and task
  vocabularies plus patterns for hardware, licensing, size, data files,
  and run intent. It needs no LLM and never invents values.
- ``parse_requirements`` adds an LLM pass for nuance (which quantities the
  user has, which they want) and merges it with the heuristic result.
  Domain and task ids are normalized against the vocabularies; labels the
  vocabularies do not know are kept verbatim so models from new fields can
  still be matched lexically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ursa.model_cards import patterns as P
from ursa.model_cards.catalog import tokenize
from ursa.model_cards.schema import DataDescription, UserRequirements
from ursa.model_cards.taxonomy import DOMAINS, TASKS, Vocabulary
from ursa.util.structured_output import invoke_structured

# Accelerator memory for common GPUs, used when a request names the card.
GPU_MEMORY_GB = {
    r"rtx\s?(?:3090|4090)(?:\s?ti)?": 24.0,
    r"rtx\s?5090": 32.0,
    r"rtx\s?(?:3080|4080)(?:\s?ti)?": 12.0,
    r"rtx\s?(?:a6000|6000)": 48.0,
    r"\ba100\b(?![-\s]?80)": 40.0,
    r"\ba100[-\s]?80\s?gb?": 80.0,
    r"\bh100\b": 80.0,
    r"\bh200\b": 141.0,
    r"\bv100\b": 16.0,
    r"\bt4\b": 16.0,
    r"\bl4\b": 24.0,
    r"\bl40s?\b": 48.0,
    r"\ba10g?\b": 24.0,
    r"\bmi250x?\b": 64.0,
    r"\bmi300x\b": 192.0,
}
_GPU_GB = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:GB|GiB|G)\b[^.;]{0,30}?\b(?:gpu|vram|memory|card)|(?:gpu|vram)[^.;]{0,30}?(\d+(?:\.\d+)?)\s*(?:GB|GiB)\b",
    re.I,
)
_CPU_ONLY = re.compile(
    r"\bcpu[- ]only\b|\bno gpu\b|\bwithout (?:a )?gpu\b|\bonly (?:have )?(?:a )?cpu\b|\blaptop\b|"
    r"\bon (?:a |an |the |my )?cpus?\b",
    re.I,
)
_COMMERCIAL = re.compile(
    r"\bcommercial(?:ly)?\b(?!\s*/)|\bfor (?:a|my) company\b|\bin a product\b",
    re.I,
)
_NONCOMMERCIAL_OK = re.compile(r"non-?commercial|academic|research only", re.I)
_OPEN = re.compile(
    r"\bopen[- ](?:source|license|licensed|weights?)\b|\bpermissive(?:ly)? licen",
    re.I,
)
_MAX_PARAMS = re.compile(
    rf"(?:under|below|less than|fewer than|smaller than|at most|max(?:imum)?|<|<=)\s*(?P<num>{P.NUMBER})\s*(?P<unit>[KMBT]|million|billion)\b",
    re.I,
)
_RUN = re.compile(
    r"\b(?:run|execute|apply|infer|inference|predict(?:ion)? (?:for|on)|use (?:it|the model|that model) (?:on|for|to)|"
    r"compute|evaluate (?:it|the model) on)\b",
    re.I,
)
_RECOMMEND_ONLY = re.compile(
    r"\b(?:which|recommend|suggest|compare|choose|select|best)\b", re.I
)
_UNCERTAINTY = re.compile(
    r"uncertaint|ensemble|probabilistic|confidence interval|error bars?", re.I
)
_PREFER = re.compile(r"\bprefer(?:ably)?\s+(?P<name>[\w./-]{3,})", re.I)
_EXCLUDE = re.compile(
    r"\b(?:not|exclude|excluding|except|avoid)\s+(?P<name>[A-Z][\w./-]{2,})"
)
# A dot inside a word ("props.csv", "0.25") does not end the phrase.
_PHRASE = r"(?:[^.;?]|\.(?=\w)){3,160}"
_HAVE = re.compile(
    rf"\b(?:i|we) (?:have|has|got|collected|measured|simulated|generated)\s+(?P<what>{_PHRASE})|"
    rf"\bmy (?:data|dataset|input|inputs|observations|measurements)\s+(?:is|are|contain|contains|include)s?\s+(?P<what2>{_PHRASE})",
    re.I,
)
# "I have X and only a laptop CPU": the tail after the data is a constraint
# or a goal, not part of the data description.
_HAVE_TAIL = re.compile(
    r"(?:,\s*|\s+)(?:and|but|with|on|using)\s+(?:(?:only|just|a|an|one|two|\d+)\s+){0,3}"
    r"(?:[\w-]+\s+){0,2}(?:cpus?|gpus?|laptops?|workstations?|clusters?|nodes?|memory|ram|budget|"
    r"licen[cs]es?)\b.*$|"
    r"(?:,\s*|\s+)(?:and|but)\s+(?:i\s+|we\s+)?(?:want|need|would|wish|hope|plan|like)\b.*$",
    re.I | re.S,
)
_COMPUTE_ONLY = re.compile(
    r"(?:(?:only|just|access to|a|an|one|two|\d+)\s+){0,3}(?:[\w-]+\s+){0,2}"
    r"(?:cpus?|gpus?|laptops?|workstations?|clusters?|nodes?|memory|ram)"
    r"(?:\s+(?:cpus?|gpus?|memory|ram|nodes?))?",
    re.I,
)
# Requests name tasks with verbs ("simulate the flow"), while cards and the
# task vocabulary use nouns ("surrogate simulation").
_TASK_VERBS = (
    (r"\b(?:simulat(?:e|es|ing)|emulat(?:e|es|ing))\b", "surrogate_simulation"),
    (r"\binvert(?:s|ing)?\b", "inversion"),
    (r"\bdetect(?:s|ing)?\b", "detection"),
    (r"\b(?:downscale|downscales|super-?resolve)\b", "super_resolution"),
    (
        r"\b(?:interpolate|interpolates|inpaint|inpaints|fill (?:in )?(?:the )?gaps)\b",
        "interpolation",
    ),
    (
        r"\b(?:generate|design)s?\s+(?:new\s+|novel\s+|candidate\s+)?"
        r"(?:molecules?|materials?|crystals?|structures?|proteins?|samples?)\b",
        "generation",
    ),
)
_WANT = re.compile(
    r"\b(?:predicts?|forecasts?|estimates?|computes?|calculates?|obtains?|infers?|reconstructs?|"
    r"generates?|classify|classifies|detects?|segments?|"
    r"(?:to|can (?:i|we)|could (?:i|we)|want to|need to|like to) get)\s+"
    r"(?P<what>(?:[^.;,?]|\.(?=\w)){3,120})",
    re.I,
)
# A wanted quantity ends where the request turns to the data: "the boiling
# points of the molecules in data.csv" wants the boiling points.
_DATA_TAIL = re.compile(
    r"\s+(?:(?:of|for)\s+(?:the\s+|each\s+|every\s+|all\s+)?"
    r"(?:molecules?|compounds?|structures?|sequences?|proteins?|materials?|"
    r"samples?|items?|entries|rows?)\b"
    r"|(?:in|from|for|of|on)\s+(?:the\s+|my\s+|this\s+)?(?:file\s+)?"
    r"`?[\w./~-]+\.[A-Za-z][A-Za-z0-9]{0,5}\b).*$",
    re.I,
)
# "models that accept SMILES" or "take NetCDF files" name an input format.
_ACCEPTS = re.compile(
    r"\b(?:accepts?|accepting|takes?|taking|handles?|reads?|works? with|consumes?)\s+"
    r"(?:(?:a|an|the|my|our)\s+)?(?P<what>[^.;,?]{2,80})",
    re.I,
)
# Physical quantities that commonly name model inputs and outputs. Matched as
# whole phrases; a match inside a "predict ..." phrase counts as wanted output.
QUANTITIES = re.compile(
    r"\b(?:density|densities|velocity|velocities|pressure|temperature|humidity|wind(?: speed)?|"
    r"precipitation|geopotential|vorticity|stream ?function|displacement|stress(?:es)?|strain|"
    r"energ(?:y|ies)|forces?|magnetic field|electric field|current density|concentration|salinity|"
    r"permeability|porosity|saturation|wavefields?|amplitude|acoustic pressure|sound speed|"
    r"seismograms?|waveforms?|radiance|reflectance|albedo|band gaps?|homo(?:-lumo)? gaps?|dipole moments?|"
    r"polarizability|heat capacity|enthalpy|formation energy|potential energy|charge density|"
    r"refractive index|permittivity|conductivity|resistivity|sea surface temperature|sea level|"
    r"soil moisture|snow cover|surface pressure|mean sea[- ]level pressure|specific humidity|"
    r"cloud cover|radiation|irradiance|magnetization|spin|phase field|order parameter|"
    r"boiling points?|melting points?|solubilit(?:y|ies)|toxicit(?:y|ies)|lipophilicity|logp|"
    r"binding affinit(?:y|ies)|viscosit(?:y|ies)|odou?rs?|scents?)\b",
    re.I,
)
_FILE = re.compile(
    r"(?:^|[\s`'\"(])((?:\.{0,2}/|~/)?[\w.-]+(?:/[\w.-]+)*\.(?:nc|nc4|h5|hdf5|npz|npy|zarr|grib2?|tif|tiff|segy|sgy|mseed|sac|csv|tsv|parquet|cif|xyz|extxyz|pdb|json|vtk|vtu|xdmf|mat|pt|pth|smi|smiles|fasta|fa|faa|he5|png|jpe?g))\b",
    re.I,
)


class RequirementsExtraction(BaseModel):
    """Structured reading of a model request."""

    summary: str = Field(
        default="",
        description="One-sentence restatement of what the user needs.",
    )
    domains: list[str] = Field(
        default_factory=list, description="Physics domain ids from the list."
    )
    tasks: list[str] = Field(
        default_factory=list, description="Task ids from the list."
    )
    inputs: list[DataDescription] = Field(
        default_factory=list, description="Data the user has."
    )
    outputs: list[DataDescription] = Field(
        default_factory=list, description="Quantities the user wants."
    )
    n_spatial_dims: int | None = None
    resolution: str | None = None
    time_step: str | None = None
    gpu_memory_gb: float | None = None
    cpu_only: bool | None = None
    max_parameters: int | None = None
    commercial_use: bool | None = None
    open_license_required: bool | None = None
    needs_uncertainty: bool | None = None
    run_requested: bool | None = None
    preferred_models: list[str] = Field(default_factory=list)
    excluded_models: list[str] = Field(default_factory=list)


REQUIREMENTS_PROMPT = """You turn a scientist's request into model requirements.

Rules:
- Record only what the request states or clearly implies. Leave other fields empty.
- domains and tasks must use ids from the lists below.
- inputs are data the user already has; outputs are quantities the user wants.
- Set run_requested to true only when the user asks to run or apply a model now.

Physics domain ids:
{domains}

Task ids:
{tasks}
"""


def heuristic_requirements(
    text: str,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
    workspace: Path | None = None,
) -> UserRequirements:
    """Read requirements with vocabularies and patterns only."""
    text = text or ""
    req = UserRequirements(summary=re.sub(r"\s+", " ", text).strip()[:300])
    req.domains = list(dict.fromkeys(hit.id for hit in domains.match(text)))[:4]
    req.tasks = list(dict.fromkeys(hit.id for hit in tasks.match(text)))
    for pattern, task in _TASK_VERBS:
        if (
            task not in req.tasks
            and tasks.get(task)
            and re.search(pattern, text, re.I)
        ):
            req.tasks.append(task)
    req.tasks = req.tasks[:4]
    lowered = text.lower()
    memory = _GPU_GB.search(text)
    if memory:
        req.gpu_memory_gb = float(memory.group(1) or memory.group(2))
    else:
        for pattern, gigabytes in GPU_MEMORY_GB.items():
            if re.search(pattern, lowered):
                req.gpu_memory_gb = gigabytes
                break
    req.cpu_only = bool(_CPU_ONLY.search(text))
    if _COMMERCIAL.search(text) and not _NONCOMMERCIAL_OK.search(text):
        req.commercial_use = True
    req.open_license_required = bool(_OPEN.search(text))
    params = _MAX_PARAMS.search(text)
    if params:
        req.max_parameters = P.parse_count(
            params.group("num"), params.group("unit")
        )
    dims = {
        int(m.group("n"))
        if m.group("n")
        else {"one": 1, "two": 2, "three": 3}[m.group("word").lower()]
        for m in P.SPATIAL_DIMS.finditer(text)
    }
    if len(dims) == 1:
        req.n_spatial_dims = dims.pop()
    degrees = P.DEGREES.search(text)
    grid = P.GRID.search(text)
    if degrees:
        req.resolution = f"{degrees.group(1)} deg"
    elif grid:
        req.resolution = " x ".join(g for g in grid.groups() if g)
    step = P.find_time_step(text)
    if step:
        req.time_step = step[0]
    req.needs_uncertainty = bool(_UNCERTAINTY.search(text))
    req.run_requested = bool(_RUN.search(text)) and not (
        _RECOMMEND_ONLY.search(text)
        and not re.search(r"\b(?:and|then) (?:run|apply|execute)\b", lowered)
    )
    req.preferred_models = [
        m.group("name").strip(".,") for m in _PREFER.finditer(text)
    ]
    req.excluded_models = [
        m.group("name").strip(".,") for m in _EXCLUDE.finditer(text)
    ]
    for match in _HAVE.finditer(text):
        what = (match.group("what") or match.group("what2") or "").strip()
        what = _HAVE_TAIL.sub("", what).strip(" ,")
        parts = [
            part.strip()
            for part in re.split(r",\s*(?:and\s+)?|\s+and\s+", what)
            if part.strip()
        ]
        data = [part for part in parts if not _COMPUTE_ONLY.fullmatch(part)]
        if len(data) < len(parts):
            what = " and ".join(data)
        if what:
            req.inputs.append(_describe(what))
    for match in _ACCEPTS.finditer(text):
        # "take NetCDF files and forecast ..." names one input format.
        what = re.split(
            r"\s+(?:and|or|with|that|which|to|for|on|at)\s+",
            match.group("what"),
        )[0]
        accepted = _describe(what.strip())
        if accepted.kind or accepted.format:
            req.inputs.append(accepted)
    wanted_spans = []
    for match in _WANT.finditer(text):
        what = re.sub(
            r"^(?:the|a|an|their|its|our|my|your)\s+",
            "",
            match.group("what").strip(),
            flags=re.I,
        )
        what = _DATA_TAIL.sub("", what).strip()
        if what and not re.match(r"(?:it|this|that|them)\b", what, re.I):
            req.outputs.append(DataDescription(name=what[:80]))
            wanted_spans.append(match.span("what"))
    named = {item.name.lower() for item in req.inputs + req.outputs}
    for match in QUANTITIES.finditer(text):
        quantity = match.group(0).lower()
        if any(quantity in name for name in named):
            continue
        wanted = any(
            start <= match.start() < end for start, end in wanted_spans
        )
        (req.outputs if wanted else req.inputs).append(
            DataDescription(name=quantity)
        )
        named.add(quantity)
    paths = []
    for match in _FILE.finditer(text):
        candidate = match.group(1)
        local = Path(candidate).expanduser()
        resolved = (
            (Path(workspace) / local)
            if workspace and not local.is_absolute()
            else local
        )
        if resolved.exists() and candidate not in paths:
            paths.append(candidate)
    req.data_paths = paths
    # "I have mols.smi" names a file, which the entry below describes.
    names = {
        n for path in paths for n in (path, Path(path).name, Path(path).stem)
    }
    req.inputs = [item for item in req.inputs if item.name not in names]
    for path in paths:
        formats = [
            name for pattern, name in P.FILE_FORMATS if pattern.search(path)
        ]
        req.inputs.append(
            DataDescription(
                name=Path(path).name,
                path=path,
                format=formats[0] if formats else None,
            )
        )
    req.keywords = list(dict.fromkeys(tokenize(text)))[:40]
    return req


def _describe(phrase: str) -> DataDescription:
    formats = [
        name for pattern, name in P.FILE_FORMATS if pattern.search(phrase)
    ]
    kind = next(
        (k for pattern, _, k in P.REPRESENTATIONS if pattern.search(phrase)),
        None,
    )
    shape = P.parse_shape(phrase)
    return DataDescription(
        name=phrase[:80],
        kind=kind,
        format=formats[0] if formats else None,
        shape=" x ".join(shape) if shape else None,
    )


def parse_requirements(
    text: str,
    llm: Any = None,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
    workspace: Path | None = None,
    config: Any = None,
) -> UserRequirements:
    """Structured requirements from a request, LLM-assisted when available."""
    base = heuristic_requirements(
        text, domains=domains, tasks=tasks, workspace=workspace
    )
    if llm is None:
        return base
    messages = [
        SystemMessage(
            content=REQUIREMENTS_PROMPT.format(
                domains=domains.describe(), tasks=tasks.describe()
            )
        ),
        HumanMessage(content=f"<request>\n{text}\n</request>"),
    ]
    extracted = invoke_structured(
        llm,
        RequirementsExtraction,
        messages,
        config=config,
        context="model requirements",
        fallback=RequirementsExtraction(),
        repair=1,
    )
    return merge_requirements(base, extracted, domains=domains, tasks=tasks)


def merge_requirements(
    base: UserRequirements,
    extracted: RequirementsExtraction,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> UserRequirements:
    """LLM values fill or extend the heuristic reading; explicit text wins."""
    req = base.model_copy(deep=True)
    if extracted.summary:
        req.summary = extracted.summary[:300]
    llm_domains = [
        domains.normalize(d) or d.strip().lower()
        for d in extracted.domains
        if d and d.strip()
    ]
    llm_tasks = [tasks.normalize(t) for t in extracted.tasks if t]
    req.domains = list(dict.fromkeys([*llm_domains, *req.domains]))[:5]
    req.tasks = list(dict.fromkeys([t for t in llm_tasks if t] + req.tasks))[:5]
    if extracted.inputs:
        paths = [i for i in req.inputs if i.path]
        req.inputs = list(extracted.inputs) + [
            p for p in paths if p.path not in {i.path for i in extracted.inputs}
        ]
    if extracted.outputs:
        req.outputs = list(extracted.outputs)
    for key in (
        "n_spatial_dims",
        "resolution",
        "time_step",
        "gpu_memory_gb",
        "max_parameters",
        "commercial_use",
    ):
        if getattr(req, key) is None and getattr(extracted, key) is not None:
            setattr(req, key, getattr(extracted, key))
    for key in ("cpu_only", "open_license_required", "needs_uncertainty"):
        if getattr(extracted, key):
            setattr(req, key, True)
    if extracted.run_requested is not None:
        req.run_requested = bool(extracted.run_requested)
    req.preferred_models = list(
        dict.fromkeys(req.preferred_models + extracted.preferred_models)
    )
    req.excluded_models = list(
        dict.fromkeys(req.excluded_models + extracted.excluded_models)
    )
    return req


def requirements_text(req: UserRequirements) -> str:
    """Flatten requirements into text for lexical matching."""
    parts = [req.summary, *req.keywords]
    for concept_id in req.domains:
        concept = DOMAINS.get(concept_id)
        parts.append(concept.label if concept else concept_id)
    for concept_id in req.tasks:
        concept = TASKS.get(concept_id)
        parts.append(concept.label if concept else concept_id)
    for item in [*req.inputs, *req.outputs]:
        parts += [
            item.name,
            item.description or "",
            item.kind or "",
            item.format or "",
        ]
    return " ".join(p for p in parts if p)
