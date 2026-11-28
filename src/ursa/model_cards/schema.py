"""Normalized data models for scientific foundation-model cards.

Model cards arrive in many dialects: Hugging Face READMEs with YAML front
matter, DOE Genesis cards served by Nomad, NVIDIA Model Card++ pages,
free-form READMEs, bare ``config.json`` files, and MCP tool schemas. The
models in this module are the single normalized representation that the
extractors produce and that recommendation and execution consume.

Every extracted value can carry an :class:`Evidence` record in
``ModelCard.provenance`` so a reader can audit where it came from. Values
that a card does not state stay empty; the extractors never fill them with
guesses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"

Severity = Literal["info", "warning", "error"]
ExtractionMethod = Literal[
    "declared", "structured", "pattern", "llm", "inferred"
]
IOKind = Literal[
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
]
InterfaceKind = Literal[
    "mcp_tool",
    "huggingface",
    "python",
    "cli",
    "onnx",
    "container",
    "other",
]
LinkKind = Literal["paper", "code", "docs", "demo", "dataset", "model", "other"]
FitLevel = Literal["strong", "moderate", "weak", "incompatible"]


class Evidence(BaseModel):
    """Provenance for one extracted value."""

    source: str = Field(
        description=(
            "Card part that supplied the value, such as 'front_matter', "
            "'config.json', 'hub_api', 'mcp_schema', or a section path."
        )
    )
    method: ExtractionMethod = Field(description="How the value was obtained.")
    quote: str | None = Field(
        default=None, description="Verbatim snippet that supports the value."
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class CardIssue(BaseModel):
    """A quality finding about a card: conflict, gap, or caveat."""

    severity: Severity
    code: str
    message: str
    field: str | None = None


class LicenseInfo(BaseModel):
    """License terms as stated by the card."""

    spdx: str | None = Field(
        default=None,
        description="Normalized license identifier, such as 'apache-2.0'.",
    )
    name: str | None = None
    link: str | None = None
    commercial_use: bool | None = Field(
        default=None,
        description=(
            "Whether commercial use is permitted; None when the card does "
            "not settle it."
        ),
    )
    restrictions: list[str] = Field(default_factory=list)


class Link(BaseModel):
    kind: LinkKind = "other"
    url: str
    label: str | None = None


class Citation(BaseModel):
    title: str | None = None
    authors: str | None = None
    year: str | None = None
    doi: str | None = None
    arxiv: str | None = None
    url: str | None = None


class ChannelSpec(BaseModel):
    """One named channel or variable of an input or output."""

    name: str
    units: str | None = None
    description: str | None = None


class IOSpec(BaseModel):
    """Format of one model input or output."""

    name: str
    description: str | None = None
    kind: IOKind | None = None
    representation: str | None = Field(
        default=None,
        description=(
            "Concrete encoding, such as 'SMILES', 'NumPy array', "
            "'WellFormat', or 'PyTorch tensor'."
        ),
    )
    shape: list[str] = Field(
        default_factory=list,
        description="Dimension labels or sizes, such as ['H', 'W'].",
    )
    layout: str | None = None
    dtype: str | None = None
    units: str | None = None
    channels: list[ChannelSpec] = Field(default_factory=list)
    normalization: str | None = None
    semantics: str | None = Field(
        default=None,
        description=(
            "Meaning of values, such as 'absolute state', 'delta', "
            "'regression', 'logits', 'probabilities', or 'embedding'."
        ),
    )
    file_formats: list[str] = Field(default_factory=list)
    required: bool = True


class SpatialSpec(BaseModel):
    n_dims: int | None = None
    grid: str | None = None
    resolution: list[str] = Field(default_factory=list)
    boundary_conditions: str | None = None


class TemporalSpec(BaseModel):
    time_step: str | None = None
    context_steps: int | None = None
    horizon: str | None = None
    sampling_rate: str | None = None
    window: str | None = None


class ArchitectureInfo(BaseModel):
    family: str | None = None
    name: str | None = None
    parameter_count: int | None = None
    parameter_count_text: str | None = None
    framework: list[str] = Field(default_factory=list)
    base_model: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ComputeInfo(BaseModel):
    inference_hardware: str | None = None
    training_hardware: str | None = None
    min_gpu_memory_gb: float | None = None
    cpu_supported: bool | None = None
    multi_gpu_required: bool | None = None
    estimated_weights_gb: float | None = Field(
        default=None,
        description="Weight memory estimated from parameter count and dtype.",
    )


class Interface(BaseModel):
    """One way to invoke the model."""

    kind: InterfaceKind
    name: str | None = Field(
        default=None, description="Tool name, command, or loader name."
    )
    server: str | None = Field(default=None, description="MCP server name.")
    repo_id: str | None = None
    loader: str | None = Field(
        default=None,
        description="Loading call, such as 'transformers.AutoModel'.",
    )
    entry_point: str | None = Field(
        default=None, description="Python import path 'module:attribute'."
    )
    trust_remote_code: bool | None = None
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    install: list[str] = Field(default_factory=list)
    snippet: str | None = None
    notes: str | None = None


class DatasetRef(BaseModel):
    name: str
    url: str | None = None
    description: str | None = None


class DomainRange(BaseModel):
    """The range of one countable quantity that the training data covered."""

    quantity: str = Field(
        description="What is counted per item, such as 'heavy atoms' or 'residues'."
    )
    minimum: float | None = None
    maximum: float | None = None
    inclusive: bool = Field(
        default=True,
        description="False when the card says 'fewer than' the maximum.",
    )

    @property
    def text(self) -> str:
        if self.maximum is not None and self.minimum is None:
            bound = "up to" if self.inclusive else "fewer than"
            return f"{bound} {self.maximum:g} {self.quantity}"
        if self.minimum is not None and self.maximum is None:
            return f"at least {self.minimum:g} {self.quantity}"
        return f"{self.minimum:g} to {self.maximum:g} {self.quantity}"

    def contains(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None:
            return (
                value <= self.maximum
                if self.inclusive
                else value < self.maximum
            )
        return True


class TrainingScope(BaseModel):
    """How much data the model learned from, and which range it covered."""

    examples: int | None = Field(
        default=None,
        description="Training or fine-tuning examples the card states.",
    )
    ranges: list[DomainRange] = Field(default_factory=list)


class MetricResult(BaseModel):
    metric: str
    value: str
    dataset: str | None = None
    split: str | None = None
    notes: str | None = None


class CardSource(BaseModel):
    kind: Literal[
        "huggingface", "nomad", "mcp", "file", "url", "inline", "catalog"
    ]
    uri: str
    retrieved_at: str | None = None
    sha256: str | None = None
    files: list[str] = Field(default_factory=list)
    dialect: str | None = Field(
        default=None,
        description=(
            "Detected card dialect: 'huggingface', 'genesis', 'nvidia', "
            "'ursa', 'freeform', 'config_only', or 'mcp_only'."
        ),
    )


class Completeness(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    recommend_ready: bool
    run_ready: bool
    missing: list[str] = Field(default_factory=list)


class ModelCard(BaseModel):
    """Normalized scientific foundation-model card."""

    schema_version: str = SCHEMA_VERSION
    id: str
    name: str
    aliases: list[str] = Field(
        default_factory=list,
        description="Other names: card title, family name, repo id.",
    )
    version: str | None = None
    summary: str | None = None
    source: CardSource
    developers: list[str] = Field(default_factory=list)
    contact: str | None = None
    license: LicenseInfo = Field(default_factory=LicenseInfo)
    gated: bool | None = None
    links: list[Link] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    architecture: ArchitectureInfo = Field(default_factory=ArchitectureInfo)
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    spatial: SpatialSpec = Field(default_factory=SpatialSpec)
    temporal: TemporalSpec = Field(default_factory=TemporalSpec)
    interfaces: list[Interface] = Field(default_factory=list)
    training_data: list[DatasetRef] = Field(default_factory=list)
    training_scope: TrainingScope = Field(default_factory=TrainingScope)
    evaluation: list[MetricResult] = Field(default_factory=list)
    compute: ComputeInfo = Field(default_factory=ComputeInfo)
    intended_uses: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    risk_level: str | None = None
    projects: list[str] = Field(default_factory=list)
    provenance: dict[str, Evidence] = Field(default_factory=dict)
    issues: list[CardIssue] = Field(default_factory=list)
    completeness: Completeness | None = None

    def issue_codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def brief(self) -> dict[str, Any]:
        """Compact, prompt-friendly view of the card."""

        def io(specs: list[IOSpec]) -> list[dict[str, Any]]:
            items = []
            for spec in specs:
                item = spec.model_dump(
                    exclude_defaults=True, exclude={"required", "channels"}
                )
                if spec.channels:
                    item["channels"] = [
                        c.name + (f" [{c.units}]" if c.units else "")
                        for c in spec.channels[:24]
                    ]
                items.append(item)
            return items

        brief = {
            "id": self.id,
            "name": self.name,
            "summary": (self.summary or "")[:600] or None,
            "domains": self.domains,
            "tasks": self.tasks,
            "license": self.license.spdx or self.license.name,
            "commercial_use": self.license.commercial_use,
            "parameters": self.architecture.parameter_count,
            "architecture": self.architecture.name or self.architecture.family,
            "inputs": io(self.inputs),
            "outputs": io(self.outputs),
            "spatial": self.spatial.model_dump(exclude_defaults=True),
            "temporal": self.temporal.model_dump(exclude_defaults=True),
            "compute": self.compute.model_dump(exclude_defaults=True),
            "interfaces": [
                interface.kind
                + (f":{interface.name}" if interface.name else "")
                for interface in self.interfaces
            ],
            "limitations": self.limitations[:5],
            "out_of_scope": self.out_of_scope[:5],
            "issues": [f"{i.severity}:{i.code}" for i in self.issues],
        }
        return {k: v for k, v in brief.items() if v not in (None, [], {})}


class DataDescription(BaseModel):
    """Data the user has or wants, as stated in a request."""

    name: str = Field(description="Quantity or data product name.")
    description: str | None = None
    kind: str | None = Field(
        default=None,
        description=(
            "One of field, tensor, scalar, vector, time_series, image, "
            "molecule, structure, graph, point_cloud, sequence, text, "
            "table, file."
        ),
    )
    shape: str | None = None
    units: str | None = None
    format: str | None = Field(
        default=None, description="File format, such as NetCDF or SEG-Y."
    )
    path: str | None = Field(
        default=None, description="Workspace path to the data, if given."
    )


class UserRequirements(BaseModel):
    """Structured statement of what the user needs from a model."""

    summary: str = Field(
        default="", description="One-sentence restatement of the need."
    )
    domains: list[str] = Field(
        default_factory=list, description="Physics domain ids."
    )
    tasks: list[str] = Field(default_factory=list, description="Task ids.")
    inputs: list[DataDescription] = Field(default_factory=list)
    outputs: list[DataDescription] = Field(default_factory=list)
    n_spatial_dims: int | None = None
    resolution: str | None = None
    time_step: str | None = None
    gpu_memory_gb: float | None = Field(
        default=None, description="Available accelerator memory in GB."
    )
    cpu_only: bool = False
    max_parameters: int | None = None
    commercial_use: bool | None = Field(
        default=None,
        description="True when the user needs commercial-use rights.",
    )
    open_license_required: bool = False
    needs_uncertainty: bool = False
    run_requested: bool = False
    preferred_models: list[str] = Field(default_factory=list)
    excluded_models: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    data_paths: list[str] = Field(default_factory=list)


class CriterionScore(BaseModel):
    criterion: str
    score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    detail: str = ""


class Recommendation(BaseModel):
    """Ranked fit of one model against user requirements."""

    model_id: str
    name: str
    rank: int = 0
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    fit: FitLevel = "weak"
    criteria: list[CriterionScore] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    rationale: str | None = None
    interfaces: list[str] = Field(default_factory=list)
