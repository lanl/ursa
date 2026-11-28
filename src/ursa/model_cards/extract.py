"""Deterministic extraction of normalized model cards.

The extractor reads every source a model publishes and reconciles them:

- MCP tool schemas (exact runtime parameter formats, highest precedence),
- ``config.json`` (machine-readable architecture and I/O metadata),
- Hugging Face Hub metadata (safetensors parameter totals, tags, gating),
- YAML front matter (license, tags, datasets, ``model-index`` results),
- Markdown structure (key-value bullets, sections, tables, code blocks).

Values are recorded with :class:`~ursa.model_cards.schema.Evidence` and a
precedence. When two strong sources disagree, the extractor keeps the
higher-precedence value and records a ``CardIssue`` instead of silently
choosing. Facts that no source states stay empty; downstream code reports
them as gaps rather than guessing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import yaml

from ursa.model_cards import patterns as P
from ursa.model_cards.markdown import (
    GENERIC_TITLES,
    CardText,
    Section,
    find_arxiv_ids,
    find_dois,
    first_paragraph,
    logical_lines,
    parse_bibtex,
    parse_card_text,
    parse_key_values,
    parse_links,
    parse_tables,
    prose,
    strip_inline_markdown,
)
from ursa.model_cards.schema import (
    ArchitectureInfo,
    CardIssue,
    CardSource,
    ChannelSpec,
    Citation,
    Completeness,
    ComputeInfo,
    DatasetRef,
    DomainRange,
    Evidence,
    Interface,
    IOSpec,
    LicenseInfo,
    Link,
    MetricResult,
    ModelCard,
)
from ursa.model_cards.taxonomy import (
    DOMAINS,
    GENESIS_SCIENCE_DOMAINS,
    PIPELINE_TASKS,
    TASKS,
    Vocabulary,
)

# Precedence of evidence sources. Higher wins; ties keep the first value.
PRIORITY = {
    "mcp_schema": 90,
    "declared": 88,
    "hub_safetensors": 85,
    "config": 80,
    "hub_api": 75,
    "front_matter": 70,
    "key_value": 60,
    "section": 50,
    "text": 40,
    "llm": 35,
    "name": 20,
    "inferred": 10,
}
_METHOD = {
    "mcp_schema": "structured",
    "declared": "declared",
    "hub_safetensors": "structured",
    "config": "structured",
    "hub_api": "structured",
    "front_matter": "structured",
    "key_value": "structured",
    "section": "pattern",
    "text": "pattern",
    "llm": "llm",
    "name": "inferred",
    "inferred": "inferred",
}


@dataclass
class CardDocument:
    """Raw materials for one model card, before normalization."""

    source: CardSource
    text: str = ""
    config: dict[str, Any] | None = None
    hub_info: dict[str, Any] | None = None
    mcp_tool: dict[str, Any] | None = None
    native: dict[str, Any] | None = None
    id_hint: str | None = None
    name_hint: str | None = None
    repo_id: str | None = None
    access_error: str | None = None


# Canonical section keys and the titles authors use for them.
SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "short_description": ("model short description", "short description"),
    "description": (
        "model description",
        "description",
        "model details",
        "overview",
        "model overview",
        "about",
        "model summary",
        "summary",
        "introduction",
    ),
    "io": (
        "inputs and outputs",
        "input and output",
        "inputs outputs",
        "input/output",
        "inputs/outputs",
        "input",
        "inputs",
        "output",
        "outputs",
        "model inputs",
        "model outputs",
        "data parameters",
        "variables",
        "input variables",
        "output variables",
        "data format",
    ),
    "architecture": (
        "model architecture",
        "architecture",
        "model type",
        "model architecture and objective",
        "technical specifications",
    ),
    "intended_use": (
        "intended use",
        "intended uses",
        "uses",
        "direct use",
        "downstream use",
        "use case",
        "use cases",
        "primary intended users",
        "applications",
        "intended use cases",
    ),
    "out_of_scope": (
        "out-of-scope use",
        "out-of-scope uses",
        "out-of-scope use cases",
        "out of scope use",
        "out of scope",
        "misuse",
    ),
    "limitations": (
        "limitations",
        "known limitations",
        "bias risks and limitations",
        "caveats",
        "caveats and recommendations",
        "recommendations",
        "strengths and weaknesses",
        "weaknesses",
        "known weaknesses",
        "disclaimer",
    ),
    "risks": ("risks", "ethical considerations", "safety"),
    "training_data": (
        "training data",
        "training dataset",
        "training datasets",
        "pretraining data",
        "datasets",
        "data",
        "training testing and evaluation datasets",
        "complete training dataset",
    ),
    "training": (
        "training procedure",
        "training details",
        "training configuration",
        "training hyperparameters",
        "pre-training information",
        "pretraining details",
        "training",
    ),
    "evaluation": (
        "evaluation",
        "evaluation results",
        "evaluation details",
        "results",
        "metrics",
        "benchmarks",
        "benchmark",
        "performance",
        "testing",
        "trained model versions",
    ),
    "hardware": (
        "hardware",
        "compute infrastructure",
        "compute requirements",
        "system requirements",
        "requirements",
        "hardware requirements",
        "speeds sizes times",
    ),
    "software": ("software", "software integration", "dependencies"),
    "how_to_use": (
        "how to use",
        "how to get started",
        "how to get started with the model",
        "getting started",
        "usage",
        "quick start",
        "quickstart",
        "inference",
        "inference quickstart",
        "inference configuration",
        "loading the model from hugging face",
        "code snippets of how to use the model",
        "example usage",
        "install instructions",
        "installation",
        "feature extraction",
        "backbone",
        "encoder",
        "embeddings",
    ),
    "license": (
        "license",
        "model license",
        "license/terms of use",
        "terms of use",
        "use and restrictions",
        "licensing",
    ),
    "contact": (
        "contact",
        "model card contact",
        "contact info and model card authors",
        "model card authors",
    ),
    "developers": ("developed by", "authors", "model developers"),
}

_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "developers": (
        "developed by",
        "model developer",
        "model developers",
        "organization",
        "model card authors",
    ),
    "license": ("license", "licence", "model license"),
    "base_model": ("finetuned from model", "fine-tuned from", "base model"),
    "model_type": ("model type", "architecture type", "model architecture"),
    "parameters": (
        "number of model parameters",
        "number of parameters",
        "parameters",
        "parameter count",
        "params",
    ),
    "version": ("model version", "version", "model version(s)"),
    "dtype": ("model datatype", "precision", "data type", "dtype"),
    "limitations": ("disclaimer", "known limitations", "limitations"),
}

_IO_KEY = re.compile(
    r"^(?:(?:model|main|other properties related to)\s+)?"
    r"(?P<role>inputs?|outputs?)\b",
    re.I,
)
_INPUT_CUE = re.compile(
    r"\b(?:expects?|takes?|accepts?|consumes?|receives?|requires?|supports?|"
    r"operates? on)\b[^.;]{0,160}?\binputs?\b|"
    r"\binputs? (?:is|are|include|consists?|and outputs? (?:is|are))\b|"
    r"\bthe inputs? (?:fields?|data|states?|tensors?|arrays?|images?|"
    r"variables?|signals?)\b|^\s*inputs?\b|^\s*given\b|\bas inputs?\b|"
    r"\bconditions? on\b|\b(?:to )?(?:encodes?|embeds?)\s+(?:\w+\s+){0,3}?into\b",
    re.I,
)
_OUTPUT_CUE = re.compile(
    r"\b(?:predicts?|outputs?|returns?|produces?|generates?|forecasts?)\b|"
    r"\binto (?:\w+ )?embeddings?\b",
    re.I,
)
# Hugging Face fills id2label with LABEL_0, LABEL_1, ... when no names exist.
_DEFAULT_LABEL = re.compile(r"^LABEL_\d+$")
_DATA_SECTION = re.compile(r"^(?:data|dataset|datasets|training data)\b")
_MODEL_CUE = re.compile(r"\b(?:model|trained|we|our|it)\b", re.I)
_NEGATION = re.compile(
    r"\b(?:not|cannot|can't|no|without|unable to|unsuitable for|never)\b[^.;]*$",
    re.I,
)
_MODEL_SUBJECT = re.compile(
    r"\b(?:model|network|emulator|surrogate|encoder|decoder|backbone)\b", re.I
)
_ENCODER_CUE = re.compile(
    r"\bencodes?\b[^.;]{0,60}\binto\b|\binto (?:\w+ )?embeddings?\b", re.I
)
_ENUMERATION = re.compile(
    r"\b(?:including|such as|e\.g\.|variety of|range of)\b", re.I
)
_SAME_AS_INPUT = re.compile(
    r"(?:variables|channels|fields)[^.;]{0,20}same as (?:above|the input|"
    r"input)",
    re.I,
)
_SAME_SHAPE = re.compile(
    r"same (?:shape|spatial resolution|resolution|grid) as (?:the )?input|"
    r"with the same shape",
    re.I,
)
_IO_ROLE_VALUES = {
    "input": ("inputs",),
    "in": ("inputs",),
    "forcing": ("inputs",),
    "forcings": ("inputs",),
    "static": ("inputs",),
    "output": ("outputs",),
    "out": ("outputs",),
    "diagnostic": ("outputs",),
    "both": ("inputs", "outputs"),
    "prognostic": ("inputs", "outputs"),
    "input/output": ("inputs", "outputs"),
    "input and output": ("inputs", "outputs"),
}
_GENESIS_HEADINGS = {
    "model short description",
    "mission relevance",
    "contributed by",
    "papers and scientific outputs",
    "contact info and model card authors",
    "code snippets of how to use the model",
    "inputs and outputs",
}
_HF_HEADINGS = {
    "model details",
    "uses",
    "direct use",
    "bias risks and limitations",
    "how to get started with the model",
    "technical specifications",
    "model card contact",
    "environmental impact",
}
# Tags that the Hub adds automatically; they say little about intent.
_AUTO_TAGS = {
    "transformers",
    "pytorch",
    "safetensors",
    "custom_code",
    "custom-code",
    "endpoints_compatible",
    "feature-extraction",
    "image-feature-extraction",
    "onnx",
}
_NONPHYSICAL = {"embedding", "logits", "probabilities", "class labels"}
_PREDICTION_SEMANTICS = {
    "regression",
    "class labels",
    "logits",
    "probabilities",
}


# Value builder ------------------------------------------------------------


@dataclass
class _Slot:
    value: Any
    priority: int
    evidence: Evidence


@dataclass
class _Builder:
    slots: dict[str, _Slot] = field(default_factory=dict)
    issues: list[CardIssue] = field(default_factory=list)

    def set(
        self,
        path: str,
        value: Any,
        source: str,
        *,
        kind: str,
        quote: str | None = None,
        compare=None,
    ) -> None:
        if value in (None, "", [], {}):
            return
        priority = PRIORITY[kind]
        evidence = Evidence(
            source=source,
            method=_METHOD[kind],
            quote=_short(quote),
            confidence=round(min(1.0, priority / 90), 2),
        )
        current = self.slots.get(path)
        if current is None:
            self.slots[path] = _Slot(value, priority, evidence)
            return
        same = (compare or _same_value)(current.value, value)
        if not same and min(priority, current.priority) >= PRIORITY["text"]:
            keep, other = (
                (value, current.value)
                if priority > current.priority
                else (current.value, value)
            )
            self.issues.append(
                CardIssue(
                    severity="warning",
                    code="conflicting_values",
                    field=path,
                    message=(
                        f"Sources disagree on {path}: kept {keep!r}; "
                        f"{other!r} is also stated "
                        f"({current.evidence.source} vs {source})."
                    ),
                )
            )
        if priority > current.priority:
            self.slots[path] = _Slot(value, priority, evidence)

    def get(self, path: str, default: Any = None) -> Any:
        slot = self.slots.get(path)
        return slot.value if slot else default

    def provenance(self) -> dict[str, Evidence]:
        return {path: slot.evidence for path, slot in self.slots.items()}


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def _same_count(a: Any, b: Any) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if not a or not b:
        return a == b
    # Cards round counts ("approx. 158M"), so allow 15 percent slack.
    return abs(a - b) / max(a, b) <= 0.15


def _short(text: str | None, limit: int = 300) -> str | None:
    if not text:
        return None
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _sentences(text: str) -> list[str]:
    """Split prose into sentences; line breaks always end a sentence."""
    out: list[str] = []
    for line in (text or "").splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            parts = re.split(r"(?<=[.;!?])\s+(?=[A-Z0-9`(*\[])", line)
            out.extend(part.strip() for part in parts if part.strip())
    return out


def _section_matches(section: Section, key: str) -> bool:
    title = section.key
    return any(
        title == alias or title.startswith(alias + " ")
        for alias in SECTION_ALIASES[key]
    )


def _sections_for(card: CardText, key: str) -> list[Section]:
    return [s for s in card.sections if _section_matches(s, key)]


def _key_match(key: str, aliases: tuple[str, ...]) -> bool:
    lowered = re.sub(r"[^a-z0-9 /()-]", "", key.lower()).strip()
    return lowered in aliases


def _preamble(card: CardText) -> str:
    """Text before the first H2, including any H1 sections."""
    lines = card.clean_body.splitlines()
    for section in card.sections:
        if section.level >= 2:
            return "\n".join(lines[: section.line])
    return card.clean_body


# Front matter, tags, hub metadata, config, MCP ---------------------------------


def _dataset_url(name: str) -> str | None:
    if name.startswith(("http://", "https://")):
        return name
    if re.fullmatch(r"[\w.-]+/[\w.-]+", name):
        return f"https://huggingface.co/datasets/{name}"
    return None


def _handle_front_matter(fm: dict[str, Any], b: _Builder, ctx: dict) -> None:
    src = "front_matter"
    license_value = fm.get("license")
    if isinstance(license_value, list) and license_value:
        license_value = license_value[0]
    if isinstance(license_value, dict):
        ctx["template_front_matter"] = True
    elif isinstance(license_value, str):
        b.set(
            "license.spdx",
            P.normalize_license(license_value),
            src,
            kind="front_matter",
            quote=f"license: {license_value}",
        )
    for key in ("license_name", "license_link"):
        value = fm.get(key)
        if isinstance(value, str) and value.strip():
            b.set(
                f"license.{key.split('_')[1]}",
                value.strip(),
                src,
                kind="front_matter",
                quote=f"{key}: {value}",
            )
        elif isinstance(value, dict):
            ctx["template_front_matter"] = True
    if isinstance(fm.get("library_name"), str):
        b.set(
            "library_name", fm["library_name"].strip(), src, kind="front_matter"
        )
    if isinstance(fm.get("pipeline_tag"), str):
        ctx["declared_pipeline"] = fm["pipeline_tag"].strip()
    frameworks = fm.get("frameworks")
    if isinstance(frameworks, str):
        frameworks = [frameworks]
    if isinstance(frameworks, list):
        ctx["declared_frameworks"].extend(str(f).lower() for f in frameworks)
    base = fm.get("base_model")
    if isinstance(base, list) and base:
        base = base[0]
    if isinstance(base, str) and base.strip():
        b.set("architecture.base_model", base.strip(), src, kind="front_matter")
    tags = fm.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    ctx["author_tags"].extend(str(tag) for tag in tags if tag)
    datasets = fm.get("datasets") or []
    if isinstance(datasets, str):
        datasets = [datasets]
    for name in datasets:
        if isinstance(name, str) and name.strip():
            name = name.strip()
            ctx["datasets"].append(
                DatasetRef(name=name, url=_dataset_url(name))
            )
    metrics = fm.get("metrics") or []
    if isinstance(metrics, str):
        metrics = [metrics]
    ctx["declared_metrics"].extend(m for m in metrics if isinstance(m, str))
    for entry in fm.get("model-index") or []:
        if not isinstance(entry, dict):
            continue
        for result in entry.get("results") or []:
            if not isinstance(result, dict):
                continue
            dataset = result.get("dataset") or {}
            task = result.get("task") or {}
            for metric in result.get("metrics") or []:
                if not isinstance(metric, dict) or "value" not in metric:
                    continue
                ctx["metrics"].append(
                    MetricResult(
                        metric=str(
                            metric.get("name") or metric.get("type") or "metric"
                        ),
                        value=str(metric.get("value")),
                        dataset=dataset.get("name") or dataset.get("type"),
                        split=dataset.get("split"),
                        notes=task.get("name") or task.get("type"),
                    )
                )
    for key in ("model_name", "name"):
        if isinstance(fm.get(key), str) and fm[key].strip():
            ctx["declared_name"] = fm[key].strip()
            break


def _handle_metadata_blocks(card: CardText, b: _Builder) -> None:
    """Read license metadata that a card repeats in a JSON or YAML block."""
    for block in card.code_blocks:
        if block.language not in {"json", "yaml", "yml", ""}:
            continue
        if "license" not in block.code.lower():
            continue
        try:
            data = (
                json.loads(block.code)
                if block.code.lstrip().startswith("{")
                else yaml.safe_load(block.code)
            )
        except (ValueError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and isinstance(data.get("license"), str):
            b.set(
                "license.spdx",
                P.normalize_license(data["license"]),
                f"code block {block.section}".strip(),
                kind="key_value",
                quote=f"license: {data['license']}",
            )


def _handle_tags(b: _Builder, ctx: dict) -> None:
    author = list(dict.fromkeys(ctx["author_tags"]))
    author_set = set(author)
    hub_only = [
        t for t in dict.fromkeys(ctx["hub_tags"]) if t not in author_set
    ]
    for raw, weight in [(t, 3.0) for t in author] + [
        (t, 1.0) for t in hub_only
    ]:
        tag = raw.strip()
        lowered = tag.lower()
        prefix, _, value = lowered.partition(":")
        if not value:
            if lowered not in _AUTO_TAGS:
                ctx["plain_tags"].append((tag, weight))
            continue
        if prefix == "science":
            domain = GENESIS_SCIENCE_DOMAINS.get(value) or ctx[
                "domains_vocab"
            ].normalize(value.replace("-", " "))
            # Unknown science tags are kept verbatim so new fields still match.
            ctx["domain_votes"].append((
                domain or value,
                3.0,
                f"tag {tag}",
                "front_matter",
            ))
        elif prefix == "risk":
            b.set("risk_level", value, "front_matter", kind="front_matter")
        elif prefix == "project":
            if value != "model_team_name":
                ctx["projects"].append(value)
        elif prefix == "arxiv":
            ctx["arxiv"].append(value)
        elif prefix == "dataset":
            ctx["datasets"].append(
                DatasetRef(name=value, url=_dataset_url(value))
            )
        elif prefix == "license":
            b.set(
                "license.spdx",
                P.normalize_license(value),
                "hub_api",
                kind="hub_api",
                quote=tag,
            )
        elif prefix == "doi":
            ctx["dois"].append(value)
        elif prefix not in {
            "region",
            "type",
            "base_model",
            "endpoints_compatible",
        }:
            ctx["plain_tags"].append((tag, weight))


def _handle_hub_info(info: dict[str, Any], b: _Builder, ctx: dict) -> None:
    safetensors = info.get("safetensors") or {}
    total = safetensors.get("total") if isinstance(safetensors, dict) else None
    if isinstance(total, int) and total > 0:
        b.set(
            "architecture.parameter_count",
            total,
            "hub_api.safetensors",
            kind="hub_safetensors",
            quote=f"safetensors total = {total}",
            compare=_same_count,
        )
        params = safetensors.get("parameters") or {}
        if isinstance(params, dict) and len(params) == 1:
            ctx["weights_dtype"] = next(iter(params)).lower()
    if isinstance(info.get("library_name"), str):
        b.set("library_name", info["library_name"], "hub_api", kind="hub_api")
    if isinstance(info.get("pipeline_tag"), str):
        ctx["hub_pipeline"] = info["pipeline_tag"]
    gated = info.get("gated")
    if gated is not None:
        b.set(
            "gated",
            bool(gated) and str(gated).lower() != "false",
            "hub_api",
            kind="hub_api",
            quote=f"gated = {gated}",
        )
    if isinstance(info.get("tags"), list):
        ctx["hub_tags"].extend(str(tag) for tag in info["tags"])
    ctx["files"].extend(
        s["rfilename"]
        for s in info.get("siblings") or []
        if isinstance(s, dict) and isinstance(s.get("rfilename"), str)
    )
    card_data = info.get("cardData")
    if isinstance(card_data, dict) and not ctx["had_front_matter"]:
        _handle_front_matter(card_data, b, ctx)


_CONFIG_IN_CHANNELS = (
    "num_channels",
    "in_channels",
    "input_channels",
    "n_input_channels",
    "dim_in",
    "in_dim",
    "num_input_channels",
)
_CONFIG_OUT_CHANNELS = (
    "num_out_channels",
    "out_channels",
    "output_channels",
    "n_output_channels",
    "dim_out",
    "out_dim",
    "num_output_channels",
)
_CONFIG_SIZE = (
    "image_size",
    "img_size",
    "spatial_resolution",
    "input_size",
    "resolution",
    "grid_size",
)
_CONFIG_IN_NAMES = (
    "input_variables",
    "in_vars",
    "input_fields",
    "input_names",
    "variables",
)
_CONFIG_OUT_NAMES = (
    "channels",
    "output_variables",
    "out_vars",
    "output_fields",
    "output_names",
    "targets",
    "target_names",
    "properties",
    "variables",
)
_CONFIG_DETAIL_KEYS = (
    "hidden_size",
    "num_layers",
    "num_hidden_layers",
    "depths",
    "num_heads",
    "num_attention_heads",
    "patch_size",
    "embed_dim",
    "modes1",
    "hidden_channels",
    "base_channels",
    "num_timesteps",
    "max_position_embeddings",
)


def _channels_from(value: Any) -> list[ChannelSpec]:
    if isinstance(value, dict):
        value = [
            str(v) for _, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        ]
    if not isinstance(value, list):
        return []
    channels = []
    for item in value:
        if isinstance(item, str) and item.strip():
            channels.append(ChannelSpec(name=item.strip()))
        elif isinstance(item, dict):
            name = item.get("name") or item.get("id") or item.get("label")
            if name:
                channels.append(
                    ChannelSpec(
                        name=str(name),
                        units=item.get("units") or item.get("unit"),
                        description=item.get("description")
                        or item.get("long_name"),
                    )
                )
    return channels


def _handle_config(config: dict[str, Any], b: _Builder, ctx: dict) -> None:
    src = "config.json"
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        b.set(
            "architecture.name",
            str(architectures[0]),
            src,
            kind="config",
            quote=f"architectures: {architectures}",
        )
    if isinstance(config.get("model_type"), str):
        ctx["model_type"] = config["model_type"]
        b.set("architecture.name", config["model_type"], src, kind="name")
    for key in ("n_params", "num_parameters", "parameter_count"):
        if isinstance(config.get(key), int):
            b.set(
                "architecture.parameter_count",
                config[key],
                src,
                kind="config",
                quote=f"{key}: {config[key]}",
                compare=_same_count,
            )
    dtype = config.get("torch_dtype") or config.get("dtype")
    if isinstance(dtype, str):
        ctx["weights_dtype"] = dtype.lower()
    details = {
        key: config[key]
        for key in _CONFIG_DETAIL_KEYS
        if key in config and not isinstance(config[key], dict)
    }
    for key in _CONFIG_IN_CHANNELS:
        if isinstance(config.get(key), int):
            details["input_channels"] = config[key]
            break
    for key in _CONFIG_OUT_CHANNELS:
        if isinstance(config.get(key), int):
            details["output_channels"] = config[key]
            break
    for key in ("num_labels", "n_classes", "num_classes"):
        if isinstance(config.get(key), int):
            details["num_classes"] = config[key]
            break
    ctx["config_details"].update(details)
    for key in _CONFIG_SIZE:
        size = config.get(key)
        if isinstance(size, int) and not isinstance(size, bool):
            size = [size, size]
        if (
            isinstance(size, list)
            and size
            and all(isinstance(s, int) for s in size)
        ):
            ctx["resolution"].append((
                " x ".join(str(s) for s in size),
                PRIORITY["config"],
            ))
            break
    if isinstance(config.get("n_spatial_dims"), int):
        b.set(
            "spatial.n_dims",
            config["n_spatial_dims"],
            src,
            kind="config",
            quote=f"n_spatial_dims: {config['n_spatial_dims']}",
        )
    rate = config.get("sampling_rate")
    if isinstance(rate, (int, float)) and not isinstance(rate, bool):
        b.set(
            "temporal.sampling_rate",
            f"{rate:g} Hz",
            src,
            kind="config",
            quote=f"sampling_rate: {rate}",
        )
        window = config.get("window_size") or config.get("window_length")
        if isinstance(window, int):
            b.set(
                "temporal.window",
                f"{window} samples",
                src,
                kind="config",
                quote=f"window_size: {window}",
            )
    field_name = config.get("field_name")
    if isinstance(field_name, str):
        ctx["config_io"]["inputs"].append(ChannelSpec(name=field_name))
        ctx["config_io"]["outputs"].append(ChannelSpec(name=field_name))
    for key in _CONFIG_IN_NAMES:
        if key == "variables" and "channels" in config:
            continue
        channels = _channels_from(config.get(key))
        if channels:
            ctx["config_io"]["inputs"].extend(channels)
            break
    regression = any("Regression" in str(a) for a in architectures or [])
    if regression:
        ctx["config_io"]["regression_head"] = str(architectures[0])
    for key in ("id2label", "label_names", "class_names", "labels"):
        channels = _channels_from(config.get(key))
        if channels and all(_DEFAULT_LABEL.match(c.name) for c in channels):
            # Placeholder names carry no meaning; never report them as outputs.
            continue
        if channels:
            ctx["config_io"]["outputs"].extend(channels)
            ctx["config_io"]["output_semantics"] = (
                "regression" if regression else "class labels"
            )
            break
    else:
        for key in _CONFIG_OUT_NAMES:
            channels = _channels_from(config.get(key))
            if channels:
                ctx["config_io"]["outputs"].extend(channels)
                break
    if isinstance(config.get("encoding"), str):
        ctx["config_io"]["representation"] = config["encoding"]
    if config.get("auto_map"):
        ctx["remote_code"] = True
    names = [
        str(value)
        for value in (
            *(architectures or []),
            config.get("model_type"),
            field_name,
        )
        if value
    ] + [c.name for c in ctx["config_io"]["outputs"]]
    ctx["config_text"] = P.split_camel(" ; ".join(names))


def _handle_mcp_tool(tool: dict[str, Any], b: _Builder, ctx: dict) -> None:
    schema = tool.get("input_schema") or {}
    ctx["interfaces"].append(
        Interface(
            kind="mcp_tool",
            name=tool.get("name"),
            server=tool.get("server"),
            input_schema=schema or None,
            output_schema=tool.get("output_schema") or None,
            notes=_short(tool.get("description"), 400),
        )
    )
    ctx["mcp_inputs"] = io_specs_from_json_schema(schema)
    ctx["mcp_outputs"] = io_specs_from_json_schema(
        tool.get("output_schema") or {}
    )
    description = tool.get("description") or ""
    if description:
        ctx["mcp_description"] = description
        for sentence in _sentences(description):
            step = P.find_time_step(sentence)
            if step:
                b.set(
                    "temporal.time_step",
                    step[0],
                    "mcp_schema",
                    kind="mcp_schema",
                    quote=sentence,
                )


def _resolve_ref(schema: Any, root: dict) -> Any:
    ref = schema.get("$ref") if isinstance(schema, dict) else None
    if not ref or not ref.startswith("#/"):
        return schema
    node: Any = root
    for part in ref[2:].split("/"):
        node = node.get(part, {}) if isinstance(node, dict) else {}
    return node if isinstance(node, dict) and node else schema


def _is_tensor_schema(prop: dict) -> bool:
    media = f"{prop.get('contentMediaType', '')}{prop.get('x-media-type', '')}"
    return "tensor" in media.lower() or "torch.Tensor" in str(
        prop.get("description", "")
    )


def io_specs_from_json_schema(schema: dict[str, Any]) -> list[IOSpec]:
    """Read JSON-schema properties (e.g., an MCP inputSchema) as IOSpecs."""
    if not isinstance(schema, dict):
        return []
    root = schema
    schema = _resolve_ref(schema, root)
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    specs = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            continue
        resolved = _resolve_ref(prop, root)
        if (
            isinstance(resolved, dict)
            and "anyOf" in resolved
            and not resolved.get("type")
        ):
            options = [
                _resolve_ref(option, root)
                for option in resolved["anyOf"]
                if isinstance(option, dict) and option.get("type") != "null"
            ]
            resolved = options[0] if options else resolved
        description = prop.get("description") or resolved.get("description")
        spec = IOSpec(
            name=str(name),
            description=_short(description, 300),
            required=name in required,
        )
        json_type = resolved.get("type")
        sub_props = resolved.get("properties") or {}
        if _is_tensor_schema(resolved):
            spec.kind = "tensor"
            spec.representation = "serialized tensor"
            shape = re.search(r'shape "([^"]+)"', str(description or ""))
            if shape:
                spec.shape = shape.group(1).split()
        elif resolved.get("title") == "WellFormat" or {
            "t0_fields",
            "n_spatial_dims",
        } <= set(sub_props):
            spec.kind = "field"
            spec.representation = "WellFormat"
            spec.layout = (
                "t0_fields 'T ...', t1_fields 'T ... i', t2_fields 'T ... i j'"
            )
        elif json_type == "string":
            text = f"{name} {description or ''}"
            found = next(
                (
                    (rep, kind)
                    for pattern, rep, kind in P.REPRESENTATIONS
                    if pattern.search(text)
                ),
                None,
            )
            spec.kind = found[1] if found else "text"
            spec.representation = found[0] if found else None
        elif json_type in {"number", "integer", "boolean"}:
            spec.kind = "scalar"
            spec.dtype = json_type
        elif json_type == "array":
            spec.kind = "vector"
            items = resolved.get("items") or {}
            if isinstance(items, dict) and items.get("type"):
                spec.dtype = f"array[{items['type']}]"
        elif json_type == "object":
            spec.kind = "other"
        spec.units = P.find_units(str(description or ""))
        specs.append(spec)
    return specs


# Inputs and outputs from text ---------------------------------------------------


def _io_role(key: str | None) -> tuple[str, ...]:
    if not key:
        return ()
    lowered = key.lower().strip()
    if re.match(r"^inputs? and outputs?$|^inputs?/outputs?$", lowered):
        return ()
    match = _IO_KEY.match(lowered)
    if match:
        return (
            ("inputs",)
            if match.group("role").startswith("input")
            else ("outputs",)
        )
    return ()


def _statement_roles(sentence: str) -> tuple[str, ...]:
    roles = []
    if _INPUT_CUE.search(sentence):
        roles.append("inputs")
    if _OUTPUT_CUE.search(sentence):
        roles.append("outputs")
    return tuple(roles)


def _quantity_list(sentence: str) -> list[str]:
    """Pull 'density, velocity in x, and pressure' style lists."""
    patterns = (
        r"\bexpects?\s+(?P<list>.+?)\s+as\s+inputs?\b",
        r"\btakes?\s+(?P<list>.+?)\s+as\s+inputs?\b",
        r"\binputs?\s+(?:are|include|consist of)\s*:?\s*(?P<list>[^.;]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, sentence, re.I)
        if not match:
            continue
        raw = strip_inline_markdown(match.group("list"))
        parts = re.split(r",\s*(?:and\s+|or\s+)?|\s+and\s+|\s+or\s+", raw)
        names = []
        for part in parts:
            part = re.sub(r"^(?:the|a|an)\s+", "", part.strip(), flags=re.I)
            if 1 <= len(part.split()) <= 6 and not re.search(r"\d{3,}", part):
                names.append(part.strip(" ."))
        if len(names) >= 2:
            return names
    return []


def _statement_head(text: str) -> str | None:
    """Return the text before a statement's first colon if it names variables."""
    in_code = False
    for index, char in enumerate(text[:120]):
        if char == "`":
            in_code = not in_code
        elif char == ":" and not in_code:
            head = text[:index]
            if re.search(r"`[^`\n]+`\s*:", text[index + 1 :]):
                return None  # "`a`: ...; `b`: ..." defines several variables
            return head if P.BACKTICK.search(head) else None
    return None


def _analyze_statement(text: str) -> dict[str, Any]:
    """Extract I/O attributes from one statement (raw markdown allowed)."""
    plain = strip_inline_markdown(text)
    info: dict[str, Any] = {"text": plain}
    identifiers = []
    # In "Output `u`: the field after `t` steps", the head before the colon
    # names the variable; code spans after it are references, not names.
    head = _statement_head(text)
    for token in P.BACKTICK.findall(head or text):
        token = token.strip()
        if (
            P.IDENTIFIER.match(token)
            and token.lower() not in P.NOT_NAMES
            and not P.DTYPE.fullmatch(token)
        ):
            identifiers.append(token)
    info["identifiers"] = list(dict.fromkeys(identifiers))
    dims = P.parse_shape(" ".join(P.BACKTICK.findall(text)) + " " + plain)
    if dims and (
        re.search(
            r"\bshape|\bsize|\bdimension|\bgrid|\bresolution", plain, re.I
        )
        or not all(d.isdigit() for d in dims)
    ):
        info["shape"] = dims
    for pattern, rep, kind in P.REPRESENTATIONS:
        if pattern.search(text):
            info.setdefault("representation", rep)
            info.setdefault("kind", kind)
    if re.search(
        r"\bfields?\b|\bgridded\b|\bgrids?\b", plain, re.I
    ) and info.get("kind") in (None, "tensor"):
        info["kind"] = "field"
    if re.search(r"\btensors?\b", plain, re.I) and not info.get("kind"):
        info["kind"] = "tensor"
    formats = [name for pattern, name in P.FILE_FORMATS if pattern.search(text)]
    if formats:
        info["file_formats"] = list(dict.fromkeys(formats))
    dtype = P.IO_DTYPE.search(plain) or P.DTYPE.search(plain)
    if dtype:
        info["dtype"] = dtype.group(1).lower()
    for pattern, meaning in P.SEMANTICS:
        if pattern.search(plain):
            info["semantics"] = meaning
            break
    units = P.find_units(plain)
    if units:
        info["units"] = units
    elif P.DIMENSIONLESS.search(plain):
        info["units"] = "dimensionless"
    if P.NORMALIZATION.search(plain):
        info["normalization"] = _short(plain, 220)
    quantities = _quantity_list(plain)
    if quantities:
        info["quantities"] = quantities
    count = re.search(
        r"\b(\d{1,4})\s+(?:[\w-]+\s+){0,4}?(?:variables|properties|channels|fields|classes|labels|targets)\b",
        plain,
    )
    if count:
        info["count"] = int(count.group(1))
    return info


def _merge_io(spec: IOSpec, info: dict[str, Any]) -> None:
    if info.get("shape") and not spec.shape:
        spec.shape = info["shape"]
    for key in ("representation", "kind", "dtype", "units", "semantics"):
        if info.get(key) and not getattr(spec, key):
            setattr(spec, key, info[key])
    if info.get("normalization") and not spec.normalization:
        spec.normalization = info["normalization"]
    for fmt in info.get("file_formats", []):
        if fmt not in spec.file_formats:
            spec.file_formats.append(fmt)
    # A bare key such as "Input Type:" has its value on the next lines.
    if (
        not spec.description
        and info.get("text")
        and not re.fullmatch(r"[^:]{1,80}:\s*", info["text"])
    ):
        spec.description = _short(_ROLE_PREFIX.sub("", info["text"]), 300)


# The "Outputs:" key of "- **Outputs**: Regression prediction for ..." is a
# label, not part of the description.
_ROLE_PREFIX = re.compile(r"^\s*(?:model\s+)?(?:inputs?|outputs?)\s*:\s*", re.I)
# "Regression prediction for boiling point (°C)" names what an output holds.
_PREDICTED = re.compile(
    r"\b(?:predictions?|estimates?|classification|regression)\s+(?:of|for)\s+"
    r"(?:the\s+)?(?P<what>[^.;:()\[\]]{3,60}?)\s*(?=[(\[.;:,]|$)",
    re.I,
)


def _predicted_quantity(text: str | None) -> str | None:
    """The quantity an output statement says the model predicts."""
    match = _PREDICTED.search(text or "")
    if not match:
        return None
    what = re.sub(r"^\d[\d,]*\s+(?:possible\s+)?", "", match.group("what"))
    what = what.strip()
    return what if 1 <= len(what.split()) <= 5 else None


def _counted_list(text: str, count: int) -> tuple[list[str], str | None]:
    """Names in "12 properties (HOMO, LUMO, gap, ...)" when the list has ``count`` items.

    Cards often name their outputs only in a code comment or a sentence. The
    stated count must equal the number of names, so an unrelated parenthesis
    never reads as a channel list. Returns the names and the line they are on.
    """
    pattern = re.compile(
        rf"\b{count}\s+(?:[\w-]+\s+){{0,4}}?(?:variables|properties|channels|"
        r"fields|classes|labels|targets|outputs|quantities)\s*\((?P<list>[^()]+)\)",
        re.I,
    )
    for match in pattern.finditer(text):
        names = [
            part.strip(" .`'\"")
            for part in re.split(r",\s*(?:and\s+)?", match.group("list"))
        ]
        names = [n for n in names if n and len(n.split()) <= 4]
        if len(names) == count:
            start = text.rfind("\n", 0, match.start()) + 1
            end = text.find("\n", match.end())
            line = text[start : end if end >= 0 else len(text)]
            return names, line.strip().lstrip("#-*> ").strip()
    return [], None


def _rename_evidence(ctx: dict, role: str, old: str, new: str) -> None:
    """Keep I/O evidence keyed by the spec's final name."""
    ctx["io_evidence"] = [
        (r, new if (r, n) == (role, old) else n, source, text)
        for r, n, source, text in ctx["io_evidence"]
    ]


def _line_key(line: str) -> tuple[str | None, str]:
    body = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line.strip())
    match = re.match(
        r"^\*\*(?P<key>[^*]{1,80}?):?\*\*:?\s*(?P<rest>.*)$", body
    ) or re.match(
        r"^(?P<key>[A-Za-z][\w /()&'.-]{0,60}?):\s+(?P<rest>.+)$", body
    )
    if match:
        return strip_inline_markdown(match.group("key")).strip(" :"), body
    return None, body


def _collect_io_statements(card: CardText) -> dict[str, list[tuple[str, str]]]:
    """Group raw I/O statements by role, each with its source label."""
    groups: dict[str, list[tuple[int, str, str]]] = {
        "inputs": [],
        "outputs": [],
    }
    seen: set[tuple[str, str]] = set()

    def add(
        roles: tuple[str, ...], text: str, source: str, rank: int = 0
    ) -> None:
        for role in roles:
            key = (role, re.sub(r"\s+", " ", text.strip().lower()))
            if text.strip() and key not in seen:
                seen.add(key)
                groups[role].append((rank, text.strip(), source))

    def scan(
        text: str,
        label: str,
        *,
        io_section: bool,
        section_roles: tuple[str, ...],
        strict: bool = False,
    ):
        for line in logical_lines(text):
            if line.startswith("#"):
                continue
            key, body = _line_key(line)
            roles = _io_role(key)
            if roles:
                add(roles, body, f"{label} > {key}")
            elif io_section:
                add(_statement_roles(body) or section_roles, body, label)
            else:
                for sentence in _sentences(body):
                    if strict and not (
                        _MODEL_SUBJECT.search(sentence)
                        or _ENCODER_CUE.search(sentence)
                    ):
                        # Usage and data sections also describe scripts and
                        # datasets; only sentences about the model count.
                        continue
                    # Prose ranks below explicit "Outputs:" keys and I/O sections.
                    if re.search(
                        r"\binputs?\b|\boutputs?\b|\bpredicts?\b|^\s*given\b|"
                        r"\bencodes?\b|\binto (?:\w+ )?embeddings?\b",
                        sentence,
                        re.I,
                    ):
                        add(_statement_roles(sentence), sentence, label, rank=1)

    descriptive = (
        "description",
        "short_description",
        "architecture",
        "how_to_use",
        "training_data",
    )
    for index, section in enumerate(card.sections):
        label = " > ".join(section.path)
        if index == 0 and not _section_matches(section, "io"):
            # The first section usually describes the model, even as an H2.
            scan(section.own_text, label, io_section=False, section_roles=())
        elif _section_matches(section, "io"):
            scan(
                section.own_text,
                label,
                io_section=True,
                section_roles=_io_role(section.key),
            )
        elif section.level == 1 or any(
            _section_matches(section, key) for key in descriptive
        ):
            strict = section.level > 1 and not any(
                _section_matches(section, key)
                for key in ("description", "short_description", "architecture")
            )
            scan(
                section.own_text,
                label,
                io_section=False,
                section_roles=(),
                strict=strict,
            )
        else:
            # Keyed lines such as "Input preprocessing: ..." matter wherever
            # they appear, including training-configuration sections.
            for line in logical_lines(section.own_text):
                key, body = _line_key(line)
                add(_io_role(key), body, f"{label} > {key}")
    first = card.sections[0].line if card.sections else None
    scan(
        "\n".join(card.clean_body.splitlines()[:first]),
        "preamble",
        io_section=False,
        section_roles=(),
    )
    return {
        role: [
            (text, source)
            for _, text, source in sorted(items, key=lambda item: item[0])
        ]
        for role, items in groups.items()
    }


def _io_tables(card: CardText) -> dict[str, list[ChannelSpec]]:
    """Read tables that tag each variable as input, output, or both."""
    found: dict[str, list[ChannelSpec]] = {"inputs": [], "outputs": []}
    for table in card.tables:
        headers = [h.lower() for h in table.headers]
        role_col = next(
            (
                i
                for i, h in enumerate(headers)
                if re.search(
                    r"input\s*/\s*output|\bi/o\b|\brole\b|\busage\b", h
                )
            ),
            None,
        )
        if role_col is None:
            continue
        name_col = next(
            (
                i
                for i, h in enumerate(headers)
                if re.search(r"field|variable|name|parameter|quantity", h)
            ),
            0,
        )
        level_col = next(
            (i for i, h in enumerate(headers) if re.search(r"level|layer", h)),
            None,
        )
        unit_col = next(
            (i for i, h in enumerate(headers) if _UNIT_HEADER.fullmatch(h)),
            None,
        )
        for row in table.rows:
            if max(role_col, name_col) >= len(row):
                continue
            roles = _IO_ROLE_VALUES.get(row[role_col].strip().lower(), ())
            level = (
                row[level_col]
                if level_col is not None and level_col < len(row)
                else None
            )
            units = (
                _clean_unit(row[unit_col])
                if unit_col is not None and unit_col < len(row)
                else None
            )
            for name in re.split(r",\s*(?![^()]*\))", row[name_col]):
                name = name.strip(" .")
                for role in roles if name else ():
                    found[role].append(
                        ChannelSpec(
                            name=name,
                            units=units,
                            description=_short(level, 160),
                        )
                    )
    return found


_UNIT_HEADER = re.compile(r"\s*\[?units?\]?(?:\s*\(.*\))?\s*")
_UNIT_NAME_HEADER = re.compile(
    r"\b(?:short name|name|variable|field|parameter|quantity|channel|symbol|"
    r"abbreviation)s?\b"
)


def _clean_unit(text: str | None) -> str | None:
    r"""Strip LaTeX and Markdown from a unit cell: '\\(J m^{-2}\\)' -> 'J m^-2'."""
    if not text:
        return None
    unit = strip_inline_markdown(text)
    unit = re.sub(r"\\+[()\[\]]|\$", "", unit)
    unit = re.sub(r"\\[,;:! ]", " ", unit)  # LaTeX spacing commands
    unit = re.sub(r"\\(?:mathrm|text|rm)\s*", "", unit)
    unit = unit.replace("{", "").replace("}", "").replace("\\", "")
    unit = re.sub(r"\s+", " ", unit).strip()
    if unit.lower() in {"", "-", "–", "—", "n/a", "na", "none"}:
        return None
    return unit


def _unit_glossary(card: CardText) -> dict[str, str]:
    """Map variable names to units from any table with a units column."""
    glossary: dict[str, str] = {}
    for table in card.tables:
        headers = [h.strip().lower() for h in table.headers]
        unit_col = next(
            (i for i, h in enumerate(headers) if _UNIT_HEADER.fullmatch(h)),
            None,
        )
        if unit_col is None:
            continue
        name_cols = [
            i
            for i, h in enumerate(headers)
            if i != unit_col and _UNIT_NAME_HEADER.search(h)
        ]
        for row in table.rows:
            unit = _clean_unit(row[unit_col]) if unit_col < len(row) else None
            for col in name_cols if unit else ():
                if col < len(row):
                    name = strip_inline_markdown(row[col]).strip(" .").lower()
                    if name:
                        glossary.setdefault(name, unit)
    return glossary


def _apply_unit_glossary(
    io: dict[str, list[IOSpec]], glossary: dict[str, str]
) -> None:
    """Fill missing units of variables whose names a units table defines."""
    for specs in io.values():
        for spec in specs:
            if not spec.units and spec.name.lower() in glossary:
                spec.units = glossary[spec.name.lower()]
            for channel in spec.channels:
                if not channel.units and channel.name.lower() in glossary:
                    channel.units = glossary[channel.name.lower()]


def _spec_tables(card: CardText) -> dict[str, list[IOSpec]]:
    """Read per-variable tables (name, shape, dtype, units) under I/O headings."""
    found: dict[str, list[IOSpec]] = {"inputs": [], "outputs": []}
    for section in card.sections:
        if not _section_matches(section, "io"):
            continue
        roles = _io_role(section.key)
        if len(roles) != 1:
            continue
        for table in parse_tables(section.own_text):
            headers = [h.strip().lower() for h in table.headers]
            name_col = next(
                (
                    i
                    for i, h in enumerate(headers)
                    if h in {"name", "variable", "field", "argument", "key"}
                ),
                None,
            )
            if name_col is None or not any(
                h in {"shape", "dtype", "type", "units", "unit", "dimensions"}
                for h in headers
            ):
                continue
            for row in table.rows:
                if name_col >= len(row) or not row[name_col].strip():
                    continue
                cells = dict(zip(headers, row))
                spec = IOSpec(name=row[name_col].strip())
                shape_text = cells.get("shape") or cells.get("dimensions") or ""
                spec.shape = P.parse_shape(shape_text) or (
                    [shape_text] if shape_text else []
                )
                dtype = P.DTYPE.search(
                    cells.get("dtype") or cells.get("type") or ""
                )
                spec.dtype = dtype.group(1).lower() if dtype else None
                spec.units = _clean_unit(
                    cells.get("units") or cells.get("unit")
                )
                description = next(
                    (
                        cells[h]
                        for h in ("meaning", "description", "details")
                        if cells.get(h)
                    ),
                    None,
                )
                spec.description = _short(description, 300)
                found[roles[0]].append(spec)
    return found


def _build_io(card: CardText, ctx: dict) -> dict[str, list[IOSpec]]:
    result: dict[str, list[IOSpec]] = {"inputs": [], "outputs": []}
    for role, statements in _collect_io_statements(card).items():
        named: dict[str, IOSpec] = {}
        generic: IOSpec | None = None
        for text, source in statements:
            info = _analyze_statement(text)
            names = info["identifiers"]
            channels: list[ChannelSpec] = []
            if len(names) >= 3:
                channels = [ChannelSpec(name=n) for n in names]
                names = []
            elif info.get("quantities"):
                channels = [ChannelSpec(name=q) for q in info["quantities"]]
            if names:
                shared = info
                if len(names) > 1:
                    # A unit in "`x`: positions in Å; `z`: ..." belongs to
                    # one of the variables, and the text does not say which.
                    shared = {k: v for k, v in info.items() if k != "units"}
                for name in names[:2]:
                    _merge_io(named.setdefault(name, IOSpec(name=name)), shared)
                    ctx["io_evidence"].append((role, name, source, text))
                continue
            if generic is None:
                generic = IOSpec(name=role[:-1])
            _merge_io(generic, info)
            for channel in channels:
                if channel.name not in {c.name for c in generic.channels}:
                    generic.channels.append(channel)
            if role == "outputs" and _SAME_AS_INPUT.search(info["text"]):
                ctx["outputs_same_as_inputs"] = True
            if role == "outputs" and _SAME_SHAPE.search(info["text"]):
                ctx["outputs_same_shape"] = True
            if info.get("count"):
                ctx["io_counts"][role] = info["count"]
            ctx["io_evidence"].append((role, generic.name, source, text))
        specs = list(named.values())
        if role == "inputs":
            # Semantics describe predictions; on inputs they are noise from
            # architecture prose such as "patch embeddings".
            for spec in specs + ([generic] if generic else []):
                spec.semantics = None
        if generic is not None:
            informative = (
                generic.channels
                or generic.shape
                or generic.representation
                or generic.semantics
                or generic.units
            )
            if len(specs) == 1 and not generic.channels:
                # One named input or output: the generic statements describe it.
                _merge_io(specs[0], generic.model_dump())
            elif specs and not informative:
                for spec in specs:
                    spec.normalization = (
                        spec.normalization or generic.normalization
                    )
            else:
                old = generic.name
                if (
                    generic.kind in {"molecule", "structure", "sequence"}
                    and generic.representation
                ):
                    generic.name = generic.representation
                elif generic.channels and generic.kind == "field":
                    generic.name = "fields"
                elif role == "outputs":
                    generic.name = (
                        _predicted_quantity(generic.description) or old
                    )
                count = ctx["io_counts"].get(role)
                if count and not generic.channels:
                    names, line = _counted_list(card.clean_body, count)
                    if names:
                        generic.channels = [ChannelSpec(name=n) for n in names]
                        ctx["io_evidence"].append((
                            role,
                            f"{generic.name}.channels",
                            "text",
                            line,
                        ))
                if generic.name != old:
                    _rename_evidence(ctx, role, old, generic.name)
                specs.append(generic)
        result[role] = specs
    for role, specs in _spec_tables(card).items():
        if specs:
            # A per-variable table is the most precise I/O statement a card
            # can make; it replaces generic prose specs for that role.
            result[role] = specs + [
                s
                for s in result[role]
                if s.name not in {"input", "output"}
                and s.name not in {x.name for x in specs}
            ]
    tables = _io_tables(card)
    for role in ("inputs", "outputs"):
        if tables[role]:
            target = next((s for s in result[role] if not s.channels), None)
            if target is None:
                target = IOSpec(name="fields", kind="field")
                result[role].append(target)
            target.channels = tables[role]
            target.kind = target.kind or "field"
    if ctx.get("outputs_same_as_inputs"):
        source = next((s for s in result["inputs"] if s.channels), None)
        target = next((s for s in result["outputs"] if not s.channels), None)
        if source and target:
            target.channels = list(source.channels)
    if ctx.get("outputs_same_shape"):
        source = next((s for s in result["inputs"] if s.shape), None)
        for target in result["outputs"]:
            if source and not target.shape:
                target.shape = list(source.shape)
    _apply_unit_glossary(result, _unit_glossary(card))
    return result


_ARCH_NORM = re.compile(
    r"layer ?norm|batch ?norm|group ?norm|instance ?norm|rms ?norm|"
    r"normali[sz]ation layers?|pre-?layer",
    re.I,
)


def _attach_normalization(io: dict[str, list[IOSpec]], card: CardText) -> None:
    """Attach card-wide normalization statements to inputs and outputs.

    Preprocessing is often described in the model description or the
    architecture notes rather than next to the I/O list.
    """
    for sentence in _sentences(prose(card.clean_body)):
        if not P.NORMALIZATION.search(sentence) or _ARCH_NORM.search(sentence):
            continue
        lowered = sentence.lower()
        note = _short(sentence, 220)
        if re.search(r"\binputs?\b|\bfields?\b|\bstates?\b|\bdata\b", lowered):
            for spec in io["inputs"]:
                if not spec.normalization:
                    spec.normalization = note
                    break
        if re.search(
            r"\boutputs?\b|de-?normali[sz]|\bpredictions? (?:are|is)\b", lowered
        ):
            for spec in io["outputs"]:
                if not spec.normalization:
                    spec.normalization = note
                    break


def _merge_config_io(io: dict[str, list[IOSpec]], ctx: dict) -> None:
    config_io = ctx["config_io"]
    for role in ("inputs", "outputs"):
        channels = config_io[role]
        if not channels:
            continue
        specs = io[role]
        if len(channels) == 1:
            if channels[0].name not in {spec.name for spec in specs}:
                specs.insert(0, IOSpec(name=channels[0].name))
            continue
        target = next((s for s in specs if not s.channels), None)
        if target is None:
            target = IOSpec(name=role[:-1])
            specs.append(target)
        target.channels = channels
        if role == "outputs" and config_io.get("output_semantics"):
            target.semantics = target.semantics or config_io["output_semantics"]
    if config_io.get("regression_head") and not io["outputs"]:
        io["outputs"].append(
            IOSpec(
                name="prediction",
                semantics="regression",
                description=f"Regression head ({config_io['regression_head']} in config.json).",
            )
        )
    representation = config_io.get("representation")
    if representation:
        for spec in io["inputs"]:
            if spec.kind == "molecule" and not spec.representation:
                spec.representation = representation


def _overlay_specs(
    primary: list[IOSpec], secondary: list[IOSpec]
) -> list[IOSpec]:
    """MCP specs define the runtime interface; card text adds detail."""
    by_name = {spec.name.lower(): spec for spec in secondary}
    for spec in primary:
        other = by_name.get(spec.name.lower())
        if other is None and len(secondary) == 1 and len(primary) == 1:
            other = secondary[0]
        if other is None:
            continue
        for key in ("units", "normalization", "semantics", "dtype"):
            if not getattr(spec, key) and getattr(other, key):
                setattr(spec, key, getattr(other, key))
        if not spec.channels and other.channels:
            spec.channels = other.channels
    # A WellFormat state carries named fields. The card text names them
    # ("Input field: `av_density`"); the tool schema does not.
    named = {spec.name.lower() for spec in primary}
    fields = [
        spec
        for spec in secondary
        if spec.name.lower()
        not in named | {"input", "inputs", "output", "outputs", "fields"}
        and spec.kind in (None, "field", "tensor")
    ]
    for spec in primary:
        if spec.representation == "WellFormat" and not spec.channels:
            spec.channels = [
                ChannelSpec(
                    name=f.name, units=f.units, description=f.description
                )
                for f in fields
            ]
    return primary


# Other text facts -------------------------------------------------------------


def _handle_key_values(
    pairs: list[tuple[str, str]], b: _Builder, ctx: dict
) -> None:
    for key, value in pairs:
        quote = f"{key}: {value}"
        lowered = key.lower()
        if _key_match(key, _KEY_ALIASES["parameters"]):
            match = re.search(
                rf"(?P<num>{P.NUMBER})\s*(?P<unit>[KMBT]|million|billion)?\b",
                value,
            )
            if match:
                b.set(
                    "architecture.parameter_count",
                    P.parse_count(match.group("num"), match.group("unit")),
                    "key_value",
                    kind="key_value",
                    quote=quote,
                    compare=_same_count,
                )
        elif _key_match(key, _KEY_ALIASES["license"]):
            b.set(
                "license.spdx",
                P.normalize_license(value.split("(")[0].strip()),
                "key_value",
                kind="key_value",
                quote=quote,
            )
        elif _key_match(key, _KEY_ALIASES["base_model"]):
            if not re.match(r"not applicable|n/?a\b|none", value, re.I):
                b.set(
                    "architecture.base_model",
                    value.split(" ")[0].strip("`[]"),
                    "key_value",
                    kind="key_value",
                    quote=quote,
                )
        elif _key_match(key, _KEY_ALIASES["model_type"]):
            ctx["model_type_text"] = value
        elif _key_match(key, _KEY_ALIASES["version"]):
            version = re.search(r"v?\d+(?:\.\d+)+", value)
            if version:
                b.set(
                    "version",
                    version.group(0),
                    "key_value",
                    kind="key_value",
                    quote=quote,
                )
        elif _key_match(key, _KEY_ALIASES["developers"]):
            ctx["kv_developers"].append(value)
        elif _key_match(key, _KEY_ALIASES["dtype"]):
            dtype = P.DTYPE.search(value)
            if dtype:
                ctx["weights_dtype"] = dtype.group(1).lower()
        elif _key_match(key, _KEY_ALIASES["limitations"]):
            ctx["kv_limitations"].append(_short(value, 300))
        if "resolution" in lowered or "grid" in lowered:
            degrees = P.DEGREES.search(value)
            if degrees:
                ctx["resolution"].append((
                    f"{degrees.group(1)} deg",
                    PRIORITY["key_value"],
                ))
            dims = P.parse_shape(value)
            if dims:
                ctx["resolution"].append((
                    " x ".join(dims),
                    PRIORITY["key_value"],
                ))
        if lowered in {"direction", "lead time", "forecast horizon", "horizon"}:
            b.set(
                "temporal.horizon",
                _short(value, 160),
                "key_value",
                kind="key_value",
                quote=quote,
            )


def _acronym(text: str) -> str:
    return "".join(word[0] for word in re.findall(r"[A-Za-z]+", text)).lower()


def _related_title(title: str, repo_name: str | None) -> bool:
    if not repo_name:
        return True
    repo_tokens = {t.lower() for t in re.split(r"[-_. ]+", repo_name) if t}
    title_tokens = {t.lower() for t in re.findall(r"[A-Za-z0-9]+", title)}
    if repo_tokens & (title_tokens - {"model", "the", "of", "for", "a"}):
        return True
    compact = _compact(repo_name)
    for part in re.split(r"\s*[:–—]\s+|\s+-\s+", title):
        # "FourCastNet 3" names the repository "nvidia/fourcastnet3".
        if len(_compact(part)) >= 4 and _compact(part) in compact:
            return True
    return _acronym(title) in repo_tokens


def _compact(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _title_name(title: str, hint: str | None) -> str:
    """Pick the model name from 'Name: subtitle' or 'Series: Name' titles."""
    parts = re.split(r"\s*[:–—]\s+|\s+-\s+", title, maxsplit=1)
    if len(parts) == 1:
        return title
    before, after = parts
    compact = _compact(hint)
    if (
        compact
        and _compact(after)
        and _compact(after) in compact
        and _compact(before) not in compact
    ):
        return after
    if re.match(r"(?:a|an|the)\b", after, re.I) or len(after.split()) > 4:
        return before
    if re.search(r"\b(?:checkpoints?|models?|release|weights)\b", before, re.I):
        return after
    return before


def _config_parameter_estimate(config: dict[str, Any] | None) -> int | None:
    """Rough transformer size from config.json: 12 L h^2 plus embeddings."""
    if not config:
        return None
    hidden = (
        config.get("hidden_size")
        or config.get("n_embd")
        or config.get("d_model")
    )
    layers = (
        config.get("num_hidden_layers")
        or config.get("num_layers")
        or config.get("n_layer")
    )
    if not (isinstance(hidden, int) and isinstance(layers, int)):
        return None
    vocab = (
        config.get("vocab_size")
        if isinstance(config.get("vocab_size"), int)
        else 0
    )
    return 12 * layers * hidden * hidden + vocab * hidden


def _handle_names(
    doc: CardDocument, card: CardText, b: _Builder, ctx: dict
) -> None:
    repo_name = (doc.repo_id or "").split("/")[-1] or None
    titles = [
        s.title
        for s in card.sections
        if s.level == 1 and s.key not in GENERIC_TITLES
    ]
    # Repository names are deliberate; local file names are not, so only a
    # repository name can veto an unrelated title.
    title = next((t for t in titles if _related_title(t, repo_name)), None)
    if title:
        short = _title_name(title, repo_name or doc.name_hint)
        short = re.sub(
            r"^(?:model card(?: for)?:?)\s*", "", short, flags=re.I
        ).strip()
        ctx["title"], ctx["title_short"] = title, short or None
    if repo_name:
        b.set("name", repo_name, "repo_id", kind="hub_api")
    elif doc.mcp_tool and doc.mcp_tool.get("name"):
        b.set("name", doc.mcp_tool["name"], "mcp_schema", kind="mcp_schema")
    elif ctx.get("declared_name"):
        b.set("name", ctx["declared_name"], "front_matter", kind="front_matter")
    elif ctx.get("title_short"):
        b.set("name", ctx["title_short"], "title", kind="section", quote=title)
    elif doc.config and isinstance(doc.config.get("model_name"), str):
        b.set("name", doc.config["model_name"], "config.json", kind="config")
    version_source = repo_name or ctx.get("title_short") or ""
    version = P.VERSION_IN_NAME.search(version_source)
    if version:
        b.set(
            "version",
            version.group(1),
            "name",
            kind="name",
            quote=version_source,
        )
    estimate = _config_parameter_estimate(doc.config)
    for candidate in (repo_name, ctx.get("title_short"), doc.name_hint):
        hint = P.PARAM_NAME.search(candidate or "")
        if not hint:
            continue
        count = P.parse_count(hint.group("num"), hint.group("unit"))
        if count and estimate and not estimate / 5 <= count <= estimate * 5:
            # "ChemBERTa-77M" names its 77M pretraining molecules, not weights.
            b.issues.append(
                CardIssue(
                    severity="info",
                    code="name_parameter_mismatch",
                    field="architecture.parameter_count",
                    message=(
                        f"The name '{candidate}' suggests {P.format_count(count)} "
                        f"parameters, but config.json implies about "
                        f"{P.format_count(estimate)}; the name was not used."
                    ),
                )
            )
            break
        b.set(
            "architecture.parameter_count",
            count,
            "name",
            kind="name",
            quote=candidate,
            compare=_same_count,
        )
        break


def _handle_summary(card: CardText) -> str | None:
    for section in _sections_for(card, "short_description"):
        paragraph = first_paragraph(section.own_text, min_words=6)
        if paragraph:
            return _short(paragraph, 600)
    lead = first_paragraph(_preamble(card), min_words=6)
    if lead:
        return _short(lead, 600)
    for section in _sections_for(card, "description"):
        paragraph = first_paragraph(section.own_text, min_words=6)
        if paragraph:
            return _short(paragraph, 600)
    return _short(first_paragraph(card.clean_body, min_words=6), 600)


def _counts_in(text: str) -> list[int]:
    counts = []
    for pattern in P.PARAM_TEXT:
        for match in pattern.finditer(text):
            count = P.parse_count(match.group("num"), match.group("unit"))
            if count:
                counts.append(count)
    return counts


def _handle_parameter_mentions(
    card: CardText, b: _Builder, doc: CardDocument
) -> None:
    repo_name = ((doc.repo_id or "").split("/")[-1] or "").lower()
    for table in card.tables:
        for row in table.rows:
            counts = _counts_in(" | ".join(row))
            first = (row[0] if row else "").strip().lower()
            # Family tables list sibling checkpoints; only this model's row counts.
            if (
                counts
                and repo_name
                and (first == repo_name or first.startswith(repo_name + " "))
            ):
                b.set(
                    "architecture.parameter_count",
                    counts[0],
                    f"table {table.section}".strip(),
                    kind="section",
                    quote=" | ".join(row),
                    compare=_same_count,
                )
    for line in prose(card.clean_body).splitlines():
        counts = _counts_in(line)
        # Prose that compares several model sizes is ambiguous.
        if (
            counts
            and len(set(counts)) == 1
            and not re.search(r"hyper|non-trainable", line, re.I)
        ):
            b.set(
                "architecture.parameter_count",
                counts[0],
                "text",
                kind="text",
                quote=line,
                compare=_same_count,
            )


def _handle_spatial_temporal(card: CardText, b: _Builder, ctx: dict) -> None:
    spatial_sources = [_preamble(card)] + [
        s.own_text
        for s in card.sections
        if s.level == 1
        or any(
            _section_matches(s, k)
            for k in ("io", "description", "architecture")
        )
    ]
    for sentence in _sentences(prose("\n".join(spatial_sources))):
        lowered = sentence.lower()
        if re.search(
            r"\bgrid|resolution|pixels?|\bmesh|domain|latitude|size of", lowered
        ):
            for match in P.GRID.finditer(sentence):
                ctx["resolution"].append((
                    " x ".join(g for g in match.groups() if g),
                    PRIORITY["text"],
                ))
            for match in P.DEGREES.finditer(sentence):
                ctx["resolution"].append((
                    f"{match.group(1)} deg",
                    PRIORITY["text"],
                ))
            for match in P.KM_RESOLUTION.finditer(sentence):
                ctx["resolution"].append((
                    f"{match.group(1)} km",
                    PRIORITY["text"],
                ))
            if re.search(
                r"equi-?rectangular|lat(?:itude)?[-/ ]lon(?:gitude)?", lowered
            ):
                b.set(
                    "spatial.grid",
                    "latitude-longitude",
                    "text",
                    kind="text",
                    quote=sentence,
                )
        matches = list(P.SPATIAL_DIMS.finditer(sentence))
        enumeration = _ENUMERATION.search(sentence)
        if matches and enumeration and enumeration.start() < matches[0].start():
            # "modalities including SMILES, 3D atom positions, ..." lists
            # options; it does not state the model's own dimensionality.
            matches = []
        dims = {
            int(m.group("n"))
            if m.group("n")
            else {"one": 1, "two": 2, "three": 3}[m.group("word").lower()]
            for m in matches
        }
        if len(dims) == 1:
            b.set(
                "spatial.n_dims",
                dims.pop(),
                "text",
                kind="text",
                quote=sentence,
            )
    for section in [None, *card.sections]:
        text = _preamble(card) if section is None else section.own_text
        label = "preamble" if section is None else " > ".join(section.path)
        for sentence in _sentences(prose(text)):
            if re.search(r"\bresolution|\bgrid\b", sentence, re.I):
                # "interpolated from native O1280 down to N320": keep the target.
                target = re.split(r"\b(?:down |up )?to\b", sentence)
                if len(target) > 1 and re.search(
                    r"\bfrom\b|\bnative\b", target[0], re.I
                ):
                    sentence = target[-1]
                for match in P.DEGREES.finditer(sentence):
                    ctx["resolution"].append((
                        f"{match.group(1)} deg",
                        PRIORITY["text"] - 1,
                    ))
                for match in P.GAUSSIAN_GRID.finditer(sentence):
                    ctx["resolution"].append((
                        match.group(1),
                        PRIORITY["text"] - 1,
                    ))
                for match in P.GRID.finditer(sentence):
                    ctx["resolution"].append((
                        " x ".join(g for g in match.groups() if g),
                        PRIORITY["text"] - 2,
                    ))
                for match in P.LENGTH_RESOLUTION.finditer(sentence):
                    value = match.group("v1") or match.group("v2")
                    unit = match.group("u1") or match.group("u2")
                    ctx["resolution"].append((
                        f"{value} {unit.lower()}",
                        PRIORITY["text"] - 2,
                    ))
            step = P.find_time_step(sentence)
            if step:
                b.set(
                    "temporal.time_step",
                    step[0],
                    label,
                    kind="section",
                    quote=sentence,
                )
            # "trained to produce 6-hour forecasts" names the step, not a horizon.
            horizon = P.find_horizon(
                sentence, exclude=step[0] if step else None
            )
            data_section = section is not None and (
                _section_matches(section, "training_data")
                or _DATA_SECTION.match(section.key)
            )
            if horizon and data_section and not _MODEL_CUE.search(sentence):
                # "GDPS, which provides 10-day forecasts" describes a data
                # source, not this model's horizon.
                horizon = None
            if horizon:
                b.set(
                    "temporal.horizon",
                    horizon[0],
                    label,
                    kind="text",
                    quote=sentence,
                )
            context = P.find_context_steps(sentence)
            if context:
                b.set(
                    "temporal.context_steps",
                    context[0],
                    label,
                    kind="section",
                    quote=sentence,
                )
            rate = P.SAMPLING.search(sentence)
            if rate and re.search(r"sampl|rate", sentence, re.I):
                b.set(
                    "temporal.sampling_rate",
                    f"{rate.group(1)} {rate.group(2)}",
                    label,
                    kind="section",
                    quote=sentence,
                )


def _handle_interfaces(
    doc: CardDocument, card: CardText, b: _Builder, ctx: dict
) -> None:
    code = "\n".join(block.code for block in card.code_blocks)
    inline = "\n".join(
        f"`{span}`" for span in P.BACKTICK.findall(card.clean_body)
    )
    install = list(
        dict.fromkeys(
            m.strip() for m in P.INSTALL.findall(code + "\n" + inline)
        )
    )[:6]
    imports: dict[str, str] = {}
    for module, names in P.IMPORT_FROM.findall(code):
        for name in re.split(r"\s*,\s*", names.strip("() ")):
            original = name.split(" as ")[0].strip()
            alias = name.split(" as ")[-1].strip()
            if alias:
                imports[alias] = f"{module}.{original}"
    loaders = [imports.get(call, call) for call in P.LOAD_CALL.findall(code)]
    tags = {t.lower() for t in ctx["author_tags"] + ctx["hub_tags"]}
    trust = (
        bool(re.search(r"trust_remote_code\s*=\s*True", code))
        or ctx.get("remote_code", False)
        or bool({"custom_code", "custom-code"} & tags)
    )
    snippet = next(
        (
            block.code
            for block in card.code_blocks
            if block.language in {"python", "py", ""}
            and re.search(r"from_pretrained|load|predict|import ", block.code)
        ),
        None,
    )
    if doc.repo_id or loaders:
        ctx["interfaces"].append(
            Interface(
                kind="huggingface" if doc.repo_id else "python",
                name=b.get("name"),
                repo_id=doc.repo_id,
                loader=loaders[0] if loaders else None,
                trust_remote_code=True if trust else None,
                install=install,
                snippet=_truncate_code(snippet),
            )
        )
    for command in dict.fromkeys(P.CLI_CALL.findall(code + "\n" + inline)):
        if not P.NOT_CLI.match(command):
            ctx["interfaces"].append(
                Interface(kind="cli", name=command, install=install)
            )
    onnx = [f for f in ctx["files"] if f.endswith(".onnx")]
    if onnx:
        ctx["interfaces"].append(
            Interface(kind="onnx", name=onnx[0], repo_id=doc.repo_id)
        )
    if trust:
        b.issues.append(
            CardIssue(
                severity="info",
                code="remote_code",
                message=(
                    "Loading requires trust_remote_code=True, which runs code "
                    "from the model repository; review it before running."
                ),
            )
        )


def _truncate_code(code: str | None, limit: int = 1500) -> str | None:
    if not code:
        return None
    return code if len(code) <= limit else code[:limit] + "\n# ... truncated"


def _handle_metrics(card: CardText, doc: CardDocument, ctx: dict) -> None:
    terms = {
        term.lower()
        for term in (
            (doc.repo_id or "").split("/")[-1],
            *(d.name.split("/")[-1] for d in ctx["datasets"]),
        )
        if term
    }
    for table in card.tables:
        headers = [h.strip() for h in table.headers]
        metric_cols = [
            i for i, h in enumerate(headers) if P.METRIC_NAMES.match(h)
        ]
        if metric_cols:
            rows = table.rows
            matching = [
                row
                for row in rows
                if row
                and any(
                    t and (t in row[0].lower() or row[0].lower() in t)
                    for t in terms
                )
            ]
            if len(rows) > 3 and not matching:
                ctx["unattributed_tables"] += 1
                continue
            if matching and len(rows) > 1:
                ctx["shared_card"] = True
            for row in matching or rows:
                for col in metric_cols:
                    if col < len(row) and re.search(r"\d", row[col]):
                        ctx["metrics"].append(
                            MetricResult(
                                metric=headers[col],
                                value=row[col],
                                dataset=row[0] if col != 0 else None,
                                notes=table.section or None,
                            )
                        )
        else:
            for row in table.rows:
                if (
                    len(row) >= 2
                    and P.METRIC_NAMES.match(row[0])
                    and re.search(r"\d", row[1])
                ):
                    ctx["metrics"].append(
                        MetricResult(
                            metric=row[0],
                            value=row[1],
                            notes=table.section or None,
                        )
                    )


def _handle_training_data(card: CardText, ctx: dict) -> None:
    for section in _sections_for(card, "training_data"):
        for name in dict.fromkeys(
            P.KNOWN_DATASETS.findall(prose(section.text))
        ):
            ctx["datasets"].append(DatasetRef(name=name))
        for label, url in parse_links(section.text):
            if re.search(
                r"dataset|data\b|\.gov/|zenodo|figshare|oceans11|zarr",
                f"{url} {label or ''}",
                re.I,
            ):
                ctx["datasets"].append(
                    DatasetRef(
                        name=label or url.rstrip("/").split("/")[-1], url=url
                    )
                )
        lead = first_paragraph(section.own_text, min_words=6)
        if lead and ctx["datasets"] and not ctx["datasets"][-1].description:
            ctx["datasets"][-1].description = _short(lead, 300)
    for name in dict.fromkeys(P.KNOWN_DATASETS.findall(prose(_preamble(card)))):
        ctx["datasets"].append(DatasetRef(name=name))


_COUNT_WORD = r"(?P<n>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?P<scale>k|thousand|million|M)?"
_EXAMPLES = re.compile(
    r"\b(?:data ?set of|set of|consist(?:s|ing)? of|contain(?:s|ing)?|comprising|with)\s+"
    rf"(?:approximately |about |around |roughly |~|over |more than )?{_COUNT_WORD}\s+"
    r"(?:[\w₀-₉-]+\s+){0,3}?(?:entries|molecules|measurements|samples|examples|"
    r"structures|compounds|images|simulations|trajectories|snapshots|sequences|"
    r"proteins|crystals|materials|records)\b",
    re.I,
)
# "up to 9 heavy atoms", "fewer than 20 heavy atoms", "at most 1,024 residues".
_RANGE = re.compile(
    r"\b(?P<bound>up to|at most|no more than|fewer than|less than|maximum of|max(?:imum)?)\s+"
    r"(?P<n>\d{1,3}(?:,\d{3})+|\d+)\s+(?P<quantity>heavy atoms|non-hydrogen atoms|atoms|"
    r"residues|amino acids)\b",
    re.I,
)
_QUANTITY_NAMES = {
    "non-hydrogen atoms": "heavy atoms",
    "amino acids": "residues",
}


def _scaled(number: str, scale: str | None) -> int:
    value = float(number.replace(",", ""))
    factor = {"k": 1e3, "thousand": 1e3, "million": 1e6, "m": 1e6}.get(
        (scale or "").lower(), 1
    )
    return int(round(value * factor))


def _handle_training_scope(card: CardText, b: _Builder) -> None:
    """Training set size and range, as the training sections state them.

    A fine-tuned model states both a pretraining and a fine-tuning set; the
    fine-tuning statement wins, since it is the data the predictor learned.
    """
    sentences = [
        (sentence, " > ".join(section.path))
        for section in _sections_for(card, "training_data")
        for sentence in _sentences(prose(section.text))
    ]
    sentences.sort(key=lambda item: not re.search(r"fine-?tun", item[0], re.I))
    for sentence, source in sentences:
        match = _EXAMPLES.search(sentence)
        if match and b.get("training_scope.examples") is None:
            b.set(
                "training_scope.examples",
                _scaled(match.group("n"), match.group("scale")),
                source,
                kind="text",
                quote=sentence,
            )
        for found in _RANGE.finditer(sentence):
            quantity = found.group("quantity").lower()
            quantity = _QUANTITY_NAMES.get(quantity, quantity)
            key = f"training_scope.{quantity}"
            if b.get(key) is None:
                b.set(
                    key,
                    DomainRange(
                        quantity=quantity,
                        maximum=float(found.group("n").replace(",", "")),
                        inclusive=found.group("bound").lower()
                        not in {"fewer than", "less than"},
                    ),
                    source,
                    kind="text",
                    quote=sentence,
                )


def _dedupe_datasets(datasets: list[DatasetRef]) -> list[DatasetRef]:
    seen: dict[str, DatasetRef] = {}
    for dataset in datasets:
        key = dataset.name.lower().split("/")[-1]
        if key not in seen:
            seen[key] = dataset
        else:
            seen[key].url = seen[key].url or dataset.url
            seen[key].description = seen[key].description or dataset.description
    return list(seen.values())[:20]


def _handle_links(
    card: CardText, ctx: dict
) -> tuple[list[Link], list[Citation]]:
    links: list[Link] = []
    for label, url in parse_links(card.clean_body):
        lowered = url.lower()
        if "img.shields.io" in lowered or lowered.endswith((
            ".png",
            ".jpg",
            ".gif",
            ".svg",
        )):
            continue
        if re.search(
            r"arxiv\.org|doi\.org|nature\.com|openreview|science\.org|journals|ieee|mlr\.press",
            lowered,
        ):
            kind = "paper"
        elif re.search(r"github\.com|gitlab", lowered):
            kind = "code"
        elif "huggingface.co/datasets" in lowered:
            kind = "dataset"
        elif "huggingface.co/spaces" in lowered or "colab" in lowered:
            kind = "demo"
        elif "huggingface.co/" in lowered:
            kind = "model"
        elif re.search(r"readthedocs|\.github\.io|/docs?\b", lowered):
            kind = "docs"
        else:
            kind = "other"
        links.append(Link(kind=kind, url=url, label=label))
    citations: list[Citation] = []
    for block in card.code_blocks:
        if "@" not in block.code:
            continue
        for entry in parse_bibtex(block.code):
            doi = entry.get("doi")
            if doi:
                doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
            eprint = entry.get("eprint", "")
            citation = Citation(
                title=entry.get("title"),
                authors=_short(entry.get("author"), 200),
                year=entry.get("year") or (entry.get("date") or "")[:4] or None,
                doi=doi,
                arxiv=eprint if re.match(r"\d{4}\.\d{4,5}", eprint) else None,
                url=entry.get("url"),
            )
            if citation.title or citation.doi or citation.arxiv or citation.url:
                citations.append(citation)
    known = {c.arxiv for c in citations if c.arxiv} | {
        (c.doi or "").lower() for c in citations if c.doi
    }
    for arxiv in dict.fromkeys(ctx["arxiv"] + find_arxiv_ids(card.clean_body)):
        if arxiv not in known and f"10.48550/arxiv.{arxiv}" not in known:
            citations.append(
                Citation(arxiv=arxiv, url=f"https://arxiv.org/abs/{arxiv}")
            )
            known.add(arxiv)
    for doi in dict.fromkeys(ctx["dois"] + find_dois(card.clean_body)):
        lowered = doi.lower()
        if lowered in known or any(k and lowered.endswith(k) for k in known):
            continue
        citations.append(Citation(doi=doi, url=f"https://doi.org/{doi}"))
        known.add(lowered)
    return links[:30], citations[:10]


def _concept_texts(
    card: CardText, summary: str | None, ctx: dict
) -> list[tuple[str, float, str]]:
    texts: list[tuple[str, float, str]] = []
    for weight in (3.0, 1.0):
        tags = " ; ".join(t for t, w in ctx["plain_tags"] if w == weight)
        if tags:
            # Tags are keyword lists: "weather-forecasting" means both words.
            texts.append((re.sub(r"[_-]", " ", tags), weight, "tags"))
    if ctx.get("title"):
        texts.append((ctx["title"], 2.0, "title"))
    if summary:
        texts.append((summary, 1.5, "summary"))
    if ctx["datasets"]:
        names = " ; ".join(
            P.split_camel(d.name.split("/")[-1]) for d in ctx["datasets"]
        )
        texts.append((names, 1.5, "datasets"))
    if ctx.get("config_text"):
        # Class and label names are chosen deliberately by model authors.
        texts.append((ctx["config_text"], 2.0, "config.json"))
    if ctx.get("mcp_description"):
        texts.append((ctx["mcp_description"], 2.0, "mcp_schema"))
    texts.append((prose(_preamble(card)), 1.5, "preamble"))
    for key, weight in (
        ("short_description", 1.5),
        ("description", 1.0),
        ("io", 1.0),
        ("intended_use", 1.0),
    ):
        for section in _sections_for(card, key):
            texts.append((
                prose(section.own_text),
                weight,
                " > ".join(section.path),
            ))
    # Out-of-scope, limitation, and risk sections say what a model does not
    # do, so they never vote for its domains or tasks.
    excluded = [
        s.text
        for key in ("out_of_scope", "limitations", "risks")
        for s in _sections_for(card, key)
    ]
    body = card.clean_body
    for text in excluded:
        body = body.replace(text, " ")
    texts.append((prose(body), 0.5, "body"))
    return texts


def _vote_concepts(
    vocab: Vocabulary,
    texts: list[tuple[str, float, str]],
    extra_votes: list[tuple[str, float, str, str]],
    *,
    threshold: float,
    limit: int,
) -> tuple[list[str], dict[str, Evidence]]:
    scores: dict[str, float] = {}
    evidence: dict[str, Evidence] = {}
    for concept, weight, quote, source in extra_votes:
        scores[concept] = scores.get(concept, 0.0) + weight
        evidence.setdefault(
            concept, Evidence(source=source, method="structured", quote=quote)
        )
    for text, weight, source in texts:
        counts: dict[str, int] = {}
        for hit in vocab.match(text):
            if _NEGATION.search(text[max(0, hit.start - 40) : hit.start]):
                # "cannot be steered by properties like band gap"
                continue
            count = counts.get(hit.id, 0)
            # Repeated phrases add diminishing weight within one source.
            if count < 4:
                scores[hit.id] = scores.get(hit.id, 0.0) + (
                    weight if count == 0 else weight * 0.25
                )
            counts[hit.id] = count + 1
            lo, hi = max(0, hit.start - 70), hit.start + len(hit.phrase) + 70
            evidence.setdefault(
                hit.id,
                Evidence(
                    source=source,
                    method="pattern",
                    quote=_short(text[lo:hi]),
                    confidence=round(min(1.0, weight / 3), 2),
                ),
            )
    ranked = sorted(
        (c for c, s in scores.items() if s >= threshold),
        key=lambda c: -scores[c],
    )
    return ranked[:limit], {c: evidence[c] for c in ranked[:limit]}


def _license(card: CardText, b: _Builder) -> LicenseInfo:
    spdx = b.get("license.spdx")
    info = LicenseInfo(
        spdx=spdx,
        name=b.get("license.name"),
        link=b.get("license.link"),
        commercial_use=P.commercial_use_for(spdx),
    )
    restrictions: list[str] = []
    for sentence in _sentences(prose(card.clean_body)):
        if P.RESTRICTION.search(sentence) and not P.BOTH_USES.search(sentence):
            item = _short(sentence, 200)
            if item not in restrictions:
                restrictions.append(item)
    info.restrictions = restrictions[:6]
    noncommercial = [r for r in restrictions if P.NONCOMMERCIAL.search(r)]
    if noncommercial:
        if info.commercial_use:
            # Quote the restriction that names commercial use when there is one.
            quote = next(
                (r for r in noncommercial if re.search(r"commercial", r, re.I)),
                noncommercial[0],
            )
            b.issues.append(
                CardIssue(
                    severity="warning",
                    code="license_conflict",
                    field="license",
                    message=(
                        f"Front matter declares '{spdx}', but the card text "
                        f"restricts use: {quote!r}. Treating "
                        "commercial use as not permitted until verified."
                    ),
                )
            )
        info.commercial_use = False
    if spdx is None and not info.name:
        b.issues.append(
            CardIssue(
                severity="warning",
                code="license_missing",
                field="license",
                message="The card does not state a license.",
            )
        )
    elif spdx == "other" and not (info.name or info.link):
        b.issues.append(
            CardIssue(
                severity="warning",
                code="license_unspecified_other",
                field="license",
                message="License is 'other' without a name or link.",
            )
        )
    return info


def _compute(card: CardText, b: _Builder, ctx: dict) -> ComputeInfo:
    compute = ComputeInfo()
    training, inference = [], []
    for sentence in _sentences(prose(card.clean_body)):
        if len(sentence) > 320:
            continue
        lowered = sentence.lower()
        if not P.COMPARATIVE.search(sentence):
            if compute.cpu_supported is None and P.CPU_OK.search(sentence):
                compute.cpu_supported = True
                b.set(
                    "compute.cpu_supported",
                    True,
                    "text",
                    kind="text",
                    quote=sentence,
                )
            elif (
                compute.cpu_supported is None
                and P.CPU_NO.search(sentence)
                and "train" not in lowered
            ):
                compute.cpu_supported = False
                b.set(
                    "compute.cpu_supported",
                    False,
                    "text",
                    kind="text",
                    quote=sentence,
                )
        if P.MULTI_GPU.search(sentence) and "train" not in lowered:
            compute.multi_gpu_required = True
        gpu_model = P.GPU.search(sentence)
        memory = P.GIGABYTES.search(sentence)
        if not (gpu_model or re.search(r"\bgpus?\b", lowered)):
            continue
        if re.search(r"\btrain|pre-?train|fine-?tun", lowered):
            training.append(sentence)
        elif (
            P.HARDWARE_REQUIREMENT.search(sentence)
            and not re.search(r"colab|try it", lowered)
            and (
                gpu_model
                or memory
                or re.search(
                    r"single gpu|gpu (?:is )?(?:recommended|required)|requires? (?:a )?gpu",
                    lowered,
                )
            )
        ):
            inference.append(sentence)
            if (
                memory
                and compute.min_gpu_memory_gb is None
                and re.search(r"memory|vram|gpu", lowered)
            ):
                compute.min_gpu_memory_gb = float(memory.group(1))
                b.set(
                    "compute.min_gpu_memory_gb",
                    compute.min_gpu_memory_gb,
                    "text",
                    kind="text",
                    quote=sentence,
                )
    compute.training_hardware = _short(training[0], 240) if training else None
    compute.inference_hardware = (
        _short(inference[0], 240) if inference else None
    )
    count = b.get("architecture.parameter_count")
    if count:
        dtype = ctx.get("weights_dtype") or "float32"
        bytes_per = (
            2 if re.search(r"16|half|bf", dtype) else 8 if "64" in dtype else 4
        )
        compute.estimated_weights_gb = round(count * bytes_per / 1e9, 3)
    return compute


def _architecture_family(card: CardText, b: _Builder, ctx: dict) -> str | None:
    text = " ".join(
        filter(
            None,
            (
                b.get("architecture.name"),
                ctx.get("model_type"),
                ctx.get("model_type_text"),
                ctx.get("title"),
                " ".join(
                    prose(s.own_text)
                    for s in _sections_for(card, "architecture")
                ),
                " ".join(
                    prose(s.own_text)
                    for s in _sections_for(card, "description")
                ),
                prose(_preamble(card)),
            ),
        )
    )
    families = (
        (
            "neural operator",
            r"neural operator|\bFNO\b|fourier neural|\bSFNO\b|DeepONet",
        ),
        ("diffusion model", r"\bdiffusion\b|\bDDPM\b|\bDDIM\b|score-based"),
        (
            "graph neural network",
            r"graph neural|\bGNN\b|message passing|equivariant graph|\bMACE\b|NequIP|graph-ml",
        ),
        ("transformer", r"transformer|\bViT\b|\bswin|RoBERTa|\bBERT\b"),
        ("U-Net", r"\bU-?Net\b"),
        ("convolutional network", r"\bCNN\b|convolutional|ResNet|ConvNeXt"),
        ("masked autoencoder", r"masked autoencoder|\bMAE\b"),
        ("MLP", r"\bMLP\b|multilayer perceptron|multi-layer perceptron"),
        ("recurrent network", r"\bLSTM\b|\bGRU\b|recurrent"),
    )
    for name, pattern in families:
        if re.search(pattern, text, re.I):
            return name
    return None


def _frameworks(card: CardText, ctx: dict, b: _Builder) -> list[str]:
    text = " ".join(block.code for block in card.code_blocks)
    text += " " + " ".join(
        ctx["author_tags"] + ctx["hub_tags"] + ctx["declared_frameworks"]
    )
    text += " " + (b.get("library_name") or "") + " " + " ".join(ctx["files"])
    return [
        name
        for name, pattern in (
            ("pytorch", r"\btorch\b|pytorch|\.pt\b|\.ckpt\b|safetensors"),
            ("jax", r"\bjax\b|\bflax\b|\bhaiku\b"),
            ("tensorflow", r"tensorflow|\bkeras\b"),
            ("onnx", r"\bonnx\b"),
        )
        if re.search(pattern, text, re.I)
    ]


def _items(section: Section, children: list[Section]) -> list[str]:
    """Bullets, table rows, sub-heading lists, or sentences of a section."""
    items: list[str] = []
    for line in logical_lines(section.own_text):
        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", line):
            item = strip_inline_markdown(
                re.sub(r"^\s*(?:[-*+]|\d+[.)])\s+", "", line)
            )
            if len(item.split()) >= 2:
                items.append(item)
    for table in parse_tables(section.own_text):
        for row in table.rows:
            cells = [c for c in row if c]
            if cells:
                items.append(": ".join(cells[:2]))
    if not items:
        for sentence in _sentences(prose(section.own_text)):
            if len(sentence.split()) >= 4 and not re.match(
                r"^this section (?:describes|documents|provides|lists)",
                sentence,
                re.I,
            ):
                items.append(sentence)
    if prose(section.own_text).rstrip().endswith(":"):
        items = [i for i in items if not i.endswith(":")]
        items.extend(
            re.sub(r"^[^\w]+", "", child.title).strip()
            for child in children
            if not child.own_text.strip()
        )
    return [_short(i) for i in items[:8] if i]


def _section_items(card: CardText, key: str) -> list[str]:
    items: list[str] = []
    for section in _sections_for(card, key):
        children = [
            s
            for s in card.sections
            if s is not section and s.path[: len(section.path)] == section.path
        ]
        for item in _items(section, children):
            if item and item not in items:
                items.append(item)
    return items[:8]


def _developers(card: CardText, ctx: dict) -> list[str]:
    values = list(ctx["kv_developers"])
    for section in _sections_for(card, "developers"):
        line = next(iter(prose(section.own_text).splitlines()), "")
        if line:
            values.append(line)
    cleaned: list[str] = []
    for value in values:
        value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip(" .")
        if value and value not in cleaned and len(value) < 200:
            cleaned.append(value)
    return cleaned[:5]


def _contact(card: CardText) -> str | None:
    for section in _sections_for(card, "contact"):
        email = P.EMAIL.search(section.own_text)
        if email:
            return email.group(0)
    return None


def _resolutions(ctx: dict) -> list[str]:
    values: list[str] = []
    for value, _ in sorted(ctx["resolution"], key=lambda r: -r[1]):
        if value not in values:
            values.append(value)
    return values[:4]


def _default_id(doc: CardDocument) -> str:
    if doc.repo_id:
        return f"hf:{doc.repo_id}"
    if doc.mcp_tool and doc.mcp_tool.get("name"):
        return (
            f"mcp:{doc.mcp_tool.get('server') or 'mcp'}:{doc.mcp_tool['name']}"
        )
    stem = re.sub(r"[^\w.-]+", "-", doc.name_hint or doc.source.uri).strip("-")
    return f"{doc.source.kind}:{stem.lower()}"


def _finalize_source(doc: CardDocument, card: CardText) -> CardSource:
    source = doc.source.model_copy(deep=True)
    if not source.retrieved_at:
        source.retrieved_at = datetime.now(UTC).isoformat(timespec="seconds")
    payload = json.dumps(
        {"text": doc.text, "config": doc.config, "mcp": doc.mcp_tool},
        sort_keys=True,
        default=str,
    )
    source.sha256 = hashlib.sha256(payload.encode()).hexdigest()
    source.dialect = _dialect(doc, card)
    return source


def _dialect(doc: CardDocument, card: CardText) -> str:
    if doc.native:
        return "ursa"
    if not (doc.text or "").strip():
        return "mcp_only" if doc.mcp_tool else "config_only"
    titles = {s.key for s in card.sections}
    tags = {str(t).lower() for t in card.front_matter.get("tags") or [] if t}
    if "project:genesis" in tags or len(titles & _GENESIS_HEADINGS) >= 3:
        return "genesis"
    keys = {k.lower() for k, _ in parse_key_values(card.clean_body)}
    if {
        "input type",
        "input format",
    } <= keys or "number of model parameters" in keys:
        return "nvidia"
    if len(titles & _HF_HEADINGS) >= 2:
        return "huggingface"
    return "freeform"


# Main entry point -------------------------------------------------------------


def extract_model_card(
    doc: CardDocument,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> ModelCard:
    """Build a normalized :class:`ModelCard` from raw card materials."""
    if doc.native:
        return card_from_native(doc, domains=domains, tasks=tasks)

    b = _Builder()
    ctx: dict[str, Any] = {
        "author_tags": [],
        "hub_tags": [],
        "plain_tags": [],
        "datasets": [],
        "declared_metrics": [],
        "declared_frameworks": [],
        "metrics": [],
        "projects": [],
        "arxiv": [],
        "dois": [],
        "files": [],
        "interfaces": [],
        "resolution": [],
        "config_details": {},
        "config_io": {"inputs": [], "outputs": []},
        "io_evidence": [],
        "io_counts": {},
        "domain_votes": [],
        "kv_developers": [],
        "kv_limitations": [],
        "unattributed_tables": 0,
        "domains_vocab": domains,
    }
    card = parse_card_text(doc.text or "")
    ctx["had_front_matter"] = bool(card.front_matter)
    if card.front_matter_error:
        b.issues.append(
            CardIssue(
                severity="warning",
                code="invalid_front_matter",
                message=card.front_matter_error,
            )
        )
    _handle_front_matter(card.front_matter, b, ctx)
    _handle_metadata_blocks(card, b)
    if doc.hub_info:
        _handle_hub_info(doc.hub_info, b, ctx)
    if doc.config:
        _handle_config(doc.config, b, ctx)
    if doc.mcp_tool:
        _handle_mcp_tool(doc.mcp_tool, b, ctx)
    _handle_tags(b, ctx)
    _handle_key_values(parse_key_values(card.clean_body), b, ctx)
    _handle_names(doc, card, b, ctx)
    _handle_parameter_mentions(card, b, doc)
    _handle_spatial_temporal(card, b, ctx)
    summary = _handle_summary(card)

    io = _build_io(card, ctx)
    _merge_config_io(io, ctx)
    _attach_normalization(io, card)
    if ctx.get("mcp_inputs"):
        io["inputs"] = _overlay_specs(ctx["mcp_inputs"], io["inputs"])
    if ctx.get("mcp_outputs"):
        io["outputs"] = _overlay_specs(ctx["mcp_outputs"], io["outputs"])
    _handle_interfaces(doc, card, b, ctx)
    _handle_metrics(card, doc, ctx)
    _handle_training_data(card, ctx)
    _handle_training_scope(card, b)
    links, citations = _handle_links(card, ctx)

    texts = _concept_texts(card, summary, ctx)
    domain_ids, domain_evidence = _vote_concepts(
        domains, texts, ctx["domain_votes"], threshold=2.0, limit=6
    )
    task_votes = []
    if ctx.get("declared_pipeline") in PIPELINE_TASKS:
        pipeline = ctx["declared_pipeline"]
        task_votes.append((
            PIPELINE_TASKS[pipeline],
            3.0,
            f"pipeline_tag: {pipeline}",
            "front_matter",
        ))
    elif ctx.get("hub_pipeline") in PIPELINE_TASKS:
        # The Hub often infers pipeline tags from the library; weak evidence.
        pipeline = ctx["hub_pipeline"]
        task_votes.append((
            PIPELINE_TASKS[pipeline],
            1.0,
            f"pipeline_tag: {pipeline}",
            "hub_api",
        ))
    task_ids, task_evidence = _vote_concepts(
        tasks, texts, task_votes, threshold=1.5, limit=4
    )
    semantics = {spec.semantics for spec in io["outputs"]}
    if (
        "embedding" in task_ids
        and task_evidence["embedding"].method == "pattern"
        and b.get("architecture.base_model")
        and semantics & _PREDICTION_SEMANTICS
        and "embedding" not in semantics
    ):
        # In a fine-tuned predictor, "pooled to produce a single embedding
        # vector" describes the encoder inside it, not what it returns.
        task_ids.remove("embedding")
        del task_evidence["embedding"]

    model = ModelCard(
        id=doc.id_hint or _default_id(doc),
        name=b.get("name") or doc.name_hint or "unnamed-model",
        version=b.get("version"),
        summary=summary,
        source=_finalize_source(doc, card),
        developers=_developers(card, ctx),
        contact=_contact(card),
        license=_license(card, b),
        gated=b.get("gated"),
        links=links,
        citations=citations,
        domains=domain_ids,
        tasks=task_ids,
        keywords=sorted(
            {t.lower() for t, _ in ctx["plain_tags"]}
            | {m.lower() for m in ctx["declared_metrics"]}
        )[:40],
        pipeline_tag=ctx.get("declared_pipeline") or ctx.get("hub_pipeline"),
        library_name=b.get("library_name"),
        architecture=ArchitectureInfo(
            family=_architecture_family(card, b, ctx),
            name=b.get("architecture.name"),
            parameter_count=b.get("architecture.parameter_count"),
            parameter_count_text=P.format_count(
                b.get("architecture.parameter_count")
            ),
            framework=_frameworks(card, ctx, b),
            base_model=b.get("architecture.base_model"),
            details=ctx["config_details"],
        ),
        inputs=io["inputs"],
        outputs=io["outputs"],
        interfaces=ctx["interfaces"],
        training_data=_dedupe_datasets(ctx["datasets"]),
        evaluation=ctx["metrics"][:30],
        compute=_compute(card, b, ctx),
        intended_uses=_section_items(card, "intended_use"),
        out_of_scope=_section_items(card, "out_of_scope"),
        limitations=list(
            dict.fromkeys(
                ctx["kv_limitations"] + _section_items(card, "limitations")
            )
        )[:8],
        risks=_section_items(card, "risks"),
        risk_level=b.get("risk_level"),
        projects=list(dict.fromkeys(ctx["projects"])),
    )
    for alias in (ctx.get("title_short"), ctx.get("title")):
        if alias and alias != model.name and alias not in model.aliases:
            model.aliases.append(alias)
    model.spatial.n_dims = b.get("spatial.n_dims")
    model.spatial.grid = b.get("spatial.grid")
    model.spatial.resolution = _resolutions(ctx)
    model.temporal.time_step = b.get("temporal.time_step")
    model.temporal.context_steps = b.get("temporal.context_steps")
    model.temporal.horizon = b.get("temporal.horizon")
    model.temporal.sampling_rate = b.get("temporal.sampling_rate")
    model.temporal.window = b.get("temporal.window")
    model.training_scope.examples = b.get("training_scope.examples")
    model.training_scope.ranges = [
        slot.value
        for key, slot in b.slots.items()
        if key.startswith("training_scope.")
        and isinstance(slot.value, DomainRange)
    ]

    provenance = b.provenance()
    for concept, evidence in domain_evidence.items():
        provenance[f"domains.{concept}"] = evidence
    for concept, evidence in task_evidence.items():
        provenance[f"tasks.{concept}"] = evidence
    for role in ("inputs", "outputs"):
        # The tool schema is the runtime interface, so it is the best source
        # for the parameters it defines.
        for spec in ctx.get(f"mcp_{role}") or []:
            provenance.setdefault(
                f"{role}.{spec.name}",
                Evidence(
                    source="mcp_schema",
                    method="structured",
                    quote=_short(
                        f"{spec.name}: {spec.description}"
                        if spec.description
                        else spec.name
                    ),
                ),
            )
    for role, name, source, text in ctx["io_evidence"]:
        provenance.setdefault(
            f"{role}.{name}",
            Evidence(
                source=source,
                method="pattern",
                quote=_short(strip_inline_markdown(text)),
                confidence=0.7,
            ),
        )
    model.provenance = provenance
    model.issues = b.issues
    _post_checks(model, card, doc, ctx)
    return model


# Post-checks: quality issues and completeness ---------------------------------


def _post_checks(
    model: ModelCard, card: CardText, doc: CardDocument, ctx: dict
) -> None:
    issues = model.issues
    words = len(prose(card.clean_body).split())
    if not (doc.text or "").strip():
        if doc.access_error or (model.gated and not doc.mcp_tool):
            issues.append(
                CardIssue(
                    severity="warning",
                    code="card_inaccessible",
                    message=doc.access_error
                    or "The card is not accessible (gated repository); request access or provide a token.",
                )
            )
        elif not doc.mcp_tool:
            issues.append(
                CardIssue(
                    severity="warning",
                    code="card_missing",
                    message="No model card text; facts come from configuration files only.",
                )
            )
    elif words < 120:
        issues.append(
            CardIssue(
                severity="warning",
                code="minimal_card",
                message=(
                    f"The card is very short ({words} words of prose); key facts "
                    "may live in the paper or code repository."
                ),
            )
        )
    if card.template_hits or ctx.get("template_front_matter"):
        issues.append(
            CardIssue(
                severity="warning",
                code="template_residue",
                message=(
                    f"Ignored {card.template_hits} line(s) of unfilled template text"
                    + (
                        " and placeholder front matter"
                        if ctx.get("template_front_matter")
                        else ""
                    )
                    + "."
                ),
            )
        )
    if model.gated:
        issues.append(
            CardIssue(
                severity="info",
                code="gated_access",
                message="Weights are gated; request access before running.",
            )
        )
    if ctx.get("shared_card"):
        issues.append(
            CardIssue(
                severity="info",
                code="shared_card",
                message="The card covers several checkpoints; metrics come from the row that matches this model.",
            )
        )
    if ctx["unattributed_tables"]:
        issues.append(
            CardIssue(
                severity="info",
                code="unattributed_metrics",
                message="A multi-row metrics table could not be attributed to this checkpoint and was skipped.",
            )
        )
    stated = ctx["io_counts"].get("inputs")
    listed = next((len(s.channels) for s in model.inputs if s.channels), None)
    if stated and listed and stated != listed:
        issues.append(
            CardIssue(
                severity="warning",
                code="channel_count_mismatch",
                field="inputs",
                message=f"The card states {stated} input variables but lists {listed}.",
            )
        )
    if model.outputs and not any(
        spec.units
        or any(c.units for c in spec.channels)
        or spec.semantics in _NONPHYSICAL
        or spec.kind in {"text", "molecule", "structure", "sequence"}
        for spec in model.outputs
    ):
        issues.append(
            CardIssue(
                severity="info",
                code="units_unspecified",
                field="outputs",
                message=(
                    "Output units are not stated; confirm physical units or "
                    "normalization before interpreting predictions."
                ),
            )
        )
    model.completeness = assess_completeness(model)


def assess_completeness(model: ModelCard) -> Completeness:
    """Score how well a card supports recommendation and execution."""
    field_like = any(
        s.kind in {"field", "tensor", "image"} for s in model.inputs
    )
    recommend = [
        ("summary", bool(model.summary)),
        ("physics domain", bool(model.domains)),
        ("task", bool(model.tasks)),
        ("input or output description", bool(model.inputs or model.outputs)),
        ("license", bool(model.license.spdx or model.license.name)),
    ]
    run = [
        (
            "invocation interface",
            any(
                i.kind == "mcp_tool"
                or i.loader
                or i.snippet
                or i.kind in {"cli", "onnx"}
                for i in model.interfaces
            ),
        ),
        (
            "input format",
            any(
                s.shape or s.representation or s.channels or s.kind
                for s in model.inputs
            ),
        ),
        ("output format", bool(model.outputs)),
    ]
    physics = [
        (
            "output units or normalization",
            any(
                s.units
                or s.normalization
                or any(c.units for c in s.channels)
                or s.semantics in _NONPHYSICAL
                for s in model.outputs
            ),
        )
    ]
    if field_like:
        physics.append((
            "spatial resolution or grid",
            bool(model.spatial.resolution or model.spatial.grid),
        ))
        physics.append((
            "input normalization",
            any(s.normalization for s in model.inputs),
        ))
    if {"forecasting", "surrogate_simulation"} & set(model.tasks):
        physics.append((
            "time semantics",
            bool(
                model.temporal.time_step
                or model.temporal.horizon
                or model.temporal.context_steps
            ),
        ))
    checklist = recommend + run + physics
    weights = [2.0] * (len(recommend) + len(run)) + [1.0] * len(physics)
    score = sum(w for (_, ok), w in zip(checklist, weights) if ok) / sum(
        weights
    )
    return Completeness(
        score=round(score, 3),
        recommend_ready=all(ok for _, ok in recommend[1:4]),
        run_ready=all(ok for _, ok in run),
        missing=[name for name, ok in checklist if not ok],
    )


# URSA-native cards -------------------------------------------------------------


def card_from_native(
    doc: CardDocument,
    *,
    domains: Vocabulary = DOMAINS,
    tasks: Vocabulary = TASKS,
) -> ModelCard:
    """Validate an URSA-native YAML/JSON card, normalizing loose fields."""
    native = dict(doc.native or {})
    data = dict(native)
    data.setdefault("id", doc.id_hint or _default_id(doc))
    data.setdefault("name", doc.name_hint or data["id"])
    data["source"] = doc.source.model_copy(
        update={"dialect": "ursa"}
    ).model_dump()
    if isinstance(data.get("license"), str):
        data["license"] = {
            "spdx": P.normalize_license(data["license"]) or data["license"]
        }
    for role in ("inputs", "outputs"):
        specs = []
        for item in data.get(role) or []:
            item = {"name": item} if isinstance(item, str) else item
            if not isinstance(item, dict):
                continue
            item = dict(item)
            if isinstance(item.get("shape"), str):
                item["shape"] = P.parse_shape(item["shape"]) or [item["shape"]]
            elif isinstance(item.get("shape"), list):
                item["shape"] = [str(d) for d in item["shape"]]
            if isinstance(item.get("channels"), list):
                item["channels"] = [
                    {"name": c} if isinstance(c, str) else c
                    for c in item["channels"]
                ]
            if isinstance(item.get("file_formats"), str):
                item["file_formats"] = [item["file_formats"]]
            specs.append(item)
        data[role] = specs
    for key, vocab in (("domains", domains), ("tasks", tasks)):
        values = data.get(key) or []
        values = [values] if isinstance(values, str) else values
        data[key] = list(
            dict.fromkeys(vocab.normalize(str(v)) or str(v) for v in values)
        )
    parameters = data.pop("parameters", None)
    if parameters is not None:
        architecture = dict(data.get("architecture") or {})
        if isinstance(parameters, int):
            architecture.setdefault("parameter_count", parameters)
        elif isinstance(parameters, str):
            match = re.search(
                rf"(?P<num>{P.NUMBER})\s*(?P<unit>[KMBT])?", parameters
            )
            if match:
                architecture.setdefault(
                    "parameter_count",
                    P.parse_count(match.group("num"), match.group("unit")),
                )
        data["architecture"] = architecture
    data["interfaces"] = [
        i for i in data.get("interfaces") or [] if isinstance(i, dict)
    ]
    model = ModelCard.model_validate(data)
    lic = model.license
    if lic.commercial_use is None:
        lic.commercial_use = P.commercial_use_for(lic.spdx)
    count = model.architecture.parameter_count
    if count and not model.architecture.parameter_count_text:
        model.architecture.parameter_count_text = P.format_count(count)
    if count and model.compute.estimated_weights_gb is None:
        model.compute.estimated_weights_gb = round(count * 4 / 1e9, 3)
    for path in (
        "name",
        "license",
        "domains",
        "tasks",
        "inputs",
        "outputs",
        "interfaces",
    ):
        if native.get(path):
            model.provenance.setdefault(
                path, Evidence(source=doc.source.uri, method="declared")
            )
    if not (lic.spdx or lic.name):
        model.issues.append(
            CardIssue(
                severity="warning",
                code="license_missing",
                field="license",
                message="The card does not state a license.",
            )
        )
    model.completeness = assess_completeness(model)
    return model
