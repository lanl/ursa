"""Plan and prepare runs of a selected model.

Two execution paths exist:

- ``mcp_tool``: the model is served as an MCP tool (for example by Nomad or
  by URSA's ``TorchModuleTool``). Arguments are validated against the
  tool's JSON schema before the call.
- ``execution_agent``: anything else. The plan becomes a brief for URSA's
  ExecutionAgent that carries the card's exact parameter formats,
  normalization, time semantics, and loading code, plus the checks and
  review figures the run must produce.

The planner never invents parameter values. Missing required arguments are
listed in the plan so the agent can ask the user for them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ursa.model_cards.data import DataProfile, is_smiles, read_items
from ursa.model_cards.extract import io_specs_from_json_schema
from ursa.model_cards.schema import (
    Interface,
    IOSpec,
    ModelCard,
    UserRequirements,
)
from ursa.util.structured_output import invoke_structured

_INTERFACE_ORDER = {
    "mcp_tool": 0,
    "python": 1,
    "huggingface": 2,
    "cli": 3,
    "onnx": 4,
    "container": 5,
    "other": 6,
}
_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


class RunPlan(BaseModel):
    model_id: str
    model_name: str
    mode: Literal["mcp_tool", "execution_agent", "unavailable"]
    interface: Interface | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    missing_arguments: list[str] = Field(default_factory=list)
    argument_errors: list[str] = Field(default_factory=list)
    data_paths: list[str] = Field(default_factory=list)
    batch: list[dict[str, Any]] = Field(
        default_factory=list,
        description="One argument set per item of the user's data file; empty for a single call.",
    )
    batch_names: list[str | None] = Field(default_factory=list)
    batch_source: str | None = None
    output_dir: str = ""
    checks: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ArgumentFill(BaseModel):
    """Tool arguments filled from the request."""

    arguments_json: str = Field(
        default="{}",
        description="JSON object with argument values taken from the request; omit unknown values.",
    )
    missing: list[str] = Field(
        default_factory=list,
        description="Required arguments the request does not provide.",
    )


def choose_interface(card: ModelCard) -> Interface | None:
    """Prefer an MCP tool, then documented loaders, CLIs, and raw repos."""
    usable = [
        i
        for i in card.interfaces
        if i.kind == "mcp_tool"
        or i.loader
        or i.snippet
        or i.kind in {"cli", "onnx", "python", "huggingface"}
    ]
    if not usable:
        return None
    return sorted(
        usable,
        key=lambda i: (
            _INTERFACE_ORDER.get(i.kind, 9),
            not (i.loader or i.snippet),
        ),
    )[0]


def _tensor_like(schema: dict[str, Any] | None) -> bool:
    text = json.dumps(schema or {})
    return (
        "vnd.nomad.tensor" in text
        or "torch.Tensor" in text
        or "t0_fields" in text
    )


def validate_arguments(
    schema: dict[str, Any] | None, arguments: dict[str, Any]
) -> list[str]:
    """Validate arguments against a JSON schema; returns error messages."""
    if not schema:
        return []
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema ships with mcp
        missing = [
            name for name in schema.get("required", []) if name not in arguments
        ]
        return [f"'{name}' is a required property" for name in missing]
    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)
    return sorted(error.message for error in validator.iter_errors(arguments))


def parse_json_arguments(text: str) -> dict[str, Any] | None:
    """Return the first JSON object embedded in text, if any."""
    for match in _JSON_OBJECT.finditer(text or ""):
        candidate = match.group(0)
        for end in range(len(candidate), 1, -1):
            if candidate[end - 1] != "}":
                continue
            try:
                value = json.loads(candidate[:end])
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    return None


# "smi CCO", "SMILES: CCO", or "smiles=`CCO`" state a SMILES argument.
_SMILES_CUE = re.compile(
    r"\b(?:smi|smiles)\b\s*(?:string\s*)?[:=]?\s*[`'\"]?(?P<value>[^\s`'\",;]+)",
    re.I,
)
_CODE_SPAN = re.compile(r"`([^`\s]+)`")
_ATOM = re.compile(r"Cl|Br|[BCNOPSFI]|[cnops]")


def item_argument(interface: Interface) -> tuple[str, str] | None:
    """The one required argument that takes a molecule or a sequence.

    Returns the argument name and its kind. A tool such as Nomad's MIST
    models takes one SMILES string per call, so a file of molecules runs as
    one call per molecule.
    """
    specs = io_specs_from_json_schema(interface.input_schema or {})
    required = [spec for spec in specs if spec.required]
    if len(required) == 1 and required[0].kind in {"molecule", "sequence"}:
        return required[0].name, required[0].kind
    return None


def smiles_in(text: str) -> str | None:
    """A SMILES string the text states, such as "smi CCO" or "`CCO`".

    A bare word counts only when it has SMILES-only characters (bonds,
    branches, or brackets), so English words never read as molecules.
    """
    for match in _SMILES_CUE.finditer(text or ""):
        value = match.group("value").rstrip(".")
        if is_smiles(value) and _ATOM.search(value):
            return value
    for match in _CODE_SPAN.finditer(text or ""):
        if is_smiles(match.group(1)) and _ATOM.search(match.group(1)):
            return match.group(1)
    for token in re.findall(r"[^\s,;'\"`]+", text or ""):
        token = token.rstrip(".")
        if (
            re.search(r"[=#()\[\]]", token)
            and is_smiles(token)
            and _ATOM.search(token)
        ):
            return token
    return None


def _stated_arguments(
    interface: Interface, request: str
) -> dict[str, Any] | None:
    item = item_argument(interface)
    if item is None or item[1] != "molecule":
        return None
    value = smiles_in(request)
    return {item[0]: value} if value else None


def fill_arguments(
    interface: Interface,
    request: str,
    llm: Any = None,
    *,
    config: Any = None,
    repair_rounds: int = 2,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Fill MCP tool arguments: explicit JSON, a stated SMILES, then the LLM.

    Returns (arguments, missing_required, validation_errors).
    """
    schema = interface.input_schema or {}
    required = list(schema.get("required") or [])
    explicit = parse_json_arguments(request)
    if explicit is None:
        explicit = _stated_arguments(interface, request)
    arguments = explicit if explicit is not None else {}
    errors = validate_arguments(schema, arguments) if arguments else []
    if (explicit is None or errors) and llm is not None:
        feedback = ""
        for _ in range(repair_rounds + 1):
            messages = [
                SystemMessage(
                    content=(
                        "Fill the arguments of a scientific model tool from the user's request. "
                        "Use only values the request states; never invent scientific inputs. "
                        "List required arguments the request does not provide in 'missing'."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Tool: {interface.name}\nDescription: {interface.notes or ''}\n"
                        f"JSON schema:\n{json.dumps(schema, indent=1)[:6000]}\n\n"
                        f"Request:\n{request}\n{feedback}"
                    )
                ),
            ]
            fill = invoke_structured(
                llm,
                ArgumentFill,
                messages,
                config=config,
                context="tool arguments",
                fallback=ArgumentFill(),
                repair=1,
            )
            candidate = parse_json_arguments(fill.arguments_json) or {}
            errors = validate_arguments(schema, candidate) if candidate else []
            arguments = candidate
            if candidate and not errors:
                break
            feedback = f"\nPrevious attempt {fill.arguments_json!r} failed validation: {errors}"
    missing = [name for name in required if name not in arguments]
    if missing:
        errors = [e for e in errors if not any(f"'{m}'" in e for m in missing)]
    return arguments, missing, errors


def _slug(text: str) -> str:
    return (
        re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-").lower()[:60]
        or "model"
    )


def build_run_plan(
    card: ModelCard,
    request: str,
    requirements: UserRequirements | None = None,
    *,
    llm: Any = None,
    config: Any = None,
    output_root: str = "fm_runs",
    profiles: list[DataProfile] | None = None,
    max_items: int = 100,
) -> RunPlan:
    """Plan how to run ``card`` for ``request``.

    When the tool takes one molecule or sequence per call and the user named
    a file of them, the plan runs the tool once per item, up to ``max_items``.
    """
    interface = choose_interface(card)
    plan = RunPlan(
        model_id=card.id,
        model_name=card.name,
        mode="unavailable",
        interface=interface,
        data_paths=list(
            (requirements.data_paths if requirements else []) or []
        ),
        output_dir=f"{output_root}/{_slug(card.name)}",
    )
    if interface is None:
        plan.warnings.append(
            "The card documents no way to load or call the model; consult its linked repository."
        )
        return plan
    if card.gated:
        plan.warnings.append("Weights are gated; the run fails without access.")
    for issue in card.issues:
        # A served model runs on its server, so no repository code loads
        # here and the remote-code warning does not apply.
        if issue.code == "remote_code" and interface.kind == "mcp_tool":
            continue
        if issue.code in {
            "remote_code",
            "license_conflict",
            "units_unspecified",
        }:
            plan.warnings.append(issue.message)
    if interface.kind == "mcp_tool" and not _tensor_like(
        interface.input_schema
    ):
        plan.mode = "mcp_tool"
        item = item_argument(interface)
        source = next(
            (
                p
                for p in profiles or []
                if item is not None and p.kind == item[1]
            ),
            None,
        )
        if source is not None and parse_json_arguments(request) is None:
            items = read_items(source, limit=max_items + 1)
            plan.batch = [{item[0]: value} for value, _ in items[:max_items]]
            plan.batch_names = [name for _, name in items[:max_items]]
            plan.batch_source = source.path
            plan.arguments = plan.batch[0] if plan.batch else {}
            if not plan.batch:
                plan.missing_arguments = [item[0]]
            if len(items) > max_items:
                plan.warnings.append(
                    f"Only the first {max_items} items of {source.name} are run; "
                    "raise max_run_items to run more."
                )
        else:
            plan.arguments, plan.missing_arguments, plan.argument_errors = (
                fill_arguments(interface, request, llm, config=config)
            )
    else:
        plan.mode = "execution_agent"
        if interface.kind == "mcp_tool":
            plan.warnings.append(
                "The tool takes tensor or WellFormat inputs; build them in a Python client script "
                "(see Nomad's code-mode gateway) rather than passing raw JSON."
            )
    plan.checks = default_checks(card)
    return plan


def default_checks(card: ModelCard) -> list[str]:
    """Scientific sanity checks implied by the card's formats."""
    checks = ["Output values are finite (no NaN or Inf)."]
    for spec in card.inputs:
        if spec.shape:
            checks.append(
                f"Input '{spec.name}' matches shape ({', '.join(spec.shape)})."
            )
        if spec.normalization:
            checks.append(
                f"Input '{spec.name}' is preprocessed as documented: {spec.normalization}"
            )
        if spec.channels:
            names = ", ".join(c.name for c in spec.channels[:12])
            checks.append(
                f"Input channels appear in the documented order: {names}."
            )
    for spec in card.outputs:
        if spec.channels and any(c.units for c in spec.channels):
            checks.append(f"Report '{spec.name}' channels with their units.")
        elif spec.units:
            checks.append(f"Report '{spec.name}' in {spec.units}.")
        if spec.semantics in {"delta"}:
            checks.append(
                f"'{spec.name}' is a change; add it to the input state for absolute values."
            )
    if card.spatial.resolution:
        checks.append(
            f"Spatial grid matches the trained resolution ({'; '.join(card.spatial.resolution)})."
        )
    if card.temporal.time_step:
        checks.append(
            f"Time step matches the trained step ({card.temporal.time_step})."
        )
    return checks


def _io_lines(specs: list[IOSpec]) -> list[str]:
    lines = []
    for spec in specs:
        facts = [
            f"kind={spec.kind}" if spec.kind else "",
            f"representation={spec.representation}"
            if spec.representation
            else "",
            f"shape=({', '.join(spec.shape)})" if spec.shape else "",
            f"dtype={spec.dtype}" if spec.dtype else "",
            f"units={spec.units}" if spec.units else "",
            f"semantics={spec.semantics}" if spec.semantics else "",
            f"formats={', '.join(spec.file_formats)}"
            if spec.file_formats
            else "",
        ]
        line = f"- {spec.name}: " + "; ".join(f for f in facts if f)
        if spec.channels:
            channel_text = ", ".join(
                c.name + (f" [{c.units}]" if c.units else "")
                for c in spec.channels[:40]
            )
            line += f"\n  channels (in order): {channel_text}"
        if spec.normalization:
            line += f"\n  normalization: {spec.normalization}"
        if spec.description:
            line += f"\n  card text: {spec.description}"
        lines.append(line)
    return lines or ["- not specified by the card"]


def execution_brief(card: ModelCard, plan: RunPlan, request: str) -> str:
    """Instructions for URSA's ExecutionAgent to run the model safely."""
    interface = plan.interface
    lines = [
        f"Run the scientific foundation model '{card.name}' ({card.id}) for this request:",
        request.strip(),
        "",
        "Model facts from its card (treat as the contract; do not guess beyond them):",
        f"- Summary: {card.summary or 'not stated'}",
    ]
    if interface is not None:
        if interface.repo_id:
            lines.append(f"- Hugging Face repository: {interface.repo_id}")
        if interface.loader:
            lines.append(f"- Loader: {interface.loader}")
        if interface.trust_remote_code:
            lines.append(
                "- Loading requires trust_remote_code=True (runs repository code)."
            )
        if interface.kind == "mcp_tool":
            lines.append(
                f"- MCP tool: {interface.name} on server {interface.server}"
            )
        if interface.kind == "cli" and interface.name:
            lines.append(f"- Command-line entry point: {interface.name}")
        if interface.install:
            lines.append("- Install: " + " ; ".join(interface.install))
        if interface.snippet:
            lines += [
                "- Usage snippet from the card:",
                "```python",
                interface.snippet,
                "```",
            ]
    lines += [
        "",
        "Inputs:",
        *_io_lines(card.inputs),
        "",
        "Outputs:",
        *_io_lines(card.outputs),
    ]
    spatial = card.spatial.model_dump(exclude_defaults=True)
    temporal = card.temporal.model_dump(exclude_defaults=True)
    if spatial:
        lines.append(f"Spatial: {json.dumps(spatial)}")
    if temporal:
        lines.append(f"Temporal: {json.dumps(temporal)}")
    if card.compute.model_dump(exclude_defaults=True):
        lines.append(
            f"Compute: {json.dumps(card.compute.model_dump(exclude_defaults=True))}"
        )
    if card.limitations:
        lines.append("Limitations: " + " | ".join(card.limitations[:4]))
    if plan.data_paths:
        lines.append("User data: " + ", ".join(plan.data_paths))
    lines += [
        "",
        "Steps:",
        f"1. Work inside '{plan.output_dir}'. Write a script run_model.py there; keep it reusable.",
        "2. Install only the dependencies the card lists. Stop and report if weights are gated or a download fails.",
        "3. Load the user's data and convert it to the documented input format. Do not fabricate inputs: "
        "if the request lacks required data, stop and say what is missing.",
        "4. Run inference on one sample first, then the full request. Use the GPU if available.",
        "5. Save outputs in an open format (NetCDF, HDF5, NPZ, or JSON) with units and a run_summary.json that records "
        "the model id, inputs, shapes, device, runtime, and the checks below.",
        "6. Save review figures (inputs, outputs, and their difference where meaningful) as PNG with clear labels and units.",
        "",
        "Checks to report:",
        *[f"- {check}" for check in plan.checks],
    ]
    if plan.warnings:
        lines += [
            "",
            "Warnings:",
            *[f"- {warning}" for warning in plan.warnings],
        ]
    return "\n".join(lines)


def summarize_tool_result(result: Any, limit: int = 4000) -> str:
    """Readable text for an MCP tool result."""
    if isinstance(result, tuple) and result:
        result = result[0]
    if isinstance(result, list):
        parts = []
        for block in result:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        result = "\n".join(parts)
    if not isinstance(result, str):
        try:
            result = json.dumps(result, indent=2, default=str)
        except TypeError:
            result = str(result)
    return (
        result if len(result) <= limit else result[:limit] + "\n... [truncated]"
    )


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    """Return ``folder/stem-<time><suffix>`` that does not exist yet.

    Runs and reports can finish within the same second, so a counter
    follows the timestamp when needed instead of overwriting a file.
    """
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = folder / f"{stem}-{stamp}{suffix}"
    count = 2
    while path.exists():
        path = folder / f"{stem}-{stamp}-{count}{suffix}"
        count += 1
    return path


_CONTAINER_KEYS = {"properties", "result", "results"}


def flatten_result(result: Any, prefix: str = "") -> dict[str, Any]:
    """Scalar columns for one tool result, for a results table.

    A ``{"value": ..., "units": ...}`` record becomes one column named after
    its key, with the units in the header, as in Nomad's property results:
    ``{"properties": {"bp": {"value": 108.0, "units": "°C"}}}`` gives the
    column ``bp [°C]``.
    """
    if isinstance(result, dict):
        if "value" in result and not isinstance(result["value"], (dict, list)):
            units = result.get("units")
            name = prefix or "value"
            return {f"{name} [{units}]" if units else name: result["value"]}
        columns: dict[str, Any] = {}
        for key, value in result.items():
            if key == "description" and isinstance(value, str):
                continue
            # Container keys such as "properties" add no meaning to a column name.
            if key in _CONTAINER_KEYS:
                name = prefix
            else:
                name = f"{prefix}.{key}" if prefix else str(key)
            columns.update(flatten_result(value, name))
        return columns
    if isinstance(result, list):
        text = json.dumps(result, default=str)
        return {
            prefix or "value": text if len(text) <= 200 else text[:200] + "…"
        }
    return {prefix or "value": result}


def save_batch_results(
    rows: list[dict[str, Any]], plan: RunPlan, workspace: Path
) -> tuple[Path, Path]:
    """Write one batch of tool results as JSON and as a CSV table."""
    import csv

    target = unique_path(Path(workspace) / plan.output_dir, "results", ".json")
    target.write_text(
        json.dumps(
            {
                "model_id": plan.model_id,
                "source": plan.batch_source,
                "results": rows,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    table = [
        {
            "item": next(iter(row["arguments"].values()), ""),
            "name": row.get("name") or "",
            **flatten_result(row.get("result")),
            "error": row.get("error") or "",
        }
        for row in rows
    ]
    columns = list(dict.fromkeys(key for row in table for key in row))
    csv_path = target.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    return target, csv_path


def save_tool_result(result_text: str, plan: RunPlan, workspace: Path) -> Path:
    """Write an MCP tool result and its arguments under the run directory.

    Each call writes a new ``result-<time>.json`` so repeated runs of one
    model keep their earlier results.
    """
    target = unique_path(Path(workspace) / plan.output_dir, "result", ".json")
    try:
        payload: Any = json.loads(result_text)
    except ValueError:
        payload = result_text
    target.write_text(
        json.dumps(
            {
                "model_id": plan.model_id,
                "arguments": plan.arguments,
                "result": payload,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return target
