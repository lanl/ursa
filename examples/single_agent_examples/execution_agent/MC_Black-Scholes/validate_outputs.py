#!/usr/bin/env python3
"""Validate outputs + policy compliance for the ExecutionAgent Monte Carlo example.

Behavior
--------
- Non-interactive.
- Writes a machine-readable report to: `<output-dir>/validation.json`.
- Terminal output is:
  - Enabled by default on a TTY (Rich if available)
  - Disabled when not a TTY, or with `--quiet`, or with `URSA_QUIET=1`

Robustness
----------
- Validation is schema-/contract-based.
- The presence of unrelated extra files does not cause failure.
- Guardrail scans are limited to known relevant artifacts (code + a few known logs)
  under the chosen output directory.

Exit codes
----------
- 0: validation OK
- 1: validation failed
- 2: validator error (unexpected exception)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ux import UX, add_ux_args, resolve_ux_config, setup_file_logging

EXPECTED_COLUMNS = [
    "n_paths",
    "mc_price",
    "mc_stderr",
    "ci95_low",
    "ci95_high",
    "bs_price",
    "abs_error",
    "runtime_seconds",
]
EXPECTED_N_PATHS = [1000, 5000, 20000, 100000]

# Guardrail patterns (best-effort). These are intended to catch obvious policy violations.
BANNED_PATTERNS: list[tuple[str, str]] = [
    ("pip_install", r"\bpip\s+install\b"),
    ("conda_install", r"\bconda\s+install\b"),
    ("apt_get", r"\bapt-get\b"),
    ("brew", r"\bbrew\s+install\b"),
    ("curl", r"\bcurl\b"),
    ("wget", r"\bwget\b"),
    ("requests_get", r"\brequests\.get\b"),
    ("urllib_request", r"\burllib\.request\b"),
    ("socket", r"\bsocket\."),
    # Any explicit URLs are suspicious; we allow OpenAI endpoints as part of LLM usage.
    ("http_url", r"https?://"),
]

# Whitelist: URLs related to OpenAI model calls may appear in logs; don't flag those.
OPENAI_URL_WHITELIST = [
    "api.openai.com",
    "openai.com/v1",
]


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    details: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _file_nonempty(path: Path) -> bool:
    try:
        return path.exists() and path.is_file() and path.stat().st_size > 0
    except Exception:
        return False


def _scan_text_for_banned(*, text: str, source: str) -> list[str]:
    findings: list[str] = []

    # If text contains URLs, ignore OpenAI endpoints.
    lowered = text.lower()
    contains_openai_url = any(w in lowered for w in OPENAI_URL_WHITELIST)

    for name, pat in BANNED_PATTERNS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            snippet = text[
                max(0, m.start() - 40) : min(len(text), m.end() + 40)
            ].replace("\n", " ")

            if name == "http_url" and contains_openai_url:
                # Still flag non-OpenAI URLs, but the naive detector can't distinguish well.
                # Best-effort: if the surrounding snippet includes an OpenAI URL, skip.
                if any(w in snippet.lower() for w in OPENAI_URL_WHITELIST):
                    continue

            findings.append(
                f"{source}: banned_pattern={name} snippet={snippet!r}"
            )

    return findings


def _is_under(base: Path, p: Path) -> bool:
    try:
        p.relative_to(base)
        return True
    except Exception:
        return False


def _resolve_under_here(
    here: Path, user_path: str | None, *, default_rel: str
) -> Path:
    raw = (
        default_rel
        if (user_path is None or str(user_path).strip() == "")
        else str(user_path)
    )
    p = Path(raw)
    if not p.is_absolute():
        p = here / p
    p = p.resolve()

    if not _is_under(here.resolve(), p):
        raise ValueError(
            f"Refusing to read/write outside the example directory: {p}"
        )

    return p


def validate(*, here: Path, outputs_dir: Path) -> ValidationResult:
    outputs_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {
        "outputs_dir": str(outputs_dir.relative_to(here)),
    }

    # --- Preflight: plotting requirement
    preflight_path = outputs_dir / "preflight.json"
    preflight = _read_json(preflight_path) if preflight_path.exists() else None
    plotting_enabled = bool(
        ((preflight or {}).get("optional") or {}).get("plotting_enabled")
    )
    details["plotting_enabled"] = plotting_enabled

    # --- Required artifacts
    code_path = here / "mc_option_convergence.py"
    report_path = outputs_dir / "report.md"
    csv_path = outputs_dir / "results.csv"
    plot_path = outputs_dir / "plot.png"

    if not _file_nonempty(code_path):
        errors.append(
            "Missing or empty required file: mc_option_convergence.py"
        )

    if not _file_nonempty(report_path):
        errors.append(
            f"Missing or empty required artifact: {report_path.relative_to(here)}"
        )
    if not _file_nonempty(csv_path):
        errors.append(
            f"Missing or empty required artifact: {csv_path.relative_to(here)}"
        )

    if plotting_enabled:
        if not _file_nonempty(plot_path):
            errors.append(
                f"Preflight indicates plotting_enabled=true, but {plot_path.relative_to(here)} is missing or empty"
            )
    else:
        if plot_path.exists() and not _file_nonempty(plot_path):
            warnings.append(
                f"{plot_path.relative_to(here)} exists but is empty"
            )

    # --- CSV content checks
    rows: list[dict[str, str]] = []
    header: list[str] = []
    if _file_nonempty(csv_path):
        try:
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                header = list(reader.fieldnames or [])
                rows.extend(dict(r) for r in reader)
        except Exception as e:
            errors.append(
                f"Failed to parse {csv_path.relative_to(here)}: {type(e).__name__}: {e}"
            )

    details["csv_header"] = header
    details["csv_row_count"] = len(rows)

    if header:
        missing_cols = [c for c in EXPECTED_COLUMNS if c not in header]
        if missing_cols:
            errors.append(
                f"{csv_path.relative_to(here)} missing expected columns: "
                + ", ".join(missing_cols)
            )

    # Row count bounds: should match our fixed list, but allow small flexibility.
    if rows:
        if not (4 <= len(rows) <= 8):
            errors.append(
                f"Unexpected results.csv row count {len(rows)} (expected 4..8)"
            )

        # Validate n_paths values
        try:
            n_vals = [int(float(r.get("n_paths", "nan"))) for r in rows]
        except Exception:
            n_vals = []
            errors.append("Could not parse n_paths values as integers")

        details["n_paths_values"] = n_vals

        if n_vals:
            missing_n = [n for n in EXPECTED_N_PATHS if n not in n_vals]
            if missing_n:
                errors.append(
                    "results.csv missing expected n_paths rows: "
                    + ", ".join(map(str, missing_n))
                )

        def _f(row: dict[str, str], key: str) -> float | None:
            v = row.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        for i, r in enumerate(rows):
            mc_stderr = _f(r, "mc_stderr")
            ci_low = _f(r, "ci95_low")
            ci_high = _f(r, "ci95_high")
            runtime = _f(r, "runtime_seconds")
            if mc_stderr is None or mc_stderr < 0:
                errors.append(f"Row {i}: mc_stderr missing or negative")
            if runtime is None or runtime < 0:
                errors.append(f"Row {i}: runtime_seconds missing or negative")
            if ci_low is None or ci_high is None or ci_low > ci_high:
                errors.append(f"Row {i}: invalid CI bounds")

    # --- Report content checks
    report_text = (
        report_path.read_text(encoding="utf-8")
        if _file_nonempty(report_path)
        else ""
    )
    details["report_length_chars"] = len(report_text)

    required_markers = [
        "Parameters",
        "Black",  # coarse guard; we do a more specific check below
        "Exact rerun command",
        "python mc_option_convergence.py",
    ]
    errors.extend(
        f"{report_path.relative_to(here)} missing required text: {marker!r}"
        for marker in required_markers
        if marker not in report_text
    )

    if ("Black–Scholes" not in report_text) and (
        "Black-Scholes" not in report_text
    ):
        errors.append(
            f"{report_path.relative_to(here)} missing required text: 'Black–Scholes' (or 'Black-Scholes')"
        )

    if ("## Summary" not in report_text) and ("Summary:" not in report_text):
        errors.append(
            f"{report_path.relative_to(here)} missing a Summary section (expected '## Summary' or 'Summary:')"
        )

    # --- Guardrails: scan for obvious banned operations
    # Note: we do NOT scan arbitrary extra files. This avoids false positives due to
    # unrelated logs/artifacts.
    scan_files = [
        here / "mc_option_convergence.py",
        outputs_dir / "run.log",
        outputs_dir / "stdouterr.log",
        outputs_dir / "agent_response.md",
        outputs_dir / "agent_state.json",
    ]

    guardrail_findings: list[str] = []
    for p in scan_files:
        if not p.exists() or not p.is_file():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if p.name.endswith(".py") and re.search(r"\binput\s*\(", text):
            guardrail_findings.append(f"{p.name}: found interactive input()")

        guardrail_findings.extend(
            _scan_text_for_banned(text=text, source=p.name)
        )

    if (outputs_dir / "agent_state.json").exists():
        try:
            st = json.loads(
                (outputs_dir / "agent_state.json").read_text(encoding="utf-8")
            )
            st_text = json.dumps(st)
            guardrail_findings.extend(
                f"agent_state.json: found disallowed tool reference {tool_name}"
                for tool_name in (
                    "run_web_search",
                    "run_arxiv_search",
                    "run_osti_search",
                )
                if tool_name in st_text
            )
        except Exception:
            pass

    if guardrail_findings:
        errors.extend(["Guardrail violation: " + f for f in guardrail_findings])

    ok = len(errors) == 0
    return ValidationResult(
        ok=ok, errors=errors, warnings=warnings, details=details
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate artifacts under an outputs directory"
    )
    add_ux_args(p)
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory (relative to this example) containing artifacts (default: outputs).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent

    args = _build_arg_parser().parse_args(argv)

    outputs_dir = _resolve_under_here(
        here, args.output_dir, default_rel="outputs"
    )
    outputs_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_ux_config(args=args, default_log_file=None)
    ux = UX(cfg)

    log_path = None
    if cfg.log_file is not None:
        log_path = _resolve_under_here(here, str(cfg.log_file), default_rel="")

    logger = setup_file_logging(log_file=log_path)

    validation_path = outputs_dir / "validation.json"

    try:
        res = validate(here=here, outputs_dir=outputs_dir)
        payload = {
            "ok": res.ok,
            "errors": res.errors,
            "warnings": res.warnings,
            "details": res.details,
        }
        validation_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "validation ok=%s errors=%d warnings=%d",
            res.ok,
            len(res.errors),
            len(res.warnings),
        )

        if not cfg.quiet:
            if res.ok:
                ux.panel("Validation", "OK — outputs satisfy the contract.")
            else:
                ux.panel(
                    "Validation",
                    f"FAILED — {len(res.errors)} error(s). See {validation_path.relative_to(here)}",
                )
                for i, e in enumerate(res.errors[:10]):
                    ux.print(f"[{i + 1}] {e}")
                if len(res.errors) > 10:
                    ux.print(f"... and {len(res.errors) - 10} more")

        return 0 if res.ok else 1

    except Exception as e:
        payload = {
            "ok": False,
            "errors": [f"Validator exception: {type(e).__name__}: {e}"],
            "warnings": [],
        }
        try:
            validation_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

        try:
            logger.exception("validator exception")
        except Exception:
            pass

        if not cfg.quiet:
            ux.panel("Validator exception", f"{type(e).__name__}: {e}")

        return 2


if __name__ == "__main__":
    raise SystemExit(main())
