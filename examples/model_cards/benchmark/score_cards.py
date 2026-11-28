"""Score URSA's model-card recognition against hand-checked facts.

The script fetches each card in ``ground_truth.json`` from Hugging Face (README,
config.json, and Hub metadata), runs URSA's deterministic extractor, and
compares the result field by field. Cards change over time, so a falling score
can mean the card changed; inspect the printed details before tuning.

Usage:
    uv run python benchmark/score_cards.py
    uv run python benchmark/score_cards.py --set B --details
"""

import argparse
import json
from pathlib import Path

from ursa.model_cards import extract_model_card
from ursa.model_cards.sources import fetch_hf_document

FIELDS = (
    "license",
    "commercial_use",
    "parameters",
    "domains",
    "tasks",
    "inputs",
    "outputs",
    "time_step",
    "horizon",
    "spatial",
    "spatial_dims",
    "how_to_run",
    "quality_flags",
)
ALIASES = {"heliophysics": "space_physics"}


def scalar(expected, got, *, close=False):
    if expected is None:
        return "correct" if got is None else "wrong"
    if got is None:
        return "missed"
    if close:
        return "correct" if abs(got - expected) / expected <= 0.15 else "wrong"
    return "correct" if str(got).lower() == str(expected).lower() else "wrong"


def concepts(primary, acceptable, got):
    primary = [ALIASES.get(p, p) for p in primary]
    allowed = set(primary) | {ALIASES.get(a, a) for a in acceptable}
    if not all(g in allowed for g in got):
        return "wrong"
    if not primary:
        return "correct"
    return "correct" if any(p in got for p in primary) else "missed"


def contains(expected, text, forbid=()):
    if any(f in text for f in forbid):
        return "wrong"
    if not expected:
        return "n/a" if not forbid else "correct"
    return "correct" if all(e in text for e in expected) else "missed"


def score(card, gt):
    inputs = json.dumps([s.model_dump() for s in card.inputs]).lower()
    outputs = json.dumps([s.model_dump() for s in card.outputs]).lower()
    runnable = any(
        i.kind in {"mcp_tool", "cli"} or i.loader or i.snippet
        for i in card.interfaces
    )
    flags = gt["issues_expect"]
    return {
        "license": scalar(gt["license"], card.license.spdx),
        "commercial_use": scalar(
            gt["commercial_use"], card.license.commercial_use
        ),
        "parameters": scalar(
            gt["parameter_count"], card.architecture.parameter_count, close=True
        ),
        "domains": concepts(
            gt["domains_primary"], gt["domains_acceptable"], card.domains
        ),
        "tasks": concepts(
            gt["tasks_primary"], gt["tasks_acceptable"], card.tasks
        ),
        "inputs": contains(gt["input_expect"], inputs),
        "outputs": contains(
            gt["output_expect"], outputs, gt.get("output_forbid", ())
        ),
        "time_step": scalar(gt["time_step"], card.temporal.time_step),
        "horizon": scalar(gt["horizon"], card.temporal.horizon),
        "spatial": contains(
            gt["spatial_expect"], " ".join(card.spatial.resolution).lower()
        ),
        "spatial_dims": scalar(gt["n_spatial_dims"], card.spatial.n_dims),
        "how_to_run": (
            "correct"
            if runnable == gt["has_code"]
            else "missed"
            if gt["has_code"]
            else "wrong"
        ),
        "quality_flags": (
            "n/a"
            if not flags
            else "correct"
            if set(flags) <= card.issue_codes()
            else "missed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", choices=("A", "B"), help="score one set only")
    parser.add_argument("--details", action="store_true", help="print misses")
    args = parser.parse_args()
    truth = json.loads(
        (Path(__file__).parent / "ground_truth.json").read_text(
            encoding="utf-8"
        )
    )
    totals = {
        field: {"correct": 0, "missed": 0, "wrong": 0} for field in FIELDS
    }
    symbol = {"correct": "+", "missed": ".", "wrong": "x", "n/a": " "}
    print(f"{'model':52s} " + " ".join(f[:4] for f in FIELDS))
    for repo, gt in truth.items():
        if repo.startswith("_") or (args.set and gt["set"] != args.set):
            continue
        card = extract_model_card(fetch_hf_document(repo))
        result = score(card, gt)
        print(
            f"{repo[:52]:52s} "
            + " ".join(f"{symbol[result[f]]:>4s}" for f in FIELDS)
        )
        for field, outcome in result.items():
            if outcome != "n/a":
                totals[field][outcome] += 1
        if args.details:
            for field, outcome in result.items():
                if outcome in {"missed", "wrong"}:
                    print(f"    {field}: {outcome}")
    print()
    correct = sum(t["correct"] for t in totals.values())
    count = sum(sum(t.values()) for t in totals.values())
    for field, t in totals.items():
        n = sum(t.values())
        print(
            f"{field:15s} {t['correct']:3d}/{n:<3d} missed {t['missed']:2d} wrong {t['wrong']:2d}"
        )
    print(
        f"\nOverall: {correct}/{count} correct ({100 * correct / count:.1f}%)"
    )


if __name__ == "__main__":
    main()
