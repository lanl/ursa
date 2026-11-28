"""Ask ModelCardAgent the curated inference tasks and score its answers.

Each task in ``tasks.yaml`` is a question a scientist asks about a data
file, with the answer that the model cards support. The script connects the
agent to a Nomad server, asks the questions one at a time, checks each
answer, and writes a summary, the agent's reports, and figures to
``results/``.

Usage, from ``examples/model_cards``:
    uv run python curated_tasks/run_tasks.py --offline
    uv run python curated_tasks/run_tasks.py --model openai:gpt-5-mini
    uv run python curated_tasks/run_tasks.py --offline --only gap run_bp
    uv run python curated_tasks/run_tasks.py --model anthropic:claude-haiku-5-5 \
        --only gap
    uv run python curated_tasks/run_tasks.py --offline \\
        --heat-file ~/nomad/docs/guides/angled_half_res.h5
"""

import argparse
import asyncio
import json
import re
import shutil
from pathlib import Path

import yaml
from langchain_core.callbacks import (
    BaseCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from ursa.agents import ModelCardAgent
from ursa.model_cards.data import heavy_atoms
from ursa.model_cards.matching import filter_catalog
from ursa.model_cards.running import flatten_result
from ursa.util.mcp import start_mcp_client

HERE = Path(__file__).parent
USABLE = {"inference", "inference_after_conversion"}
PATHWAY_STYLE = {
    # pathway: (fill, cell text, text color, legend label)
    "inference": ("#0000FF", "Infer", "white", "Inference as is"),
    "inference_after_conversion": (
        "#00A000",
        "Convert",
        "white",
        "Inference after conversion",
    ),
    "fine_tune": ("#FFFF00", "Tune", "black", "Fine-tune"),
    "unknown": ("#FF0000", "?", "white", "Not enough information"),
    "not_suitable": ("#FFFFFF", "No", "black", "Not suitable"),
}


class RulesOnly(BaseChatModel):
    """Stand-in for an LLM that makes the agent use its rule-based path alone.

    Every structured call returns the empty schema, so the agent reads the
    request, the cards, and the data with patterns only. The scores are then
    reproducible and need no API key.
    """

    @property
    def _llm_type(self) -> str:
        return "rules-only"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=""))]
        )

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, *, include_raw=False, **kwargs):
        def empty(_):
            parsed = schema()
            if include_raw:
                return {
                    "raw": AIMessage(content=""),
                    "parsed": parsed,
                    "parsing_error": None,
                }
            return parsed

        return RunnableLambda(empty)


class Progress(BaseCallbackHandler):
    """Print the agent's steps as they happen, so a slow LLM call shows."""

    def on_custom_event(self, name, data, **kwargs):
        message = data.get("message") if isinstance(data, dict) else None
        print(f"    - {message or name}", flush=True)


def write_heat_layout(path: Path) -> None:
    """Write a zero-valued file with the layout of Nomad's HEAT-PLI demo data.

    The file carries the facts that decide the pathway (The Well layout, the
    560 x 200 grid, and the av_density field), not physics. Pass --heat-file
    to use a real HEAT-PLI state instead.
    """
    import h5py
    import numpy as np

    with h5py.File(path, "w") as handle:
        handle.attrs["dataset_name"] = "heat_ic"
        handle.attrs["grid_type"] = "cartesian"
        handle.attrs["n_spatial_dims"] = 2
        dims = handle.create_group("dimensions")
        dims.attrs["spatial_dims"] = np.array(
            ["z", "r"], dtype=h5py.string_dtype()
        )
        dims["time"] = np.zeros(1)
        fields = handle.create_group("t0_fields")
        for name in ("av_density", "av_energy", "av_pressure"):
            fields[name] = np.zeros((1, 1, 560, 200), dtype="float32")


def prepare_workspace(out: Path, heat_file: str | None) -> Path:
    workspace = out / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    for path in (HERE / "data").iterdir():
        shutil.copy(path, workspace / path.name)
    if heat_file:
        shutil.copy(Path(heat_file).expanduser(), workspace / "heat_ic.h5")
    else:
        write_heat_layout(workspace / "heat_ic.h5")
    return workspace


def display_name(model_id: str, names: dict[str, str]) -> str:
    return next(
        (label for key, label in names.items() if key in model_id), model_id
    )


def run_rows(state, workspace: Path) -> list[dict]:
    """Rows of the batch run the agent made in this turn, if any."""
    match = re.search(r"`(fm_runs/[^`]+\.json)`", state.get("run_result") or "")
    if not match:
        return []
    return json.loads((workspace / match.group(1)).read_text(encoding="utf-8"))[
        "results"
    ]


def evaluate(task, state, agent, rows) -> tuple[str, list[dict], str | None]:
    """The answer in a few words, each expectation checked, and the model named."""
    expect = task.get("expect", {})
    assessments = state.get("assessments") or []
    plan = state.get("run_plan")
    result = state.get("run_result") or ""
    ran = result.startswith("Ran ") or result.startswith("Saved to")
    report = agent.format_result(state)
    usable = [a for a in assessments if a.pathway in USABLE]
    if plan is not None and ran:
        chosen = plan.model_id
    elif assessments:
        chosen = (usable or assessments)[0].model_id
    else:
        chosen = None
    best = next((a for a in assessments if a.model_id == chosen), None)
    req = state.get("requirements")
    matches, maybes = (
        filter_catalog(
            req,
            agent.catalog.all(),
            state.get("request", ""),
            domains=agent.domains,
            tasks=agent.tasks,
        )
        if req is not None
        else ([], [])
    )
    got = {
        "catalog_size": len(agent.catalog),
        "matches": len(matches),
        "might_match": len(maybes),
        "usable": len(usable),
        "best": chosen,
        "pathway": best.pathway if best else None,
        "caution": bool(
            best
            and any(
                c.aspect == "training range" and c.status == "unknown"
                for c in best.checks
            )
        ),
        "base_model": next(
            (
                m.group(1)
                for m in [re.search(r"fine-tuned from `([^`]+)`", report)]
                if m
            ),
            None,
        ),
        "runs": ran,
        "calls": len(rows),
    }
    checks = []
    for key, wanted in expect.items():
        value = got[key]
        ok = (wanted in (value or "")) if key == "best" else value == wanted
        checks.append({
            "check": key,
            "expected": wanted,
            "got": value,
            "ok": bool(ok),
        })
    if "list" in state.get("intents", []) and not assessments:
        answer = (
            f"{got['matches']} match, {got['might_match']} might"
            if req is not None and (req.cpu_only or req.inputs or req.domains)
            else f"{got['catalog_size']} models"
        )
    elif ran:
        answer = f"Ran {len(rows) or 1} call(s)"
    elif best is not None:
        answer = best.label
    else:
        answer = "No model checked"
    return answer, checks, chosen


def score_reference(task, rows, workspace: Path) -> dict | None:
    """Compare run outputs with the measured values in the task's data file."""
    reference = task.get("reference")
    if not reference or not rows:
        return None
    import pandas as pd

    table = pd.read_csv(workspace / task["data"])
    measured = dict(zip(table["smiles"], table[reference["column"]]))
    names = dict(zip(table["smiles"], table["name"]))
    points = []
    for row in rows:
        smiles = next(iter(row["arguments"].values()))
        values = flatten_result(row.get("result"))
        key = next(
            (k for k in values if k.split(" [")[0] == reference["output"]),
            None,
        )
        if key is not None and smiles in measured:
            points.append((names[smiles], measured[smiles], values[key]))
    errors = [abs(predicted - true) for _, true, predicted in points]
    worst = max(zip(errors, points)) if points else None
    return {
        "quantity": reference["quantity"],
        "units": reference["units"],
        "points": points,
        "mae": sum(errors) / len(errors) if errors else None,
        "worst": {"name": worst[1][0], "error": worst[0]} if worst else None,
    }


# Figures ----------------------------------------------------------------------


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        # Subscripts such as BF$_3$ come from mathtext, set in Arial too.
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial",
        "mathtext.cal": "Arial",
    })
    return plt


def _place_labels(ax, points, fontsize=10) -> None:
    """Label each point where the label overlaps no other label or point.

    Candidate spots are right, left, above, and below the point; the first
    spot inside the axes that collides with nothing placed so far wins.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    taken = [
        ax.transData.transform((x, y)) for _, x, y in points
    ]  # point centers in pixels
    boxes = []
    frame = ax.get_window_extent(renderer)
    for name, x, y in points:
        best = None
        for dx, dy, ha, va in (
            (7, 0, "left", "center"),
            (-7, 0, "right", "center"),
            (0, 8, "center", "bottom"),
            (0, -8, "center", "top"),
        ):
            label = ax.annotate(
                name,
                (x, y),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                va=va,
                fontsize=fontsize,
                # A white ground keeps reference lines from crossing the text.
                bbox={
                    "boxstyle": "square,pad=0.05",
                    "fc": "white",
                    "ec": "none",
                },
            )
            box = label.get_window_extent(renderer).expanded(1.05, 1.15)
            inside = (
                box.x0 >= frame.x0
                and box.x1 <= frame.x1
                and box.y0 >= frame.y0
                and box.y1 <= frame.y1
            )
            clear = not any(box.overlaps(other) for other in boxes) and not any(
                box.contains(px, py) for px, py in taken
            )
            if inside and clear:
                best = (label, box)
                break
            label.remove()
        if best is None:
            label = ax.annotate(
                name,
                (x, y),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=fontsize,
            )
            best = (label, label.get_window_extent(renderer))
        boxes.append(best[1])


def plot_pathways(rows, models, names, path: Path) -> None:
    """Task-by-model grid of pathways; a bold frame marks the model named."""
    from matplotlib.patches import Patch, Rectangle

    plt = _style()
    fig, ax = plt.subplots(
        figsize=(0.95 * len(models) + 1.9, 0.42 * len(rows) + 1.0)
    )
    for i, (label, pathways, chosen) in enumerate(rows):
        for j, model in enumerate(models):
            pathway = pathways.get(model)
            fill, text, color, _ = PATHWAY_STYLE.get(
                pathway, ("#FFFFFF", "", "black", "")
            )
            ax.add_patch(
                Rectangle(
                    (j, i), 1, 1, facecolor=fill, edgecolor="black", lw=0.8
                )
            )
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
            if model == chosen:
                ax.add_patch(
                    Rectangle(
                        (j + 0.06, i + 0.06),
                        0.88,
                        0.88,
                        fill=False,
                        edgecolor="black",
                        lw=2.6,
                    )
                )
    ax.set_xlim(0, len(models))
    ax.set_ylim(len(rows), 0)
    # Two-line model names ("MIST" over "boiling point") keep columns narrow.
    ax.set_xticks(
        [j + 0.5 for j in range(len(models))],
        [display_name(m, names).replace(" ", "\n", 1) for m in models],
        fontsize=10.5,
    )
    ax.xaxis.tick_top()
    ax.set_yticks([i + 0.5 for i in range(len(rows))], [r[0] for r in rows])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    shown = {p for _, pathways, _ in rows for p in pathways.values()}
    handles = [
        Patch(facecolor=fill, edgecolor="black", label=label)
        for pathway, (fill, _, _, label) in PATHWAY_STYLE.items()
        if pathway in shown
    ]
    if any(chosen for _, _, chosen in rows):
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                lw=2.6,
                label="Model the answer names",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_reference(score: dict, path: Path) -> None:
    """Predicted against measured values, with the line of perfect agreement."""
    plt = _style()
    names, measured, predicted = zip(*score["points"])
    lo = min(min(measured), min(predicted))
    hi = max(max(measured), max(predicted))
    pad = 0.08 * (hi - lo)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="black",
        ls="--",
        lw=1.2,
        label="Predicted = measured",
    )
    ax.scatter(
        measured,
        predicted,
        s=60,
        color="#0000FF",
        edgecolor="white",
        lw=1.0,
        zorder=3,
        label="Model prediction",
    )
    units = score["units"]
    ax.set_xlabel(f"Measured {score['quantity'].lower()} ({units})")
    ax.set_ylabel(f"Predicted {score['quantity'].lower()} ({units})")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.grid(color="#D0D0D0", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        frameon=False,
        title=f"Mean absolute error {score['mae']:.1f} {units}",
        title_fontsize=11,
        alignment="left",
    )
    _place_labels(
        ax, [(n[0].upper() + n[1:], x, y) for n, x, y in score["points"]]
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_outputs_in_range(
    rows, card, output: str, label: str, path: Path
) -> None:
    """One output per molecule, colored by whether the molecule is in range."""
    from matplotlib.patches import Patch

    plt = _style()
    domain = next(
        (r for r in card.training_scope.ranges if r.quantity == "heavy atoms"),
        None,
    )
    labels, values, inside = [], [], []
    for row in rows:
        smiles = next(iter(row["arguments"].values()))
        value = flatten_result(row.get("result")).get(output)
        if value is None:
            continue
        name = row.get("name") or smiles
        labels.append((name[0].upper() + name[1:]).replace("_", " "))
        values.append(value)
        inside.append(domain is None or domain.contains(heavy_atoms(smiles)))
    fig, ax = plt.subplots(figsize=(5.6, 0.36 * len(labels) + 1.0))
    ax.barh(
        labels,
        values,
        height=0.62,
        color=["#0000FF" if ok else "#FF0000" for ok in inside],
    )
    ax.invert_yaxis()
    ax.set_xlabel(f"Predicted {label} (units not stated by the card)")
    ax.grid(axis="x", color="#D0D0D0", lw=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if domain is not None:
        ax.legend(
            handles=[
                Patch(
                    facecolor="#0000FF",
                    label=f"Inside training range ({domain.text})",
                ),
                Patch(facecolor="#FF0000", label="Outside training range"),
            ],
            loc="lower left",
            bbox_to_anchor=(0.0, 1.01),
            frameon=False,
        )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Main -------------------------------------------------------------------------


def write_summary(
    results, scores, out: Path, mode: str, tokens: dict | None = None
) -> None:
    passed = sum(all(c["ok"] for c in r["checks"]) for r in results)
    lines = [
        "# Curated inference tasks",
        "",
        f"Mode: {mode}. {passed} of {len(results)} tasks passed every check.",
        "",
    ]
    llm_failed = [r for r in results if r.get("llm_error")]
    if llm_failed:
        # A failed LLM call leaves the rules to answer, so the run does not
        # test the LLM steps; say so rather than report an LLM result.
        lines += [
            f"Warning: the LLM call failed in {len(llm_failed)} of "
            f"{len(results)} tasks, so those answers come from the rules "
            f"alone. First error: {llm_failed[0]['llm_error'][:200]}",
            "",
        ]
    lines += [
        "| Task | Question | Answer | Checks passed |",
        "|---|---|---|---|",
    ]
    for r in results:
        ok = sum(c["ok"] for c in r["checks"])
        lines.append(
            f"| {r['label']} | {r['question']} | {r['answer']} | {ok} of {len(r['checks'])} |"
        )
    failed = [(r, c) for r in results for c in r["checks"] if not c["ok"]]
    if failed:
        lines += ["", "## Failed checks", ""]
        lines += [
            f"- {r['label']}: {c['check']} expected {c['expected']!r}, got {c['got']!r}."
            for r, c in failed
        ]
    for task_id, score in scores.items():
        if score and score["mae"] is not None:
            lines += [
                "",
                f"## {score['quantity']} against measurements ({task_id})",
                "",
                f"Mean absolute error {score['mae']:.1f} {score['units']} over "
                f"{len(score['points'])} molecules; the largest error is "
                f"{score['worst']['error']:.1f} {score['units']} ({score['worst']['name']}).",
            ]
    if tokens:
        lines += ["", "## LLM tokens", ""]
        lines += [
            f"- {model}: {used['input']:,} input and {used['output']:,} output tokens"
            for model, used in tokens.items()
        ]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(
            {"results": results, "scores": scores, "tokens": tokens or {}},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


async def main(args) -> None:
    spec = yaml.safe_load((HERE / "tasks.yaml").read_text(encoding="utf-8"))
    names = spec.get("models", {})
    out = Path(args.out)
    workspace = prepare_workspace(out, args.heat_file)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    if args.offline:
        llm, mode = RulesOnly(), "rules only (no LLM)"
    else:
        from langchain.chat_models import init_chat_model

        # Without limits, LangChain asks Claude for its maximum output and
        # waits on a call indefinitely; a structured answer needs far less.
        limits = (
            {"max_tokens": 16000, "timeout": 120}
            if args.model.startswith("anthropic:")
            else {}
        )
        llm = init_chat_model(args.model, **limits)
        mode = f"LLM {args.model}"
    agent = ModelCardAgent(
        llm=llm,
        workspace=workspace,
        catalog_dir=workspace / "model_cards",
        write_reports=False,
        refine_discovered=args.refine_cards,
    )
    client = start_mcp_client({
        "nomad": {"transport": "streamable-http", "url": args.nomad_url}
    })
    await agent.add_mcp_tools(client)

    results, scores, grid, runs = [], {}, [], {}
    # Counts the tokens every LLM call reports, so a run states its cost.
    with get_usage_metadata_callback() as usage:
        for task in spec["tasks"]:
            if args.only and task["id"] not in args.only:
                continue
            print(f">>> {task['id']}: {task['question']}", flush=True)
            # LLM calls take seconds each; show the steps while they run.
            progress = (
                {} if args.offline else {"config": {"callbacks": [Progress()]}}
            )
            state = await agent.ainvoke(task["question"], **progress)
            (out / "reports" / f"{task['id']}.md").write_text(
                agent.format_result(state) + "\n", encoding="utf-8"
            )
            rows = run_rows(state, workspace)
            answer, checks, chosen = evaluate(task, state, agent, rows)
            print(
                f"    {answer}; {sum(c['ok'] for c in checks)} of {len(checks)} checks"
            )
            if state.get("llm_error"):
                print(f"    LLM call failed: {state['llm_error'][:120]}")
            results.append({
                "id": task["id"],
                "label": task["label"],
                "question": task["question"],
                "answer": answer,
                "checks": checks,
                "llm_error": state.get("llm_error") or "",
            })
            scores[task["id"]] = score_reference(task, rows, workspace)
            assessments = state.get("assessments") or []
            if len(assessments) > 1:
                # Frame a model only when the answer names one ("use QM9"), not
                # when it lists choices or recommends a base model.
                named = chosen if "best" in task.get("expect", {}) else None
                grid.append((
                    task["label"],
                    {a.model_id: a.pathway for a in assessments},
                    named,
                ))
            if rows:
                runs[task["id"]] = (rows, state["run_plan"].model_id)

    tokens = {
        model: {"input": used["input_tokens"], "output": used["output_tokens"]}
        for model, used in usage.usage_metadata.items()
    }
    for model, used in tokens.items():
        print(
            f"Tokens used: {model}: {used['input']} input + {used['output']} output"
        )
    write_summary(results, scores, out, mode, tokens)
    if grid:
        models = sorted(
            {m for _, pathways, _ in grid for m in pathways},
            key=lambda m: display_name(m, names),
        )
        plot_pathways(grid, models, names, out / "fig_pathways.png")
    for task_id, score in scores.items():
        if score and score["points"]:
            plot_reference(score, out / f"fig_{task_id}.png")
    if "run_qm9" in runs:
        rows, model_id = runs["run_qm9"]
        plot_outputs_in_range(
            rows,
            agent.catalog.get(model_id),
            "gap",
            "HOMO-LUMO gap",
            out / "fig_run_qm9.png",
        )
    agent.close()
    print(f"\nWrote {out / 'summary.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--nomad-url", default="http://localhost:38217/mcp")
    parser.add_argument("--model", default="openai:gpt-5-mini")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="answer with the agent's rule-based path alone (no LLM, no API key)",
    )
    parser.add_argument(
        "--heat-file", help="a real HEAT-PLI state in The Well HDF5 layout"
    )
    parser.add_argument("--only", nargs="*", help="task ids to run")
    parser.add_argument(
        "--refine-cards",
        action="store_true",
        help="have the LLM review each card found on Nomad (one call per card)",
    )
    parser.add_argument("--out", default=str(HERE / "results"))
    asyncio.run(main(parser.parse_args()))
