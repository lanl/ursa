# URSA Metrics CLI — Plotting & Aggregation Guide

This guide covers how to use `metrics_cli.py` to generate per-run, per-thread, and cross-thread (SUPER) charts from Telemetry JSON. It includes the new model- and agent-level aggregations and the interactive timeline.

---

## Quickstart

```bash
# Generate thread-level + per-run charts for every thread under a directory
python ursa/scripts/metrics_cli.py --dir /path/to/metrics --chart all
```

```bash
# Walk all subdirectories, run `all` in each, then build a SUPER rollup at the root
python ursa/scripts/metrics_cli.py --dir /path/to/workspaces --chart all-recursive
```

```bash
# Single JSON: make a time lollipop, token totals, KDE, or tokens/sec
python ursa/scripts/metrics_cli.py path/to/agent_metrics.json --chart lollipop
python ursa/scripts/metrics_cli.py path/to/agent_metrics.json --chart tokens-bar
python ursa/scripts/metrics_cli.py path/to/agent_metrics.json --chart tokens-kde
python ursa/scripts/metrics_cli.py path/to/agent_metrics.json --chart tokens-rate
```

---

## What the CLI reads & what it makes

**Input (per file):**
- `context.agent`, `context.thread_id`, `context.run_id`, `context.started_at`, `context.ended_at`
- `tables.llm[]`, `tables.tool[]`, `tables.runnable[]`
- `llm_events[][].metrics.usage_rollup` (for token counts & samples)

**Outputs (PNG/HTML), depending on mode:**
- Time lollipop: `*_lollipop.png` (or `thread_<id>_lollipop.png`, `super_lollipop.png`)
- Token totals bar: `*_tokens_bar.png`
- Token KDE overlay: `*_tokens_kde.png`
- Tokens per second (two baselines): `*_tokens_rate.png`
- Interactive timeline: `thread_<id>_timeline.html`
- SUPER by-model: `super_tokens_bar_by_model.png`, `super_tokens_rate_by_model.png`
- Agent rollups (thread or SUPER):  
  - tokens: `thread_<id>_agents_tokens.png`, `super_agents_tokens.png`  
  - tokens/sec: `thread_<id>_agents_tps.png`, `super_agents_tps.png`

> Filenames for per-run charts are derived from the JSON path:  
> `path/to/run.json` → `path/to/run_breakdown_<chart>.png`.

---

## Modes

### 1) Single JSON (targeted charts)

```bash
python ursa/scripts/metrics_cli.py path/to/run.json --chart lollipop
python ursa/scripts/metrics_cli.py path/to/run.json --chart tokens-bar
python ursa/scripts/metrics_cli.py path/to/run.json --chart tokens-kde
python ursa/scripts/metrics_cli.py path/to/run.json --chart tokens-rate
```

Use `--title` and `--out` to customize:

```bash
python ursa/scripts/metrics_cli.py run.json \
  --chart tokens-rate \
  --title "Model TPS (build #814)" \
  --out out/my_rate.png
```

---

### 2) Thread-level (aggregate all runs of one `thread_id`)

List threads the CLI can see in a directory:

```bash
python ursa/scripts/metrics_cli.py --dir /metrics --list-threads
```

Generate a specific thread’s charts:

```bash
# Time lollipop
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-lollipop --thread <thread_id>

# Token totals / KDE / TPS
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-tokens-bar  --thread <thread_id>
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-tokens-kde  --thread <thread_id> --log-x
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-tokens-rate --thread <thread_id>

# Interactive timeline (HTML); y-axis grouped by agent (default) or one row per run
python ursa/scripts/metrics_cli.py --dir /metrics --chart timeline-html --thread <thread_id> --group-by agent
python ursa/scripts/metrics_cli.py --dir /metrics --chart timeline-html --thread <thread_id> --group-by run
```

**Agent rollups for a thread:**

```bash
# Tokens stacked by agent
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-agents-tokens --thread <thread_id>

# Tokens/sec by agent (two baselines)
python ursa/scripts/metrics_cli.py --dir /metrics --chart thread-agents-tps --thread <thread_id>
```

> Thread-level TPS uses two denominators:  
> - **per LLM-sec (sum):** sum of `tables.llm[].total_s` across all runs in the thread  
> - **per thread-sec:** `max(ended_at) - min(started_at)` across the thread’s runs

---

### 3) “All” (non-recursive) for a directory

Run **all** thread-level charts and per-run charts inside a directory:

```bash
python ursa/scripts/metrics_cli.py --dir /metrics --chart all
```

What it produces:
- For each thread: lollipop, tokens-bar, tokens-kde, tokens-rate, timeline HTML
- For each JSON: lollipop, tokens-bar, tokens-kde, tokens-rate

(Use `--log-x` to log-scale the lollipops & KDE.)

---

### 4) “All-recursive” + SUPER (cross-thread rollups)

Walk subdirectories, run `all` in each, then build a rollup at the root:

```bash
python ursa/scripts/metrics_cli.py --dir /workspaces --chart all-recursive
```

SUPER artifacts at `--dir`:
- `super_lollipop.png` (time by component across all threads)
- `super_tokens_bar.png` & `super_tokens_kde.png` (totals & distribution)
- `super_tokens_rate.png` (TPS using Σ thread windows & Σ LLM-sec)
- **By model:** `super_tokens_bar_by_model.png`, `super_tokens_rate_by_model.png`
- **By agent:** `super_agents_tokens.png`, `super_agents_tps.png`  
  *(Use the explicit charts below if you want just the agent rollups without re-running everything.)*

Build *only* the SUPER agent rollups for a directory you’ve already processed:

```bash
python ursa/scripts/metrics_cli.py --dir /workspaces --chart super-agents-tokens
python ursa/scripts/metrics_cli.py --dir /workspaces --chart super-agents-tps
```

> In SUPER charts, the bottom footer **does not** show a single start→end “window”, since different threads can overlap or run on different machines. Where relevant, the footer shows **Σ thread windows** and **Σ LLM-active seconds** instead.

---

## Understanding the denominators (for TPS)

- **LLM-active seconds (sum)**  
  From `tables.llm[].total_s` (or via event intervals for single runs). If multiple LLM calls overlap, the sum can exceed the wall window; this indicates parallelism.

- **Thread window seconds**  
  For a thread: `max(ended_at) - min(started_at)`.  
  For SUPER: **sum** of per-thread windows (not a single global wall window).

The TPS chart shows **both** denominators side-by-side to make parallelism visible.

---

## Useful options

| Flag | Meaning | Notes |
|---|---|---|
| `--dir PATH` | Directory to scan for metrics JSONs. | Required for `all`, thread-level, and SUPER modes. |
| `--chart` | Which artifact(s) to generate. | See lists above; default is `all`. |
| `--thread ID` | Limit to one thread for thread-level charts. | Use with `--chart thread-*` or `timeline-html`. |
| `--list-threads` | Print discovered threads in `--dir`. | Great to copy/paste a `--thread` ID. |
| `--group-llm` | Group all LLM rows into `llm:total` in time charts. | Affects lollipop/pie/bar (time). |
| `--group-by {agent,run}` | Timeline y-axis grouping. | `agent` is compact; `run` gives one lane per run. |
| `--log-x` | Log-scale for lollipop & KDE. | Helpful when components vary by orders of magnitude. |
| `--min-label-pct FLOAT` | Hide dot labels below this percent in lollipop. | Default `0.0` (show all). |
| `--title TEXT` | Custom chart title. | For targeted modes. |
| `--out PATH` | Custom output file path. | For targeted modes. |
| `--check` | Print attribution totals for a single JSON and exit. | Verifies `llm+tool+other ≈ graph:graph`. |
| `--epsilon FLOAT` | Tolerance for `--check`. | Default `0.050` seconds. |

---

## Examples (copy/paste)

```bash
# See what threads are in a directory
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun --list-threads
```

```bash
# Thread-level bundle for a single thread (PNG + HTML)
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun \
  --chart thread-tokens-rate --thread modsim_predict_final_mild-orange
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun \
  --chart timeline-html --thread modsim_predict_final_mild-orange --group-by agent
```

```bash
# Agent breakdowns for one thread (stacked tokens + TPS)
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun \
  --chart thread-agents-tokens --thread <thread_id>
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun \
  --chart thread-agents-tps --thread <thread_id>
```

```bash
# Directory-wide (non-recursive) batch
python ursa/scripts/metrics_cli.py --dir ./workspaces/myrun --chart all
```

```bash
# Recursive batch + SUPER rollups at the root
python ursa/scripts/metrics_cli.py --dir ./workspaces --chart all-recursive
```

```bash
# Only the SUPER agent charts (when you already have per-thread results)
python ursa/scripts/metrics_cli.py --dir ./workspaces --chart super-agents-tokens
python ursa/scripts/metrics_cli.py --dir ./workspaces --chart super-agents-tps
```

```bash
# Single JSON – compare denominators in TPS
python ursa/scripts/metrics_cli.py ./workspaces/t1/run_0007.json --chart tokens-rate --title "step 7 TPS"
```

---

## Output naming & where to find things

**Per run (single JSON):**
```
<path>/run_breakdown_lollipop.png
<path>/run_breakdown_tokens_bar.png
<path>/run_breakdown_tokens_kde.png
<path>/run_breakdown_tokens_rate.png
```

**Thread-level (in --dir):**
```
thread_<thread_id>_lollipop.png
thread_<thread_id>_tokens_bar.png
thread_<thread_id>_tokens_kde.png
thread_<thread_id>_tokens_rate.png
thread_<thread_id>_timeline.html
thread_<thread_id>_agents_tokens.png
thread_<thread_id>_agents_tps.png
```

**SUPER (at the root --dir for all-recursive):**
```
super_lollipop.png
super_tokens_bar.png
super_tokens_kde.png
super_tokens_rate.png
super_tokens_bar_by_model.png
super_tokens_rate_by_model.png
super_agents_tokens.png
super_agents_tps.png
```

---

## Tips

- Use `--log-x` for lollipop & KDE when a few components dominate.
- Use `--group-llm` to collapse many LLM rows into a single “llm:total” bar for readability.
- The interactive timeline (`timeline-html`) is ideal for human inspection of overlaps; set `--group-by run` to see one lane per run.
- In SUPER TPS, the footer reports **Σ thread windows** and **Σ LLM-active** seconds rather than a single start→end time.

---

## Troubleshooting

**No thread IDs found**  
- Ensure `--dir` points at a directory containing Telemetry JSON files.
- JSON must include `context.thread_id`, `context.agent`, `context.run_id`, `context.started_at`, `context.ended_at`.

**Tokens charts look empty (all zeros)**  
- Check that `llm_events[].metrics.usage_rollup` has `input_tokens` / `output_tokens` (or `prompt_tokens` / `completion_tokens`) fields.
- If a provider omits `total_tokens`, the CLI computes `max(total, input+output)`.

**TPS “LLM sum exceeds window → parallel LLM work”**  
- Expected when multiple LLM calls overlap. The note is helpful, not an error.

**Attribution check fails (`--check`)**  
- The CLI prints: `graph:graph`, `LLM total_s`, `Tool total_s`, `Unattributed`, and any overage.  
  Small residuals under `--epsilon` are tolerated.

**Agent plots still zero**  
- The agent aggregators depend on `context.agent` and `llm_events` being present per run. If your pipeline writes tokens only at the model level without `llm_events`, the totals per agent will be zero.

---

## Version notes

- SUPER “by model” and “by agent” charts are additive across all discovered threads.  
- SUPER footers avoid a single run-window timestamp (threads can overlap and run elsewhere); they report **sums** instead.

---

Happy charting!
