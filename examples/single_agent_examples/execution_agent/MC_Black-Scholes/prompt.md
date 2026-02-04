# Execution plan (follow exactly, using tools)

You are a tool-using execution agent. You must execute the steps below. **Do not stop after writing a plan.**

## Constraints (hard)
- No internet / no web searches.
- No installing packages.
- No interactive prompts / no user input (`input()` is forbidden).
- Fixed random seed: **1337**.
- Runtime target: **< 10 seconds**.
- Do not create/modify any files outside this example directory.
- The only non-output file you may create/modify is: `mc_option_convergence.py`.
- All run artifacts (CSV/MD/PNG/logs/db/etc.) must be written **only under the chosen outputs directory** (see `outputs_dir` below).
- Do **not** print to the terminal when quiet mode is enabled.
- Avoid Rich animations/control characters when stdout is not a TTY.

## outputs_dir (IMPORTANT)
- Default output directory is `./outputs/`.
- If the prompt contains a line like:
  - `[Runner hint] outputs_dir=...`
  then treat that as the outputs directory (relative to the workspace) and write artifacts there.
- Otherwise, use `./outputs/`.

## Fixed parameters (do not change)
- `S0 = 100.0`
- `K = 100.0`
- `r = 0.05`
- `sigma = 0.2`
- `T = 1.0`
- `seed = 1337`
- `n_paths_list = [1000, 5000, 20000, 100000]`

## Output contract (must match exactly)
Write under `<outputs_dir>/`:

1) `results.csv` (non-empty) with columns exactly:
   `n_paths, mc_price, mc_stderr, ci95_low, ci95_high, bs_price, abs_error, runtime_seconds`
   - Must have 4–8 rows.
   - Must include at least the rows for `n_paths` in `[1000, 5000, 20000, 100000]`.
   - Rows must be sorted by `n_paths` ascending.
   - Numeric sanity:
     - `mc_stderr >= 0`
     - `runtime_seconds >= 0`
     - `ci95_low <= ci95_high`

2) `report.md` (non-empty) containing (case-sensitive markers required):
   - a **Parameters** section (must contain the word `Parameters`)
   - the Black–Scholes reference (must contain `Black–Scholes` or `Black-Scholes`)
   - a markdown table summarizing the rows of `results.csv`
   - a Summary section with heading line exactly: `## Summary`
   - an exact rerun line that contains exactly:
     - `Exact rerun command: python mc_option_convergence.py`
     (you may append flags after that, but the substring above must appear verbatim)

3) `plot.png` (ONLY if required):
   - The runner will have produced `<outputs_dir>/preflight.json`.
   - If that file indicates `optional.plotting_enabled == true`, then you **must** create a non-empty `plot.png`.
   - If plotting is not enabled, do not create an empty plot.
   - If you create a plot, force a headless backend: `matplotlib.use('Agg')` before importing `pyplot`.

## UX / logging contract for the generated script (mc_option_convergence.py)
The generated `mc_option_convergence.py` must be runnable by humans, with **informative progress output**, but must remain CI-safe.

Implement CLI flags:
- `--quiet` : no terminal output at all.
- `--log-file PATH` : optional file logging (Python `logging`).
- `--output-dir DIR` : outputs directory (default `outputs`).

TTY + Rich behavior:
- Use Rich **optionally**:
  - `try/except ImportError` around Rich imports.
  - Use Rich panels/status/progress only if stdout is a TTY **and** not quiet.
- If stdout is not a TTY, do not use spinners/progress bars; be quiet by default.
- If `--quiet` is set, do not print anything (no Rich, no plain prints).

Consistency hint:
- Prefer importing and using the local helper `ux.py` (in the same directory) so behavior matches `run_example.py` and `validate_outputs.py`.

## Steps to execute (do these with tool calls)

1) **Create** `mc_option_convergence.py` (use `write_code`) implementing:
   - Black–Scholes call price (normal CDF via `math.erf`).
   - Monte Carlo discounted payoff estimator with streaming mean/variance (Welford).
   - Timing for each `n_paths`.
   - Writes `<outputs_dir>/results.csv` and `<outputs_dir>/report.md` exactly per contract.
   - Reads `<outputs_dir>/preflight.json` (if present) to decide whether a plot is required; if required and matplotlib available, write `<outputs_dir>/plot.png`.
   - Implements the UX/logging contract above.

2) **Run** the script (use `run_command`):
   - `python mc_option_convergence.py` (optionally with `--output-dir` if `outputs_dir` is not `outputs`).

3) **Verify** artifacts exist and are non-empty (use `run_command`):
   - `test -s <outputs_dir>/results.csv`
   - `test -s <outputs_dir>/report.md`
   - If preflight says plotting_enabled=true: `test -s <outputs_dir>/plot.png`

4) **Read** the first ~30 lines of `<outputs_dir>/report.md` (use `read_file`).

## Output requirements (for your final chat response)
Your final response must be non-empty and include:
- what files you created/ran
- which artifacts exist under `<outputs_dir>/`
- the first ~30 lines of `<outputs_dir>/report.md`

## IMPORTANT
Start immediately with a **tool call** to `write_code` to create `mc_option_convergence.py`.
