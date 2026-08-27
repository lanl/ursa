# Choose experiments for risk buy-down

Give URSA two small CSV files, ask an execution agent to rank candidate
experiments under cost and schedule constraints, and then audit its selected
campaign. This guided exercise emphasizes transparent calculations and
reviewable artifacts rather than treating an agent's recommendation as a real
qualification or safety decision.

You can run the same exercise in the TUI or dashboard. In either interface,
make this directory the agent's workspace so the relative input paths resolve
and the requested outputs stay beside the example.

## Inspect the inputs

Open [the design-risk table](design_risks.csv) and
[the candidate-experiment table](candidate_experiments.csv) before launching an
agent. The files use these schemas:

```text
risk_id,risk_name,risk_area,initial_probability,impact,uncertainty,description
```

```text
experiment_id,experiment_name,cost_kusd,duration_days,risk_targets,expected_uncertainty_reduction,confidence,description
```

The `risk_targets` field contains semicolon-separated risk IDs. All values and
scoring rules are synthetic. Do not use the resulting ranking as evidence of
readiness, qualification, or safety.

## Prepare the example

Open a terminal in the URSA repository, enter this directory, and install its
dependencies. The dashboard extra is included so either interface is available.

=== "macOS/Linux"

    ```bash
    cd examples/single_agent_examples/execution_agent/risk_buydown
    uv sync
    export OPENAI_API_KEY="..."
    ```

=== "Windows PowerShell"

    ```powershell
    Set-Location examples\single_agent_examples\execution_agent\risk_buydown
    uv sync
    $env:OPENAI_API_KEY = "..."
    ```

OpenAI works with URSA's built-in defaults and needs no config file. If you use
another endpoint or model, configure it as described in the
[configuration guide](../../../../configuration/index.md).

## Choose an interface

=== "TUI"

    Start URSA from this directory:

    === "macOS/Linux"

        ```bash
        uv run ursa
        ```

    === "Windows PowerShell"

        ```powershell
        uv run ursa
        ```

    Keep this example directory as the workspace. Paste the prompt in the next
    section with its leading `#execute`, which selects the execution agent. See
    the [TUI guide](../../../../getting-started/tui.md) for agent selection and
    application commands.

=== "Dashboard"

    Start the dashboard from this directory:

    === "macOS/Linux"

        ```bash
        uv run ursa-dashboard
        ```

    === "Windows PowerShell"

        ```powershell
        uv run ursa-dashboard
        ```

    Open the printed address, normally `http://127.0.0.1:8080`. Create an
    **Execution Agent** session and select this example directory as its
    workspace. Paste the prompt below without the leading `#execute`, because
    the session already selects the agent. See the
    [dashboard guide](../../../../getting-started/dashboard.md) for credential
    storage and workspace behavior.

## Ask the agent to rank the experiments

Paste this prompt into your chosen interface. Keep `#execute` when using the
TUI; remove only that prefix in an Execution Agent dashboard session.

```text
#execute Read ./design_risks.csv and ./candidate_experiments.csv. Select an
experiment campaign that maximizes expected risk reduction subject to total
cost <= 150 kUSD, total duration <= 45 days, and at most 4 experiments.

For each risk use:
initial_risk_score = initial_probability * impact * uncertainty

For each experiment use:
expected_risk_reduction =
  sum(initial_risk_score for targeted risks)
  * expected_uncertainty_reduction
  * confidence

Search the feasible experiment combinations. Keep the calculation simple and
explainable. Write only under ./risk_buydown_outputs/ and create:
- experiment_rankings.csv
- selected_campaign.csv
- risk_before_after.csv
- experiment_value_scatter.png
- risk_before_after.png
- risk_buydown_reasoning.txt
- risk_buydown_recommendation.txt

Validate the input columns, do not overwrite inputs, and do not present this
toy analysis as a real qualification, readiness, or safety determination.
```

Review each proposed tool action before approving it. The execution agent can
write files and run commands in its workspace; use a disposable copy when you
adapt this exercise to data you cannot replace. The
[ExecutionAgent guide](../../../../agents/execution.md) describes its tools and
workspace behavior, and
[Sandboxing and information control](../../../../best-practices/sandboxing.md)
explains stronger isolation options.

## Audit the recommendation

Open `risk_buydown_outputs/` after the agent finishes. Do not stop at the
recommendation text. Check the work in this order:

1. Confirm that both source CSVs are unchanged.
2. Verify that `experiment_rankings.csv` contains every candidate and exposes
   the score components used for ranking.
3. Recompute the selected campaign's total cost and duration from
   `candidate_experiments.csv`; confirm cost is at most 150 kUSD, duration is at
   most 45 days, and no more than four experiments were selected.
4. Recompute the initial risk scores and expected reductions from the formulas
   in the prompt.
5. Confirm that `risk_before_after.csv` accounts for every risk, including
   risks untouched by the selected campaign.
6. Inspect both plots and make sure selected experiments are distinguishable
   from alternatives.
7. Read `risk_buydown_reasoning.txt` and
   `risk_buydown_recommendation.txt`; require them to state remaining risks,
   assumptions, and the toy nature of the exercise.

If an artifact is missing or a constraint fails, ask the same execution-agent
session to inspect its work and correct the output rather than silently
accepting a partial result. Because an LLM chooses the implementation, exact
rankings and filenames' internal formats may vary even though the requested
files and constraints are fixed.
