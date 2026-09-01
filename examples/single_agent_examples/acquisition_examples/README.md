# Compare research sources with acquisition agents

Use this walkthrough to investigate one scientific question through three
different source collections. URSA's acquisition agents search for material,
cache the retrieved pages or papers, summarize each item, and synthesize a final
answer:

- `ArxivAgent` searches arXiv and downloads paper PDFs.
- `OSTIAgent` searches U.S. Department of Energy OSTI records.
- `WebSearchAgent` searches the open web with DDGS.

You will try the workflow from the TUI, the dashboard, and Python. Each entry
point uses the same acquisition machinery, but exposes it differently.

## What you will produce

The guided question asks how graph neural networks solve partial differential
equations, with an emphasis on shock hydrodynamics. Retrieved documents and
summaries are written beneath this directory when you run the Python example.
Treat the summaries as research leads: follow their source links and verify
important claims in the original documents.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- Internet access to arXiv, OSTI, and web-search results
- An OpenAI API key, or an equivalent provider in your
  [URSA configuration](../../../docs/configuration/index.md)

Run the following commands from this `acquisition_examples` directory.

## 1. Install the example environment

=== "macOS/Linux"

    ```bash
    uv sync
    export OPENAI_API_KEY="..."
    uv run ursa --help
    ```

=== "Windows PowerShell"

    ```powershell
    uv sync
    $env:OPENAI_API_KEY = "..."
    uv run ursa --help
    ```

The example uses the editable URSA checkout three directories above it and
includes dashboard support. If you use a non-OpenAI endpoint, configure it
before continuing and omit the `OPENAI_API_KEY` command. See
[models and inference providers](../../../docs/configuration/models.md).

## 2. Explore acquisition in the TUI

Start the terminal interface:

```bash
uv run ursa
```

Type `#` to open the agent picker. The TUI directly registers the arXiv and web
acquisition agents. Run these prompts one at a time:

```text
#arxiv Find papers about graph neural networks for partial differential
equations. Compare methods and benchmarks, emphasizing possible applications
to shock hydrodynamics, and cite the papers used.
```

```text
#web Find reliable sources about graph neural networks for partial differential
equations. Compare methods and benchmarks, emphasizing possible applications
to shock hydrodynamics, and cite the pages used.
```

Watch the activity cards as URSA searches, retrieves, and summarizes sources.

See the [TUI guide](../../../docs/getting-started/tui.md) for agent macros,
workspaces, and controls. The
[acquisition-agent overview](../../../docs/agents/acquisition/index.md) explains
the shared acquire-then-summarize graph and cached outputs.

## 3. Run the same research task in the dashboard

Launch the dashboard with external search tools enabled:

=== "macOS/Linux"

    ```bash
    URSA_DASHBOARD_USE_WEB=1 uv run ursa-dashboard
    ```

=== "Windows PowerShell"

    ```powershell
    $env:URSA_DASHBOARD_USE_WEB = "1"
    uv run ursa-dashboard
    ```

Open `http://127.0.0.1:8080`, then:

1. Confirm your model and credential source under **Settings → LLM**.
2. Create a session with a disposable workspace.
3. Select the **Execution Agent**.
4. Submit this prompt:

```text
Use the arXiv, OSTI, and web-search tools to investigate graph neural networks
for partial differential equations. Compare methods and benchmarks, emphasize
possible applications to shock hydrodynamics, distinguish claims by source
collection, and include source links.
```

Setting `URSA_DASHBOARD_USE_WEB=1` is required: it opts supported dashboard
agents into the arXiv, OSTI, and web-search tools. Follow the activity timeline
to see which tool supplied each part of the answer.

See the [dashboard guide](../../../docs/getting-started/dashboard.md) for
credential storage, workspace selection, and remote-access safety.

## 4. Compare all three agents from Python

Run the included script:

```bash
uv run acquisition_agents.py
```

The script initializes its chat model from [`config.yaml`](config.yaml), gives every
acquisition agent the same query and context, and prints three summary panels.
It limits the web and OSTI searches to five results and arXiv to three results.
Edit `config.yaml` to select another configured model or inference provider;
edit `QUERY` or `CONTEXT` in `acquisition_agents.py` to run your own comparison.

Inspect these generated paths after the run:

| Source | Retrieved material | Summaries |
| --- | --- | --- |
| Web | `web_db/` | `web_summaries/` |
| OSTI | `osti_db/` | `osti_summaries/` |
| arXiv | `arxiv_papers/` | `arxiv_generated_summaries/` |

The script intentionally performs real network requests and LLM calls. Result
availability, runtime, and cost depend on the upstream services and selected
model. Reduce each `max_results` value before experimenting if you want a
smaller first run.

For programmatic concepts and model initialization, read the
[Python guide](../../../docs/getting-started/python-scripts.md). The individual
[arXiv](../../../docs/agents/acquisition/arxiv.md),
[OSTI](../../../docs/agents/acquisition/osti.md), and
[web-search](../../../docs/agents/acquisition/web-search.md) pages document each
agent's parameters and outputs.

## Troubleshooting

- If `uv run ursa` cannot find a key, run `uv run ursa --print-config` and
  review the [configuration guide](../../../docs/configuration/index.md).
- If the dashboard does not expose search activity, stop it, set
  `URSA_DASHBOARD_USE_WEB=1`, and restart it as shown above.
- If a source returns no items, try a shorter query or rerun later; arXiv, OSTI,
  and DDGS are independent upstream services.
- If an earlier run affects the comparison, move or remove that source's cache
  and summary directories before rerunning.
