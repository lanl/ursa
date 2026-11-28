# Recognize, recommend, and run foundation models

In this tutorial, we use URSA's `ModelCardAgent` to work with scientific foundation models. The agent reads model cards, turns your request into structured requirements, recommends the best-fitting models with reasons, and runs the chosen model. We use it in the terminal interface as `#fm` and from Python as `ModelCardAgent`.

The example ships a small physics model, `toy-heat-1d`, served over MCP. It solves the 1D heat equation exactly on a periodic rod, so you can check every output against theory. Its server publishes a model card through a `get_model_card` tool, the same interface that [Nomad](https://github.com/lanl/nomad) uses for its scientific foundation models (SciFMs). We also read public cards from Hugging Face that cover fluid dynamics, weather, and chemistry.

## What the agent does

The agent first recognizes model cards. It reads Hugging Face repositories, URLs, local files, and models on MCP servers. It understands the Hugging Face template, the DOE Genesis template, NVIDIA Model Card++, free-form cards, and URSA's own YAML format. Each card becomes one normalized record that states the model's domains, tasks, license, size, and exact input and output formats: names, shapes, units, data types, channels, normalization, time steps, and grids. Key facts keep a pointer to where they came from. The agent reports missing facts and conflicts between sources instead of guessing them.

Next, the agent reads your request. It extracts the physics domain, the task, the data you have, the quantities you need, and limits on hardware, license, and size. It then ranks the cataloged models with a weighted score and explains each criterion. An LLM review can adjust the ranking, but only with evidence from the cards.

Finally, the agent runs the chosen model. For a model served over MCP, it fills the tool arguments from your request, validates them against the tool's JSON schema, and calls the tool. For other models, it hands a precise brief with the documented loader to URSA's `ExecutionAgent`.

## Prerequisites

You need [uv](https://docs.astral.sh/uv/) and an API key for the inference provider URSA will use. Reading public cards from Hugging Face needs network access. The toy model runs on a laptop CPU in well under a second.

Run every command from the [`examples/model_cards`](.) directory.

## Install the example environment

```bash
uv sync
uv run ursa --help
```

The environment installs URSA from the checkout two directories above this one, with the `fm` extra that provides PyTorch. If this is your first URSA session, follow the [configuration overview](../../docs/configuration/index.md) to set up your LLM credentials.

The example contains these files:

| File | Purpose |
| --- | --- |
| [`toy_heat_server.py`](toy_heat_server.py) | MCP server for `toy-heat-1d`, with a Genesis-style model card |
| [`ursa.yaml`](ursa.yaml) | Starts the toy server and loads the native card for the `#fm` agent |
| [`nomad.yaml`](nomad.yaml) | Connects the `#fm` agent to a local Nomad server for the live demo |
| [`model_card_template.yaml`](model_card_template.yaml) | URSA-native card for a fictional 2D reacting-flow surrogate |
| [`run_example.py`](run_example.py) | The same workflow through the Python API |
| [`benchmark/`](benchmark) | Hand-checked facts for 24 public cards and a scoring script |
| [`curated_tasks/`](curated_tasks) | Curated inference tasks for Nomad's demo server, with data and a scoring script |

## Use the terminal interface

Start URSA with the example configuration:

```bash
uv run ursa --config ursa.yaml
```

The configuration starts `toy_heat_server.py` as an MCP server over stdio. It also loads `model_card_template.yaml` into the model catalog. Relative catalog paths resolve against the workspace, which defaults to the current directory.

Ask the agent to read a few public cards:

```text
#fm Read hf://camlab-ethz/Poseidon-B, hf://polymathic-ai/walrus, and
hf://nvidia/fourcastnet3, and describe their inputs and outputs.
```

The report lists each model with its license, size, domains, and tasks. It then gives every input and output with its shape, units, and channels, and flags facts the card leaves out. You can also name a card by URL, by a local path, or by a repository id in backticks, such as `` `nvidia/fourcastnet3` ``.

Now describe a need in plain language:

```text
#fm Which model should I use to simulate heat conduction along a periodic rod?
I have a temperature profile in kelvin and only a laptop CPU.
```

On its first turn, the agent asks the toy server for its model card and adds `toy-heat-1d` to the catalog. It should rank `toy-heat-1d` first as a strong fit. The report shows the requirements it understood and explains each score with the domain, task, input, and practicality criteria. It also lists the checks a user should make, such as missing CPU support or loading code.

Then run the recommended model:

```text
#fm Run the top model with {"temperature": [300, 300, 310, 340, 340, 310, 300, 300],
"diffusivity_m2_s": 0.001, "time_s": 20}
```

The agent resolves "the top model" from the previous recommendation. You can also say "#2" or name the model. It validates the arguments, calls the tool, and saves the result as a new `result-<time>.json` file in `fm_runs/toy-heat-1d/`. Diffusion on a periodic rod conserves the mean, so the mean temperature stays at 312.5 K, the mean of the input profile.

Each turn also writes a Markdown report and a JSON record to `fm_reports/`. To see the tools that the toy server provides, enter `/agents` and open the `#fm` tab. The tools `toy-heat-1d` and `get_model_card` show `MCP: toy`.

## Use the Python API

The script runs the same three steps without the terminal interface:

```bash
uv run python run_example.py
uv run python run_example.py --offline --model openai:gpt-5-mini
```

`--offline` skips the Hugging Face cards. The script writes its catalog, reports, and results to `workspace/` in this folder. The core of the script looks like this:

```python
agent = ModelCardAgent(llm=llm, workspace="workspace")
await agent.add_mcp_tools(client)
state = await agent.ainvoke("Recommend a model to simulate heat conduction ...")
print(agent.format_result(state))
```

You can also use the recognition step without an agent or an LLM:

```python
from ursa.model_cards import extract_model_card, load_document

card = extract_model_card(load_document("hf://nvidia/fourcastnet3"))
print(card.brief())
for spec in card.inputs + card.outputs:
    print(spec.name, spec.kind, spec.shape, spec.units)
    print("  channels:", [channel.name for channel in spec.channels])
```

## Describe your own model

The most reliable way to describe your own model is a URSA-native card. Copy [`model_card_template.yaml`](model_card_template.yaml) and edit it. The card states each input and output with its kind, shape, data type, channels with units, and normalization. It also states the spatial grid, the time step, the compute needs, and how to load or call the model. Leave out what you do not know; the agent reports it as a gap and never fills it in. Domains and tasks use the identifiers listed in [`ursa.model_cards.taxonomy`](../../src/ursa/model_cards/taxonomy.py). A new label is also accepted.

A Markdown card works too. The agent extracts the most from a card that names each input and output with its shape and units, for example in a table with Name, Shape, and Dtype columns. It also helps to give the license in the front matter and to state the time step and forecast horizon in a sentence.

Put your cards in a folder and list it under `catalog_paths`:

```yaml
agent_config:
  fm:
    catalog_paths:
      - my_cards
```

The agent keeps its catalog under `~/.cache/ursa/<group>/model_cards`, one JSON file per model. Agents in the same group share the catalog. The agent reads a source again when you name it, and it repeats the LLM refinement only when the source has changed.

## Serve your model over MCP or Nomad

The agent discovers a model on an MCP server when the server offers a `get_model_card` tool that returns the card for a tool name. Nomad offers this tool for every SciFM it serves. To connect a running Nomad server, add it to `ursa.yaml`:

```yaml
mcp_servers:
  nomad:
    transport: streamable-http
    url: http://localhost:38217/mcp
```

The [Nomad and MIST example](../nomad_mist/README.md) shows how to start the server. After that, `#fm List the available models` shows the Nomad models next to the toy model.

To serve your own PyTorch model, subclass `TorchModuleTool` and pass the card text as `model_card`, as `toy_heat_server.py` does. The `add_to_fastmcp` method registers the tool and the shared `get_model_card` tool together. For a server without `get_model_card`, list its model tools under `mcp_model_tools` in the `fm` agent configuration. The agent then builds a card from each tool's description and input schema.

A server that needs a token takes a header, as [MCP server configuration](../../docs/configuration/mcp.md) shows. To see which models the agent will find, run `uv run python check_cards.py nomad.yaml` with your config file in place of `nomad.yaml`. The script connects to each server, asks for each tool's card as the agent does, and ends with the number of models the agent will count.

## Ask about your models and data

The agent also answers questions about the catalog and about your data, and it cites the card lines behind each answer. Start the [Nomad demo container](../nomad_mist/README.md#start-the-nomad-container), copy the curated data into a demo workspace, and start URSA with [`nomad.yaml`](nomad.yaml), which connects the agent to Nomad:

```bash
mkdir -p demo && cp curated_tasks/data/* demo/
uv run ursa --config nomad.yaml --workspace demo
```

Then try:

```text
#fm How many models do I have?
#fm I have molecules.smi. Which models are my choices?
#fm I have molecules.smi. Can I just run inference to get the HOMO-LUMO gap, or do I need fine-tuning?
#fm Run the QM9 model on molecules.smi.
```

The agent reads the file, checks every catalog model against it, and gives a verdict: run inference as is, run inference after converting the data, fine-tune before use, not suitable, or not enough information. For the gap question, it names MIST QM9, cites the card line that lists the gap among its outputs, and warns that aspirin and caffeine have more heavy atoms than QM9's training molecules. The first three questions never run a model. The last one runs QM9 once per molecule and saves a CSV table of the results in `fm_runs/`.

## Run the curated inference tasks

The [`curated_tasks`](curated_tasks) folder holds ten questions that a scientist asks about a data file, each with the answer that the model cards support. The expected answers hold for Nomad's demo server: five MIST molecular property models and the Janus HEAT-PLI field model. The data are small files of common molecules with measured boiling points and refractive indices, and two protein sequences.

Start the Nomad container as the [Nomad and MIST example](../nomad_mist/README.md#start-the-nomad-container) shows, then run:

```bash
uv run python curated_tasks/run_tasks.py --offline
```

The script asks the agent each question in turn and checks the answer against [`tasks.yaml`](curated_tasks/tasks.yaml). The option `--offline` uses the agent's rule-based path alone, so the run needs no API key and repeats exactly. Replace it with `--model anthropic:claude-haiku-5-5` or another model to include the LLM steps; the script then prints each step as it runs and the tokens it used. The option `--refine-cards` also has the LLM review each card found on Nomad, at one call per card. The heat task uses a zero-valued file with the HEAT-PLI layout, which carries the facts that decide the pathway; pass `--heat-file` with a real state, such as Nomad's `docs/guides/angled_half_res.h5`, to use real data. The option `--only` runs selected tasks, for example `--only gap run_bp`.

The script writes `curated_tasks/results/summary.md` with one row per task, the agent's report for each task in `results/reports/`, and three figures. `fig_pathways.png` shows the pathway of every model for every task. `fig_run_bp.png` compares the predicted boiling points with measured values. `fig_run_qm9.png` shows the predicted HOMO-LUMO gap of each molecule and marks the molecules outside QM9's training range.

On October 27, 2026, all ten tasks passed every check in offline mode. The boiling-point run showed a mean absolute error of 52.0 °C against the measured values, with the largest error for toluene. That error belongs to the served model, not to the model selection, and it shows why a reference task is worth keeping for every served model.

To add a task, append an entry to `tasks.yaml` with a question, a data file in `curated_tasks/data/`, and the expected answer. The file lists the expectation keys it supports.

## Check recognition accuracy

The [`benchmark`](benchmark) folder holds hand-checked facts for 24 public cards: 23 science models from materials, chemistry, proteins, astronomy, Earth observation, weather, hydrology, fluid dynamics, solar physics, gravitational waves, cosmology, and PDE solvers, plus one non-science control. We wrote the facts by reading each card before running the extractor. The script fetches the cards and scores 240 items over 13 fields: 113 facts that the cards state, and 127 checks that the extractor reports nothing where a card is silent.

```bash
uv run python benchmark/score_cards.py --details
```

On October 17, 2026, the extractor matched 237 of 240 items (98.8%) and 110 of 113 stated facts (97.3%). These scores are in-sample. The one scoring of the 12 cards in set B, before any tuning on them, gave 94.1% overall and 86.8% (46 of 53) on stated facts. Units and normalization are not scored yet. The three remaining errors are two task labels and one parameter count. That card states the count only in a sentence that names two model variants, and the extractor does not guess which variant the checkpoint is. Cards change over time, so check the details before you read a lower score as a regression.

## Configuration options

The `fm` agent accepts these options under `agent_config.fm` in `ursa.yaml` or as keyword arguments of `ModelCardAgent`. The option `catalog_paths` lists folders or files to load at start. The option `hub_search` lets the agent search Hugging Face for more models when no cataloged model fits; it follows `--use-web` by default. The option `allow_run: false` limits the agent to planning a run without executing it. The option `top_k` sets the number of recommendations, and `mcp_model_tools` names MCP tools to treat as models. The options `custom_domains` and `custom_tasks` add vocabulary for fields the built-in taxonomy does not cover.

## Troubleshooting

If the agent does not list `toy-heat-1d`, check that `uv run ursa --config ursa.yaml --print-config` shows `mcp_servers.toy`. Then run `uv run python toy_heat_server.py` by itself to see any import error, and stop it with **Ctrl+C**.

If `#fm How many models do I have?` reports 0 models, read the "Model discovery" section of the answer. It says whether no MCP server is connected, whether a server offers no `get_model_card` tool, or whether every card request to a server failed, with the server's first error. Nomad downloads a model's `README.md` from Hugging Face when it has no local copy, so a Nomad server without access to Hugging Face returns no cards. `check_cards.py` shows the same facts before you start URSA.

If an answer ends with an "LLM" section, the LLM call failed, for example on an invalid API key, and the agent answered with its rules alone. The section quotes the error.

If a Hugging Face card fails with a 401 or 403 status, the repository is gated or private. Accept its terms on the Hub and set `HF_TOKEN`.

If a run stops on missing or invalid arguments, the message lists the missing names or the schema errors. The `temperature` list needs 8 to 4096 values, and `diffusivity_m2_s` must be positive.

## Clean up

Delete `fm_reports/`, `fm_runs/`, `workspace/`, `demo/`, and `curated_tasks/results/` in this folder. To clear the catalog, delete `~/.cache/ursa/default/model_cards`.
