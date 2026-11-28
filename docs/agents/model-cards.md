# ModelCardAgent Documentation

[`ModelCardAgent`][ursa.agents.model_card_agent.ModelCardAgent] helps scientists find and use foundation models across physics. It reads model cards, turns a request into structured requirements, recommends models with explained scores, and runs the chosen model. In the terminal interface, the agent is `#fm`.

The agent is generic. It does not depend on one field or one model family. Its vocabularies cover weather, climate, oceans, hydrology, the cryosphere, Earth observation, seismology, exploration geophysics, fluid dynamics, heat transfer, solid mechanics, wave physics, electromagnetics, plasma physics, space physics, astrophysics, particle physics, nuclear engineering, materials, chemistry, and biomolecules, plus general PDE surrogates. You can add fields from configuration.

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ModelCardAgent

llm = init_chat_model("openai:gpt-5-mini")
agent = ModelCardAgent(llm=llm, workspace="fm_workspace")

state = agent.invoke(
    "Read hf://camlab-ethz/Poseidon-B and hf://polymathic-ai/walrus. "
    "Which one suits 2D compressible flow on a 24 GB GPU?"
)
print(agent.format_result(state))
```

The result is a Markdown report. It lists the cards read, the requirements the agent understood, the ranked models with their criteria, and the run result when a run was requested. The agent also writes each report and a JSON record of the turn to `fm_reports/` in the workspace.

In the terminal interface, use the same requests with the `#fm` prefix:

```text
#fm Read hf://nvidia/fourcastnet3 and describe its inputs and outputs.
#fm Recommend a model for 6-hourly global weather forecasts at 0.25 degrees.
#fm Run the top model with {"temperature": [300, 310, 320, 310, 300, 290, 280, 290]}
```

A follow-up run request can refer to earlier recommendations as "the top model", "#2", or "that model"; "that model" means the top-ranked compatible one. In the terminal interface, typing `#` opens the agent picker, so press Esc before typing "#2". The [model-card example](../examples/model_cards/index.md) walks through a full session with a model served over MCP.

## Using the dashboard

In the [web dashboard](../getting-started/dashboard.md), choose the agent named Foundation Model Advisor and type the same requests without the `#fm` prefix. Its advanced options set the card folders to load (`catalog_paths`), Hugging Face search (`use_web`, which `ursa-dashboard --use-web` turns on by default), whether runs execute (`allow_run`), and the number of recommendations (`top_k`). MCP servers added in the global Settings, under MCP tools, are attached to the agent, so it discovers their models as it does in the terminal interface. `ursa-dashboard --config my_ursa.yaml` also loads the `mcp_servers` of that file. Reports and run results land in the session workspace, and the activity timeline shows each stage.

Each dashboard message starts a fresh agent, and the dashboard sends the session transcript with it. The agent treats only the newest message as the request and the previous answer as context, so an earlier run never repeats. A follow-up such as "Run the top model with {...}" resolves against the ranking in the previous report. Cards stay in the group catalog between messages, so a card read once is available to later ones.

## Asking about your models and data

The agent answers three kinds of questions with the model cards as evidence: what the catalog holds, which models can use a data file, and whether a model can run on that data as is or needs fine-tuning. Every answer ends with a numbered list of sources. A claim about a model cites the card field, the quote that states it, and where the quote sits: a card section, the MCP tool schema, or `config.json`. A claim about your data cites the file and the library that read it.

```text
#fm How many models do I have?
#fm How many models accept SMILES and run on a CPU?
#fm I have molecules.smi. Which models are my choices?
#fm I have molecules.smi. Can I just run inference to get the HOMO-LUMO gap, or do I need fine-tuning?
```

An inventory question counts the catalog by source and applies the filters it names: domain, task, input format, CPU or GPU memory, license, size, and whether the model is served over MCP. A model whose card leaves a filtered fact unstated counts as one that might match, and the answer names the missing fact, so a count never hides a model. For a model served over MCP, the answer adds that it runs on the server, not on your machine. When the catalog is empty, a "Model discovery" section says why: no MCP server is connected, or a server offers no `get_model_card` tool. When every card request to a server fails, the section quotes the server's first error. When the LLM call fails, for example on an invalid API key, the agent answers with its rules alone, and an "LLM" section at the end of the answer quotes the error.

When a request names a data file, the agent reads the file first. It reads NetCDF, HDF5 (including The Well layout), NumPy, MATLAB, CSV and TSV tables, SMILES lists, FASTA, XYZ, CIF, PDB, images, and JSON. The profile states what the file holds: the data kind and representation, the variables with their units and shapes, the grid, resolution, and time step, the columns that could serve as fine-tuning targets, and the size of each molecule or sequence.

The agent then checks every catalog model against the profile and assigns one of five pathways: run inference as is, run inference after converting the data, fine-tune before use, not suitable, or not enough information. Each check pairs one card fact with one data fact: the data kind, the variables (with common aliases such as `t2m` and `2t`), the spatial dimensions, the grid, the time step, the file format, the units, the normalization, the training range, and the outputs you want. A mismatch in data kind or dimensions makes a model not suitable. A missing output or variable calls for fine-tuning. A difference in grid, units, format, or normalization calls for a conversion, and the answer lists each step. A card that leaves the input format unstated gives "not enough information", never a guess.

The answer starts with a one-line verdict. On Nomad's demo server, the gap question above gets the answer "run inference as is" with MIST QM9, and it cites the card line that lists the twelve QM9 properties. The answer also warns that two molecules in the file have more heavy atoms than the nine that bound QM9's training data, and it quotes that sentence of the card, since their predictions extrapolate. When you name no target but the file has a target column, such as measured solubility, the agent takes that column as the quantity you want and says so. A fine-tuning answer states the labeled examples in your file, the size of the data the card reports, the base model that the candidates share, whether the weights are published or the model is served only for inference, license limits, and a memory estimate for full fine-tuning.

A question never runs a model. The agent runs one only when a sentence orders it, such as "Run the QM9 model on molecules.smi" or "Predict the boiling points of the molecules in boiling_points.csv". When the order names no model, the agent runs the model whose card supports the request as is; when no card does, it runs nothing and points to the pathways. The pathway decision is deterministic, and the LLM takes no part in it.

## How the agent works

Each turn runs a LangGraph graph with eight nodes: `route`, `ingest`, `requirements`, `recommend`, `assess`, `plan_run`, `execute_run`, and `report`. The `route` node finds card sources and catalog models named in the request and decides the intents: ingest, describe, list, recommend, assess, or run. Regular expressions handle common phrasings, and an LLM call with structured output covers the rest. A request can carry several intents, such as "read these cards and recommend one". A run intent stays only when a sentence of the request orders a run, which also overrides an LLM that reads "Can I just run inference?" as an order.

The `ingest` node reads every named source and stores the normalized card in the catalog. On its first turn, it also loads the cards listed in `catalog_paths` and discovers models on connected MCP servers. The `requirements` node reads the request into [`UserRequirements`][ursa.model_cards.schema.UserRequirements] and inspects the data files it names. The `recommend` node ranks the catalog. The `assess` node checks the catalog against the data and assigns pathways. The `plan_run` and `execute_run` nodes run the chosen model. The `report` node assembles the answer and its sources.

The agent emits progress events for every stage, so the terminal interface and the dashboard show cards read, requirements, rankings, and runs as they happen.

## How the agent uses the LLM

The agent combines an LLM with deterministic code. The LLM helps the agent understand language: free phrasing in requests and facts written as prose in cards. Deterministic code decides the facts, the scores, and the hard limits. The LLM never invents a model fact, changes a score on its own, or overrides a limit. Every LLM step also has a rule-based fallback, so the agent still works when the LLM fails or returns unusable output.

The table follows one request through the agent: "Which model should I use to simulate heat conduction along a periodic rod? I have a temperature profile in kelvin and only a laptop CPU."

| Step | Deterministic code | LLM contribution | Guard on the LLM |
| --- | --- | --- | --- |
| Route | Patterns find the intents ("which model" means recommend) and the card sources | Reads intents and sources from free phrasing | A source counts only if it appears in the request, and a model only if it is in the catalog |
| Read cards | Extracts metadata and exact input and output formats from the card text, `config.json`, Hub metadata, and MCP tool schemas | Fills facts that the parser missed in prose | Each value needs a quote that matches the card, and its numbers must appear in that quote |
| Understand the request | Finds heat transfer, simulation, a temperature profile in kelvin, and the CPU-only limit | Reads the need, the data, and the wanted outputs more fully; its data and output lists replace the rule-based ones | Numeric limits only fill blanks, and a stated limit always stays |
| Rank | Scores each cataloged model on seven weighted criteria; a license, hardware, size, or exclusion conflict blocks a model | Reviews the shortlist with a fit, a short rationale, and caveats such as regridding or unit conversion | It sees only card facts, counts for 35% of the score, and cannot lift a block |
| Run | Validates arguments against the tool's JSON schema, calls the MCP tool, and saves the result | Fills arguments when the request gives no JSON, with up to three tries; for tensor inputs and models outside MCP, the ExecutionAgent, itself an LLM agent, writes and runs the code | Invalid arguments stop the run |

A recommendation therefore reads as "this model ranks first, for these reasons; check these points." The ranking and its reasons come from the code, and the LLM review adds caveats and adjusts the score. We split the work this way for three reasons. A misread shape or unit gives wrong physics, and an LLM can misread numbers, so facts come from the card and are checked. Every score breaks down by criterion, so a user can see why a model won. The same cards and request also give the same deterministic ranking.

## Recognizing model cards

The agent reads cards from five kinds of sources. A Hugging Face repository is named as `hf://org/name`, as a Hub URL, or as `` `org/name` `` in backticks; the agent fetches the README, `config.json`, and Hub metadata such as safetensors parameter counts. A URL can point to a Markdown, YAML, or JSON card. A local path can name a card file or a model directory. An MCP server can serve models with cards, as Nomad does. Finally, a card can be an URSA-native YAML or JSON file.

The extractor labels the card's dialect: the Hugging Face template, the DOE Genesis template, NVIDIA Model Card++, a free-form card, or a native card. Two more labels cover sources without card text: a repository with only `config.json` and Hub metadata, and an MCP tool described only by its description and input schema. The label is informational; every non-native card goes through the same extraction, and `science:` and `risk:` tags are read from any card. Both the extractor and the LLM pass ignore HTML comments and unfilled template text. The result is a [`ModelCard`][ursa.model_cards.schema.ModelCard] with these groups of facts:

| Group | Facts |
| --- | --- |
| Identity | name, aliases, version, developers, links, citations |
| Science | physics domains, tasks, intended and out-of-scope uses, limitations |
| Inputs and outputs | name, kind, representation, shape, layout, data type, units, channels, normalization, semantics, file formats |
| Space and time | number of spatial dimensions, grid, resolution, boundary conditions, time step, horizon, context steps |
| Model | architecture family, parameter count, framework, training data, metrics |
| Use | license, commercial use, compute needs, interfaces to load or call the model |
| Quality | provenance of each fact, issues, completeness |

Key facts record their source and method: single-valued facts such as the license, parameter count, grid, and time step, each domain and task, inputs and outputs read from prose, and every value the LLM adds. When sources disagree on a single-valued fact, the more reliable source wins: MCP tool metadata, then safetensors counts, `config.json`, Hub metadata, front matter, key-value lines, section text, free text, and finally hints in the model name, which are marked as inferred. A disagreement that matters, such as two parameter counts, becomes a `conflicting_values` issue. Native cards skip this step and keep their declared values.

The extractor favors precision over recall. A missed fact appears as a gap that a user or the LLM can fill, while a wrong fact would mislead the recommendation. For example, it reads a parameter count from a table row only when the row names this checkpoint, and it attaches a unit to a variable only when the statement names a single unit.

When an LLM is available, the agent refines each new card you name after the deterministic pass. Cards it finds on MCP servers or loads from `catalog_paths` keep the deterministic reading unless `refine_discovered` is on, since one LLM call per card would delay the first answer by minutes on a large server. The LLM fills empty fields and adds list items such as domains, tasks, and input specs; it never overrides a deterministic value. The one exception narrows use: a verified non-commercial restriction sets commercial use to false. Each value needs a quote from the card. The check ignores case, whitespace, quote styles, and Markdown markup, and it accepts "..." elisions. A quote of six or more words may differ from the card in up to 10% of its word trigrams, but never in a trigram that holds a number, and every number in the value must appear in the quote. Values that fail are dropped and reported as `llm_unverified`.

Each card also gets a completeness score. Facts needed for recommendation and running count twice as much as physics details. The card is marked ready to recommend when it states a domain, a task, and an input or output description. It is marked ready to run when it states an interface and the input and output formats.

## Recommending models

[`rank_models`][ursa.model_cards.matching.rank_models] scores each catalog model on seven criteria. The domain criterion (weight 0.26) and the task criterion (0.20) use the taxonomy, so related fields earn partial credit. The inputs (0.14) and outputs (0.10) criteria compare the data you have and the quantities you want with the model's formats. The relevance criterion (0.14) compares the request and the card text with TF-IDF, which keeps matching useful for fields outside the vocabularies. The practicality criterion (0.10) checks hardware, license, access, and how to call the model. The evidence criterion (0.06) rewards complete cards.

A criterion that the request leaves open carries less weight. A stated mismatch in spatial dimensions lowers the score, and a matching resolution or a preferred model raises it. A score of 0.62 or more is a strong fit, and a score of 0.45 or more is a moderate fit. Hard blockers mark a model incompatible regardless of its score. Blockers include a license that forbids your use, a model that does not fit your GPU or CPU limit, a size limit, and an explicit exclusion.

With an LLM, the agent reviews the shortlist. The review blends into the score with a weight of 0.35 and adds caveats, but it cannot override a blocker. When `hub_search` is on and no cataloged model fits, the agent searches Hugging Face, reads the top results, and ranks again.

## Running models

The agent runs a model in one of two ways. When the card describes an MCP tool with plain JSON inputs, the agent takes the arguments from a JSON object in your request. If there is none, or it fails validation, the LLM fills them from your request and the card, with up to three tries. The agent validates the arguments against the tool's JSON schema. Missing arguments stop the run and are listed by name; invalid values stop it with the schema's error messages. A successful call saves the result and its arguments to a new `result-<time>.json` file in `fm_runs/<model>/`. A tool that takes one SMILES string also gets it from a request such as "smi CCO" or a SMILES in backticks. A tool that takes one molecule or sequence per call runs once per item of the data file you name, up to `max_run_items` items, one call at a time. The report shows the first results as a table, and the agent saves all of them as `results-<time>.csv` and `results-<time>.json`.

MCP tools with tensor inputs and any other model go to [`ExecutionAgent`][ursa.agents.execution_agent.ExecutionAgent] with a precise brief. The brief states the card's exact input and output formats, normalization, time semantics, and loading code. It also lists sanity checks implied by the card, such as finite outputs and the documented shapes, channel order, and units, and it asks for labeled review figures of the inputs and outputs. Set `allow_run=False` to plan runs without executing them.

## Model catalog

The catalog holds one JSON file per model under `~/.cache/ursa/<group>/model_cards`, so agents in the same group share it. Pass `catalog_dir` to use another folder. The catalog finds models by id, name, alias, or repository, and it searches by keywords.

Other agents can use the catalog through five LangChain tools: `parse_model_card`, `search_model_cards`, `describe_model_card`, `recommend_models`, and `check_model_inputs`. [`model_card_tools`][ursa.model_cards.tools.model_card_tools] returns them for the `extra_tools` argument of a tool-using agent. The `ExecutionAgent` that runs models for `ModelCardAgent` receives these tools automatically.

## Describing your own model

The most reliable card is an URSA-native YAML file. It states every input and output with its kind, shape, data type, channels, units, and normalization. It also states the grid, time step, compute needs, and how to load or call the model. The [example template](../examples/model_cards/index.md#describe-your-own-model) shows every field. Fields you leave out are reported as gaps.

To serve a PyTorch model over MCP, subclass [`TorchModuleTool`][ursa.tools.fm_base_tool.TorchModuleTool] and pass the card text as `model_card`. The `add_to_fastmcp` method then registers a `get_model_card` tool next to the model tool. This tool matches Nomad's interface. `ModelCardAgent` asks each connected server's `get_model_card` for a card for every tool on that server, so models served by URSA and by Nomad are found the same way, even side by side. For an MCP server without `get_model_card`, list its model tools in `mcp_model_tools`; the agent then builds each card from the tool description and input schema.

## Extending the vocabularies

The agent matches domains and tasks with two vocabularies in [`ursa.model_cards.taxonomy`][ursa.model_cards.taxonomy]. Each concept has an id, a label, synonyms, an optional parent, and related concepts with similarity weights. To add a field, pass `custom_domains` or `custom_tasks`:

```yaml
agent_config:
  fm:
    custom_domains:
      magnetic_confinement:
        synonyms: [tokamak equilibrium, stellarator]
        parent: plasma_physics
```

A concept id may also map directly to a list of synonyms. A new concept with a parent earns partial credit against its parent and siblings. A grouping concept can lower the credit between its children with `sibling_similarity`; the built-in Physics group uses 0.2, because its fields, from particle physics to astrophysics, are far apart.

## Options

Pass these options as keyword arguments of `ModelCardAgent` or under `agent_config.fm` in the URSA configuration.

| Option | Default | Meaning |
| --- | --- | --- |
| `catalog_dir` | group cache | Folder of the model catalog |
| `catalog_paths` | none | Card files or folders to load on the first turn; relative paths resolve against the workspace |
| `use_web` | `False` | Follows `--use-web`; also the default of `hub_search` |
| `hub_search` | `use_web` | Search Hugging Face when no cataloged model fits |
| `llm_extraction` | `True` | Refine new cards with evidence-checked LLM extraction |
| `refine_discovered` | `False` | Also refine cards found on MCP servers or in `catalog_paths` |
| `llm_review` | `True` | Review the shortlist with the LLM |
| `llm_routing` | `True` | Use the LLM to read intents and sources |
| `allow_run` | `True` | Execute runs; `False` only plans them |
| `top_k` | `3` | Number of recommendations in the report |
| `discover_mcp_models` | `True` | Discover models on connected MCP servers |
| `mcp_model_tools` | none | MCP tools to treat as models without `get_model_card` |
| `custom_domains`, `custom_tasks` | none | Extra vocabulary concepts |
| `executor_kwargs` | none | Keyword arguments for the `ExecutionAgent` that runs models |
| `write_reports` | `True` | Write reports to `fm_reports/` |
| `fetch_timeout` | `30.0` | Timeout in seconds for network reads |
| `max_run_items` | `100` | Most items of a data file to run, one tool call per item |

The agent fetches a card over the network only when you name a Hugging Face repository or a URL; local cards are read from disk. It searches the Hugging Face Hub only when `hub_search` is on (it follows `use_web`, which `--use-web` sets) and no cataloged model fits. Card text and requests go to the configured LLM provider. Set `HF_TOKEN` to read gated Hugging Face repositories.

## Accuracy

We checked recognition against 24 public Hugging Face cards: 23 science models from materials, chemistry, proteins, astronomy, Earth observation, weather, hydrology, fluid dynamics, solar physics, gravitational waves, cosmology, and PDE solvers, plus one non-science control. We wrote the expected facts by reading each card before running the extractor. The benchmark scores 240 items over 13 fields: 113 facts that the cards state, and 127 checks that the extractor reports nothing where a card is silent. Twelve cards were held out from all development; their one scoring gave 94.1% (111 of 118), and 86.8% (46 of 53) on stated facts. After fixes, the extractor matched 237 of 240 items (98.8%) and 110 of 113 stated facts (97.3%); these scores are in-sample. Units and normalization are not scored yet. These numbers measure the deterministic extractor alone: the benchmark does not run the LLM refinement, and the LLM steps have not yet been evaluated with a production model. The example folder contains the facts and the scoring script, so you can repeat the check.

The folder `examples/model_cards/curated_tasks` holds ten curated inference tasks for Nomad's demo server, which serves five MIST molecular property models and the Janus HEAT-PLI field model. Each task is a question about a data file with the answer that the cards support: a model count, a filtered count, the choices for a list of molecules, the HOMO-LUMO gap, the boiling point, the refractive index (no model predicts it, so the answer is fine-tuning), a HEAT-PLI state (inference after a WellFormat conversion), protein sequences (no model fits), and two runs. On October 27, 2026, all ten tasks passed every check with the rule-based path alone, without an LLM. The boiling-point run also compares the predictions with measured values. As served by the demo server, the MIST boiling-point model missed the measured normal boiling points of eight common liquids by 52.0 °C on average. The agent selects models by what their cards state; it cannot vouch for a model's accuracy. We therefore recommend a reference task like this one for every served model.

Recognition can only be as good as the card. Many cards leave out units, normalization, or loading code, and some state key facts only in a paper. The agent reports these gaps instead of guessing, and the recommendation lists them as checks for the user.
