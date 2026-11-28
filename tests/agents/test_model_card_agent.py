import asyncio
import json
import math
from collections import Counter

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException

from tests.model_cards import cards
from ursa.agents import ModelCardAgent
from ursa.agents.model_card_agent import (
    _FEASIBILITY,
    RouteDecision,
    _orders_run,
    _previous_choice,
    _split_request,
)
from ursa.model_cards.catalog import ModelCatalog

from .utils import RecordingChatModel, assert_requests_provider_valid

HEAT_CARD = """---
license: bsd-3-clause
tags:
- heat transfer
---

# Toy Heat Propagator

Toy Heat Propagator advances a 1D periodic temperature profile by solving the heat equation exactly in Fourier space.

## Inputs and outputs

- Input: `profile`, temperature samples in kelvin on a uniform periodic grid.
- Output: `profile`, the temperature profile after the requested time, in kelvin.
"""


class StageRecorder(BaseCallbackHandler):
    def __init__(self):
        self.stages = Counter()

    def on_custom_event(self, name, data, **kwargs):
        self.stages[data.get("stage")] += 1


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "cards" / "plasma").mkdir(parents=True)
    (root / "cards" / "plasma" / "README.md").write_text(
        cards.GENESIS_CARD, encoding="utf-8"
    )
    (root / "cards" / "crystal.md").write_text(cards.HF_CARD, encoding="utf-8")
    (root / "cards" / "ocean.md").write_text(
        cards.NVIDIA_CARD, encoding="utf-8"
    )
    (root / "cards" / "chemnet.md").write_text(
        cards.LICENSE_CONFLICT_CARD, encoding="utf-8"
    )
    return root


@pytest.fixture
def make_agent(workspace, tmp_path):
    agents = []

    def build(llm=None, **kwargs):
        llm = llm or RecordingChatModel(
            structured_factory=lambda schema: schema()
        )
        agent = ModelCardAgent(
            llm=llm,
            workspace=workspace,
            catalog_dir=tmp_path / "catalog",
            **kwargs,
        )
        agents.append(agent)
        return agent, llm

    yield build
    for agent in agents:
        agent.close()


def test_reads_and_describes_cards(make_agent, workspace, tmp_path):
    agent, llm = make_agent()
    recorder = StageRecorder()
    state = agent.invoke(
        "Read cards/plasma and describe its inputs.",
        config={"callbacks": [recorder]},
    )
    report = agent.format_result(state)

    assert state["intents"][:2] == ["ingest", "describe"]
    assert "### PlasmaEdge-2D" in report
    assert "shape (3, 256, 512)" in report
    assert "channels (3): density, electron_temperature, potential" in report
    assert "10 microseconds" in report
    assert {"route", "ingest", "ingest_result", "report"} <= set(
        recorder.stages
    )
    assert len(ModelCatalog(tmp_path / "catalog")) == 1
    assert list((workspace / "fm_reports").glob("report-*.md"))
    assert_requests_provider_valid(llm.calls)


def test_recommends_with_constraints_from_seeded_catalog(
    make_agent, workspace, tmp_path
):
    agent, llm = make_agent(catalog_paths=["cards"])
    state = agent.invoke(
        "Recommend a model to predict reaction yields of molecules from SMILES. "
        "It must allow commercial use."
    )
    recs = state["recommendations"]
    chemnet = next(r for r in recs if r.name == "ChemNet")
    assert chemnet.fit == "incompatible"
    assert any("commercial" in b for b in chemnet.blockers)
    report = agent.format_result(state)
    assert "## Requirements understood" in report and "commercial use" in report
    assert "## Recommendations" in report
    assert_requests_provider_valid(llm.calls)

    state = agent.invoke(
        "Which model forecasts sea surface temperature at 1 degree?"
    )
    assert state["recommendations"][0].name == "OceanCast"


def heat_tools():
    async def get_model_card(tool_name: str) -> str:
        if tool_name == "toy-heat":
            return HEAT_CARD
        raise ToolException(f"No model registered with the name '{tool_name}'.")

    async def toy_heat(profile: list[float], alpha: float, time: float) -> str:
        n = len(profile)
        out = []
        for i in range(n):
            value = 0.0
            for k in range(n):
                coeff = (
                    sum(
                        p * math.cos(2 * math.pi * k * j / n)
                        for j, p in enumerate(profile)
                    )
                    / n
                )
                coeff_s = (
                    sum(
                        p * math.sin(2 * math.pi * k * j / n)
                        for j, p in enumerate(profile)
                    )
                    / n
                )
                wavenumber = 2 * math.pi * min(k, n - k)
                decay = math.exp(-alpha * wavenumber**2 * time)
                value += decay * (
                    coeff * math.cos(2 * math.pi * k * i / n)
                    + coeff_s * math.sin(2 * math.pi * k * i / n)
                )
            out.append(round(value, 6))
        return json.dumps({"profile": out, "units": "K"})

    async def search_pubchem(query: str) -> str:
        return "CCO"

    schema = {
        "type": "object",
        "properties": {
            "profile": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Temperatures in K",
            },
            "alpha": {"type": "number", "description": "Thermal diffusivity"},
            "time": {"type": "number", "description": "Elapsed time"},
        },
        "required": ["profile", "alpha", "time"],
    }
    return [
        StructuredTool(
            name="get_model_card",
            description="Return the Markdown model card for a registered SciFM tool.",
            args_schema={
                "type": "object",
                "properties": {"tool_name": {"type": "string"}},
                "required": ["tool_name"],
            },
            coroutine=get_model_card,
        ),
        StructuredTool(
            name="toy-heat",
            description="Advance a periodic 1D temperature profile.",
            args_schema=schema,
            coroutine=toy_heat,
        ),
        StructuredTool(
            name="search_pubchem",
            description="Look up a SMILES string.",
            args_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            coroutine=search_pubchem,
        ),
    ]


def test_discovers_and_runs_mcp_served_model(make_agent, workspace, tmp_path):
    agent, llm = make_agent()
    agent.add_tool(heat_tools())
    agent._tool_sources = {
        "get_model_card": "nomad",
        "toy-heat": "nomad",
        "search_pubchem": "nomad",
    }

    state = agent.invoke(
        'Run toy-heat with {"profile": [300, 310, 300, 290], "alpha": 0.001, "time": 1.0}'
    )

    catalog = ModelCatalog(tmp_path / "catalog")
    assert [card.id for card in catalog] == ["mcp:nomad:toy-heat"]
    card = catalog.get("toy-heat")
    assert card.source.kind == "nomad"
    assert card.inputs[0].name == "profile" and card.inputs[0].units == "K"
    plan = state["run_plan"]
    assert plan.mode == "mcp_tool" and plan.model_id == "mcp:nomad:toy-heat"
    (saved,) = (workspace / plan.output_dir).glob("result-*.json")
    result = json.loads(saved.read_text(encoding="utf-8"))
    profile = result["result"]["profile"]
    assert len(profile) == 4
    assert abs(sum(profile) / 4 - 300.0) < 1e-6, "diffusion conserves the mean"
    assert max(profile) < 310.0
    assert "Saved to" in agent.format_result(state)


class TwoServerClient:
    """Stand-in MultiServerMCPClient: two servers, each with get_model_card."""

    def __init__(self, groups):
        self.groups = groups
        self.connections = dict.fromkeys(groups, {})

    async def get_tools(self, server_name=None):
        return self.groups[server_name]


def card_server(model, card):
    async def get_model_card(tool_name: str) -> str:
        if tool_name == model:
            return card
        raise ToolException(f"No model registered with the name '{tool_name}'.")

    async def run(steps: int) -> str:
        return json.dumps({"steps": steps})

    schema = {
        "type": "object",
        "properties": {"steps": {"type": "integer"}},
        "required": ["steps"],
    }
    return [
        StructuredTool(
            name="get_model_card",
            description="Return the Markdown model card for a model tool.",
            args_schema={
                "type": "object",
                "properties": {"tool_name": {"type": "string"}},
                "required": ["tool_name"],
            },
            coroutine=get_model_card,
        ),
        StructuredTool(
            name=model,
            description=f"Run {model}.",
            args_schema=schema,
            coroutine=run,
        ),
    ]


def test_discovers_models_on_every_server_with_model_cards(
    make_agent, tmp_path
):
    # Both servers name their card tool get_model_card; neither may shadow
    # the other.
    agent, _ = make_agent()
    client = TwoServerClient({
        "nomad": card_server("toy-heat", HEAT_CARD),
        "local": card_server(
            "toy-wave", "# Toy Wave\n\nToy Wave advances 1D acoustic waves.\n"
        ),
    })
    sources = asyncio.run(agent.add_mcp_tools(client))
    assert sources == {
        "get_model_card": "local",
        "toy-heat": "nomad",
        "toy-wave": "local",
    }

    agent.invoke("List the available models")

    catalog = ModelCatalog(tmp_path / "catalog")
    assert sorted(card.id for card in catalog) == [
        "mcp:local:toy-wave",
        "mcp:nomad:toy-heat",
    ]
    # Each card came from its own server's get_model_card.
    assert "periodic temperature profile" in catalog.get("toy-heat").summary
    assert "acoustic waves" in catalog.get("toy-wave").summary


def test_mcp_run_without_arguments_asks_for_them(
    make_agent, workspace, tmp_path
):
    agent, _ = make_agent()
    agent.add_tool(heat_tools())
    agent._tool_sources = {"get_model_card": "nomad", "toy-heat": "nomad"}
    state = agent.invoke("Run toy-heat on my rod.")
    assert "needs valid values for profile, alpha, time" in state["run_result"]


class StubExecutor:
    def __init__(self):
        self.calls = []

    def invoke(self, inputs, config=None):
        self.calls.append((inputs, config))
        return {"messages": [AIMessage(content="done")]}

    def format_result(self, result):
        return "Wrote fm_runs/plasmaedge-2d/run_model.py and saved predictions."

    def close(self):
        pass

    async def aclose(self):
        pass


def test_hands_non_mcp_runs_to_execution_agent(make_agent, workspace, tmp_path):
    agent, _ = make_agent(catalog_paths=["cards"])
    stub = StubExecutor()
    agent._execution_agent = stub

    state = agent.invoke("Run PlasmaEdge-2D on my state.npz for 20 steps.")

    plan = state["run_plan"]
    assert plan.mode == "execution_agent" and plan.model_name == "PlasmaEdge-2D"
    ((inputs, config),) = stub.calls
    brief = inputs["messages"][0].content
    assert "shape=(3, 256, 512)" in brief and "trust_remote_code=True" in brief
    assert config["configurable"]["model_catalog_dir"] == str(
        tmp_path / "catalog"
    )
    assert "saved predictions" in agent.format_result(state)


def test_follow_up_turn_reuses_recommendations(make_agent, workspace, tmp_path):
    agent, _ = make_agent(catalog_paths=["cards"], allow_run=False)
    first = agent.invoke(
        "Recommend a model for tokamak edge plasma turbulence rollouts."
    )
    assert first["recommendations"][0].name == "PlasmaEdge-2D"

    follow_up = agent.format_query("Now run the top model.", state=first)
    assert follow_up["recommendations"] == first["recommendations"]
    assert follow_up["report"] == "" and follow_up["intents"] == []
    state = agent.invoke(follow_up)
    assert state["run_plan"].model_name == "PlasmaEdge-2D"
    assert "allow_run=False" in state["run_result"]


def test_previous_agent_output_is_context_not_request(
    make_agent, workspace, tmp_path
):
    agent, _ = make_agent()
    state = agent.invoke(
        "The last agent output was: hf://org/should-not-be-read and other notes\n\n"
        "The user stated: List the catalog."
    )
    assert state["intents"] == ["list"]
    assert state["card_sources"] == []
    assert "should-not-be-read" in state["context"]


def test_router_cannot_turn_a_data_file_into_a_card(make_agent, workspace):
    (workspace / "molecules.smi").write_text("CCO\nc1ccccc1\n", "utf-8")

    def route(schema):
        if schema is RouteDecision:
            return RouteDecision(
                intents=["ingest", "recommend"],
                card_sources=["molecules.smi"],
            )
        return schema()

    agent, _ = make_agent(llm=RecordingChatModel(structured_factory=route))
    state = agent.invoke("I have molecules.smi. Which models are my choices?")
    assert state["card_sources"] == []
    assert not [c.id for c in agent.catalog.all() if "molecules" in c.id]


HEAT_SOURCES = {
    "get_model_card": "nomad",
    "toy-heat": "nomad",
    "search_pubchem": "nomad",
}
HEAT_RUN = (
    'Run toy-heat with {"profile": [300, 310, 300, 290], "alpha": 0.001, '
    '"time": 1.0}'
)


def heat_agent(make_agent):
    agent, _ = make_agent()
    agent.add_tool(heat_tools())
    agent._tool_sources = dict(HEAT_SOURCES)
    return agent


def test_dashboard_transcript_runs_only_the_new_message(make_agent, workspace):
    # The dashboard sends the whole session, then builds a fresh agent per run.
    first = heat_agent(make_agent).invoke(f"User: {HEAT_RUN}")
    run_dir = workspace / first["run_plan"].output_dir
    assert len(list(run_dir.glob("result-*.json"))) == 1

    transcript = (
        f"User[model_card_agent]: {HEAT_RUN}\n\n"
        "Assistant[model_card_agent]: Saved to the run folder.\n\n"
        "User: Which models are in the catalog?"
    )
    state = heat_agent(make_agent).invoke(transcript)

    assert state["request"] == "Which models are in the catalog?"
    assert "run" not in state["intents"]
    assert state.get("run_plan") is None
    assert len(list(run_dir.glob("result-*.json"))) == 1, "no second run"


def test_dashboard_follow_up_runs_the_top_model_of_the_last_report(
    make_agent, workspace
):
    first_agent = heat_agent(make_agent)
    ask = "Recommend a model for periodic heat conduction in 1D."
    first = first_agent.invoke(f"User: {ask}")
    top = first["recommendations"][0].model_id
    transcript = (
        f"User[model_card_agent]: {ask}\n\n"
        f"Assistant[model_card_agent]: {first_agent.format_result(first)}\n\n"
        "User: " + HEAT_RUN.replace("toy-heat", "the top model")
    )
    state = heat_agent(make_agent).invoke(transcript)

    assert state["run_plan"].model_id == top
    assert list(
        (workspace / state["run_plan"].output_dir).glob("result-*.json")
    )


def test_a_follow_up_takes_its_requirements_from_the_new_message(
    make_agent, workspace
):
    molecule_workspace(workspace)
    first_agent, _ = make_agent(catalog_paths=["cards"])
    ask = "I have mols.smi. Can I run inference to get the HOMO-LUMO gap?"
    first = first_agent.invoke(f"User: {ask}")
    transcript = (
        f"User[model_card_agent]: {ask}\n\n"
        f"Assistant[model_card_agent]: {first_agent.format_result(first)}\n\n"
        "User: I have sol.csv. Can I just run inference, or do I need "
        "fine-tuning?"
    )
    agent, _ = make_agent(catalog_paths=["cards"])
    state = agent.invoke(transcript)

    # The previous answer names mols.smi and the gap; neither carries over.
    req = state["requirements"]
    assert req.data_paths == ["sol.csv"]
    assert "gap" not in " ".join(o.name.lower() for o in req.outputs)

    # "The same file" still resolves against the previous turn.
    again = (
        f"User[model_card_agent]: {ask}\n\n"
        f"Assistant[model_card_agent]: {first_agent.format_result(first)}\n\n"
        "User: Which models can take the same file?"
    )
    agent, _ = make_agent(catalog_paths=["cards"])
    assert agent.invoke(again)["requirements"].data_paths == ["mols.smi"]


def test_split_request_keeps_only_the_newest_message():
    transcript = (
        "User[model_card_agent]: first\n\n"
        "Assistant[model_card_agent]: report\n\nwith a blank line\n\n"
        "User: second"
    )
    assert _split_request(transcript) == (
        "second",
        "report\n\nwith a blank line",
    )
    assert _split_request("User: Read hf://a/b") == ("Read hf://a/b", "")
    assert _split_request("plain request") == ("plain request", "")
    note = "Compare these.\n\nUser: notes"
    assert _split_request(note) == (note, "")


def test_previous_choice_reads_the_ranking_of_a_report():
    report = (
        "## Recommendations\n"
        "1. **A** (`hf:a/a`): strong fit, score 0.90\n"
        "2. **B** (`hf:b/b`): moderate fit, score 0.50\n"
        "3. **C** (`hf:c/c`): incompatible fit, score 0.80\n"
    )
    assert _previous_choice("Run #2 with {}", [], report) == "hf:b/b"
    assert _previous_choice("Run that model", [], report) == "hf:a/a"
    assert _previous_choice("Run the third one", [], report) is None
    assert _previous_choice("Run it", [], "") is None


GAP_CARD = """
name: GapNet
summary: Predicts the HOMO-LUMO gap of small organic molecules from SMILES.
domains: [molecular_chemistry]
tasks: [property_prediction]
license: mit
inputs:
  - name: smi
    kind: molecule
    representation: SMILES
outputs:
  - name: homo_lumo_gap
    kind: scalar
    units: eV
    description: HOMO-LUMO gap
interfaces:
  - kind: python
    entry_point: gapnet:load
"""


def molecule_workspace(workspace):
    (workspace / "cards" / "gapnet.yaml").write_text(GAP_CARD, encoding="utf-8")
    (workspace / "mols.smi").write_text(
        "CCO\nc1ccccc1\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C\n"
    )
    (workspace / "sol.csv").write_text(
        "smiles,solubility\nCCO,0.5\nc1ccccc1,-1.6\n"
    )


def test_inventory_questions_count_and_filter_without_running(
    make_agent, workspace
):
    molecule_workspace(workspace)
    agent, _ = make_agent(catalog_paths=["cards"])

    state = agent.invoke("How many models do I have?")
    assert state["intents"] == ["list"]
    report = agent.format_result(state)
    assert "You have **5** model(s) in the catalog" in report

    state = agent.invoke("How many chemistry models run on a CPU?")
    assert "run" not in state["intents"] and state.get("run_plan") is None
    report = agent.format_result(state)
    assert "With your filters (" in report and "runs on a CPU" in report
    assert "## Sources" in report


def test_feasibility_question_gives_a_cited_verdict_and_never_runs(
    make_agent, workspace
):
    molecule_workspace(workspace)
    agent, _ = make_agent(catalog_paths=["cards"])
    state = agent.invoke(
        "I have mols.smi. Can I just perform inference to get the HOMO-LUMO gap, "
        "or do I need fine-tuning?"
    )
    assert "assess" in state["intents"] and "run" not in state["intents"]
    assert state.get("run_plan") is None
    (profile,) = state["data_profiles"]
    assert profile.kind == "molecule" and profile.n_samples == 3
    best = state["assessments"][0]
    assert (best.name, best.pathway) == ("GapNet", "inference")
    report = agent.format_result(state)
    assert "## Your data" in report and "## Inference or fine-tuning?" in report
    assert "Run inference as is" in report
    assert "## Sources" in report and "GapNet card, input 'smi'" in report
    # Data sources name the file by its workspace path, not a home folder.
    assert "from mols.smi in the workspace" in report

    state = agent.invoke(
        "I have sol.csv. Can I use GapNet to predict solubility, or do I need to fine-tune?"
    )
    (verdict,) = state["assessments"]
    assert verdict.pathway == "fine_tune"
    assert any("solubility" in note for note in verdict.fine_tune_notes)


def test_data_question_lists_the_choices_for_that_data(make_agent, workspace):
    molecule_workspace(workspace)
    agent, _ = make_agent(catalog_paths=["cards"])
    state = agent.invoke(
        "I have mols.smi that I want to use. Which models are my choices?"
    )
    assert "recommend" in state["intents"] and state["assessments"]
    report = agent.format_result(state)
    assert "## Your choices for this data" in report
    gapnet = next(a for a in state["assessments"] if a.name == "GapNet")
    assert gapnet.pathway == "inference"
    others = [a for a in state["assessments"] if a.name != "GapNet"]
    assert all(
        a.pathway in {"not_suitable", "unknown", "fine_tune"} for a in others
    )


def test_only_explicit_orders_run_models():
    orders = [
        "Run the top model with {}",
        "Now run toy-heat-1d.",
        "Please run it on my data.",
        "Can you run MolProp on caffeine?",
        "Recommend a model and run it.",
        "I have mols.smi. Predict their boiling points.",
    ]
    questions = [
        "How many weather models run on a CPU?",
        "Can I just perform inference, or do I need fine-tuning?",
        "Can I just run inference or do I need fine-tuning?",
        "Can I load my data and run inference?",
        "I want to run inference on my data; which models are my choices?",
        "Which models can predict the gap for my molecules?",
    ]
    assert all(_orders_run(text) for text in orders)
    assert not any(_orders_run(text) for text in questions)
    assert _FEASIBILITY.search(questions[1])


def property_tools():
    """A Nomad-style server with one property model that takes one SMILES."""

    async def get_model_card(tool_name: str) -> str:
        if tool_name == "quantaprop_qm":
            return cards.SERVED_PROPERTY_CARD
        raise ToolException(f"No model registered with the name '{tool_name}'.")

    async def quantaprop_qm(smi: str) -> str:
        atoms = sum(ch.isalpha() and ch.upper() in "CNO" for ch in smi)
        return json.dumps({
            "molecule": smi,
            "properties": {
                "gap": {"value": round(0.4 - 0.01 * atoms, 4), "units": None}
            },
        })

    return [
        StructuredTool(
            name="get_model_card",
            description="Return the Markdown model card for a registered SciFM tool.",
            args_schema={
                "type": "object",
                "properties": {"tool_name": {"type": "string"}},
                "required": ["tool_name"],
            },
            coroutine=get_model_card,
        ),
        StructuredTool(
            name="quantaprop_qm",
            description="Predicts a molecular property for one SMILES string.",
            args_schema=cards.SERVED_PROPERTY_SCHEMA,
            coroutine=quantaprop_qm,
        ),
    ]


def served_agent(make_agent, workspace):
    agent, llm = make_agent()
    agent.add_tool(property_tools())
    agent._tool_sources = {"get_model_card": "nomad", "quantaprop_qm": "nomad"}
    (workspace / "mols.smi").write_text(
        "CCO ethanol\nc1ccccc1 benzene\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C caffeine\n"
    )
    (workspace / "sol.csv").write_text(
        "smiles,solubility\nCCO,0.5\nc1ccccc1,-1.6\n"
    )
    return agent


def test_an_order_runs_the_fitting_model_on_every_molecule(
    make_agent, workspace
):
    agent = served_agent(make_agent, workspace)
    state = agent.invoke(
        "I have mols.smi. Predict the HOMO-LUMO gap of these molecules."
    )
    assert "run" in state["intents"]
    (best,) = state["assessments"]
    assert best.pathway == "inference"
    plan = state["run_plan"]
    assert [call["smi"] for call in plan.batch][:2] == ["CCO", "c1ccccc1"]
    (table,) = (workspace / plan.output_dir).glob("results-*.csv")
    rows = table.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("item,name,molecule,gap") and len(rows) == 4
    report = agent.format_result(state)
    assert "Ran 3 of 3 item(s) from `mols.smi`" in report
    assert "| `CCO` | ethanol |" in report
    # Caffeine has 14 heavy atoms; the card's training data stops at 9.
    assert "Caution: 1 of 3 molecules lies outside the training range" in report


def test_labeled_data_implies_the_target_and_nothing_runs(
    make_agent, workspace
):
    agent = served_agent(make_agent, workspace)
    state = agent.invoke(
        "I have sol.csv. Can I just run inference or do I need fine-tuning?"
    )
    assert "run" not in state["intents"] and state.get("run_plan") is None
    assert state["implied_targets"] == ["solubility"]
    (verdict,) = state["assessments"]
    assert verdict.pathway == "fine_tune"
    report = agent.format_result(state)
    assert (
        "I take it to be the target column(s) in `sol.csv`: solubility"
        in report
    )
    assert "**Answer: fine-tune.**" in report
    assert "1,200 examples" in report and "quanta-base-20M" in report


def test_feasibility_without_data_asks_for_the_file(make_agent, workspace):
    molecule_workspace(workspace)
    agent, _ = make_agent(catalog_paths=["cards"])
    state = agent.invoke("Can I just run inference or do I need fine-tuning?")
    assert state["assessments"] == [] and state.get("run_plan") is None
    assert "Name the data file" in agent.format_result(state)

    state = agent.invoke(
        "Can I just run inference to get the HOMO-LUMO gap, or do I need fine-tuning?"
    )
    best = state["assessments"][0]
    assert (best.name, best.pathway) == ("GapNet", "inference")
    report = agent.format_result(state)
    assert "no fine-tuning is needed for HOMO-LUMO gap" in report
    assert "Name your data file" in report and "from this data" not in report


def test_discovery_skips_the_llm_review_unless_asked(make_agent, workspace):
    def reviews(llm):
        return [
            call
            for call in llm.calls
            if "You read scientific foundation-model cards"
            in str(call[0].content)
        ]

    agent, llm = make_agent()
    agent.add_tool(property_tools())
    agent._tool_sources = {"get_model_card": "nomad", "quantaprop_qm": "nomad"}
    agent.invoke("How many models do I have?")
    assert len(agent.catalog) == 1 and not reviews(llm)

    # A stored card is reused, so remove it before discovering again.
    agent.catalog.remove("mcp:nomad:quantaprop_qm")
    agent, llm = make_agent(refine_discovered=True)
    agent.add_tool(property_tools())
    agent._tool_sources = {"get_model_card": "nomad", "quantaprop_qm": "nomad"}
    agent.invoke("How many models do I have?")
    assert len(reviews(llm)) == 1


def count_models(agent) -> str:
    return agent.format_result(agent.invoke("How many models do I have?"))


def test_an_empty_catalog_says_that_no_mcp_server_is_connected(make_agent):
    agent, _ = make_agent()
    report = count_models(agent)
    assert "You have **0** model(s) in the catalog" in report
    assert "No MCP server is connected" in report


def test_an_empty_catalog_names_a_server_without_model_cards(make_agent):
    agent, _ = make_agent()
    agent.add_tool([t for t in heat_tools() if t.name != "get_model_card"])
    agent._tool_sources = {"toy-heat": "lab", "search_pubchem": "lab"}
    report = count_models(agent)
    assert "Server 'lab' offers no `get_model_card` tool" in report


def test_discovery_reports_a_server_whose_card_requests_all_fail(make_agent):
    async def unreachable(tool_name: str) -> str:
        raise ToolException(
            f"Unable to download README.md for repository 'lab/{tool_name}'."
        )

    tools = heat_tools()
    tools[0] = StructuredTool(
        name="get_model_card",
        description=tools[0].description,
        args_schema=tools[0].args_schema,
        coroutine=unreachable,
    )
    agent, _ = make_agent()
    agent.add_tool(tools)
    agent._tool_sources = {t.name: "nomad" for t in tools}
    report = count_models(agent)
    assert (
        "`get_model_card` on server 'nomad' failed for all 2 tool(s)" in report
    )
    assert "Unable to download README.md" in report


def test_helper_tools_without_cards_raise_no_discovery_warning(make_agent):
    agent, _ = make_agent()
    tools = heat_tools()
    agent.add_tool(tools)
    agent._tool_sources = {t.name: "nomad" for t in tools}
    report = count_models(agent)
    assert "You have **1** model(s) in the catalog" in report
    assert "## Model discovery" not in report
    # The catalog table shows vocabulary labels, not concept ids.
    assert "| toy-heat | Heat transfer |" in report
    assert "## LLM" not in report


def test_an_answer_says_when_the_llm_call_failed(make_agent):
    def reject(schema):
        raise PermissionError("Error code: 401 - invalid x-api-key")

    agent, _ = make_agent(llm=RecordingChatModel(structured_factory=reject))
    report = count_models(agent)
    assert "You have **0** model(s) in the catalog" in report
    assert "## LLM" in report and "invalid x-api-key" in report
