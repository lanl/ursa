from pathlib import Path

from langchain_core.messages import AIMessage

from ursa.workflows import CriticalMineralsWorkflow


class _FakePlanner:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return {"plan": "1) collect sources 2) synthesize findings"}


class _FakeExecutor:
    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return {"messages": [AIMessage(content="final minerals synthesis")]}


class _FakeAgent:
    def __init__(self, summary: str):
        self.summary = summary
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return {"final_summary": self.summary}


class _FakeRAG:
    def __init__(self):
        self.calls = []
        self.database_path = None

    def invoke(self, payload):
        self.calls.append(payload)
        return {"summary": "rag evidence summary"}


def test_critical_minerals_workflow_orchestrates_modules(tmp_path):
    planner = _FakePlanner()
    executor = _FakeExecutor()
    osti = _FakeAgent("osti signal")
    arxiv = _FakeAgent("arxiv signal")
    rag = _FakeRAG()
    materials = _FakeAgent("materials candidate list")

    workflow = CriticalMineralsWorkflow(
        planner=planner,
        executor=executor,
        acquisition_agents={"osti": osti, "arxiv": arxiv},
        rag_agent=rag,
        materials_agent=materials,
        workspace=tmp_path,
    )

    result = workflow.invoke(
        {
            "task": "Assess supply risk for scandium and gallium",
            "source_queries": {
                "osti": "scandium gallium supply chain policy",
                "arxiv": "materials substitution for gallium compounds",
            },
            "materials_query": {
                "elements": ["Sc", "Ga", "Al", "O"],
                "band_gap_min": 1.0,
                "band_gap_max": 5.0,
            },
        }
    )

    assert "Assess supply risk for scandium and gallium" in planner.prompts[0]
    assert osti.calls[0]["query"] == "scandium gallium supply chain policy"
    assert arxiv.calls[0]["query"] == "materials substitution for gallium compounds"
    assert rag.calls[0]["context"] == "Assess supply risk for scandium and gallium"
    assert executor.prompts
    assert "Acquisition (osti) summary" in executor.prompts[0]
    assert "Materials intelligence summary" in executor.prompts[0]
    assert result["final_summary"] == "final minerals synthesis"


def test_critical_minerals_workflow_sets_local_corpus_on_rag(tmp_path):
    planner = _FakePlanner()
    executor = _FakeExecutor()
    rag = _FakeRAG()
    workflow = CriticalMineralsWorkflow(
        planner=planner,
        executor=executor,
        rag_agent=rag,
        workspace=tmp_path,
    )

    corpus_path = "/tmp/cmm-corpus"
    workflow.invoke(
        {
            "task": "Summarize domestic rare earth supply constraints",
            "local_corpus_path": corpus_path,
        }
    )

    assert rag.database_path == Path(corpus_path)


def test_critical_minerals_workflow_runs_optimization(tmp_path):
    planner = _FakePlanner()
    executor = _FakeExecutor()
    workflow = CriticalMineralsWorkflow(
        planner=planner,
        executor=executor,
        workspace=tmp_path,
    )

    result = workflow.invoke(
        {
            "task": "Allocate cobalt supply for North America and Europe",
            "optimization_input": {
                "commodity": "CO",
                "demand": {"NA": 100, "EU": 80},
                "suppliers": [
                    {
                        "name": "US_mine",
                        "capacity": 120,
                        "unit_cost": 8.0,
                        "risk_score": 0.2,
                    },
                    {
                        "name": "Allied_import",
                        "capacity": 90,
                        "unit_cost": 9.2,
                        "risk_score": 0.1,
                    },
                ],
                "shipping_cost": {
                    "US_mine": {"NA": 1.0, "EU": 2.0},
                    "Allied_import": {"NA": 1.4, "EU": 1.2},
                },
                "risk_weight": 2.0,
                "max_supplier_share": 0.8,
            },
        }
    )

    optimization = result["optimization"]
    assert optimization is not None
    assert "objective_value" in optimization
    assert "allocations" in optimization
    assert "constraint_residuals" in optimization
    assert "sensitivity_summary" in optimization
    assert "Optimization output (deterministic JSON)" in executor.prompts[0]
