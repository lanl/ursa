"""Cross-agent regression coverage for provider-valid message sequences.

Follow-up to the planner prefill bug: every request an agent sends must
be non-empty and must not end on an assistant turn, or providers without
assistant-message prefill support reject it with a 400.
"""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tests.agents.utils import (
    RecordingChatModel,
    assert_requests_provider_valid,
)
from ursa.agents.acquisition_agents import (
    ArxivAgent,
    OSTIAgent,
    WebSearchAgent,
)
from ursa.agents.chat_agent import BasicChatAgent, ChatAgent
from ursa.agents.deep_review_agent import DeepReviewAgent
from ursa.agents.execution_agent import ExecutionAgent
from ursa.agents.hypothesizer_agent import HypothesizerAgent
from ursa.agents.planning_agent import PlanningAgent
from ursa.agents.prompting_agent import PromptingAgent
from ursa.agents.rag_agent import RAGAgent
from ursa.agents.recall_agent import RecallAgent


class StubMemory:
    def __init__(self, memories):
        self._memories = memories

    def retrieve(self, query):
        return self._memories


def _plan_factory(schema):
    return schema(
        steps=[
            {
                "name": "Add numbers",
                "description": "Add 1 and 2.",
                "requires_code": False,
                "expected_outputs": ["sum"],
                "success_criteria": ["sum equals 3"],
            }
        ]
    )


async def test_basic_chat_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = BasicChatAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("What is URSA?"))

    assert_requests_provider_valid(llm.calls)


async def test_chat_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = ChatAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("What is URSA?"))

    assert_requests_provider_valid(llm.calls)


async def test_prompting_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = PromptingAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke(agent.format_query("Refine this prompt: hello"))

    assert_requests_provider_valid(llm.calls)


async def test_rag_agent_role_sequences(embedding_model, monkeypatch, tmp_path):
    (tmp_path / "database").mkdir()
    (tmp_path / "database" / "doc.pdf").write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(
        "ursa.agents.rag_agent.read_text_from_file",
        lambda path_name: "Entangled resonators enable sensitive detection.",
    )

    llm = RecordingChatModel()
    agent = RAGAgent(
        llm=llm,
        embedding=embedding_model,
        workspace=tmp_path,
        database_path="database",
        summaries_path="summaries",
        vectorstore_path="vectors",
        return_k=1,
        chunk_size=256,
        chunk_overlap=0,
    )

    query = "Explain entangled resonators."
    await agent.ainvoke({"context": query, "query": query})

    assert_requests_provider_valid(llm.calls)


async def test_recall_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = RecallAgent(
        llm=llm,
        memory=StubMemory(["remembered detail one", "remembered detail two"]),
        workspace=tmpdir,
    )

    await agent.ainvoke({"query": "what happened last run?"})

    assert_requests_provider_valid(llm.calls)


async def test_hypothesizer_agent_role_sequences(tmp_path):
    llm = RecordingChatModel()
    agent = HypothesizerAgent(llm=llm, workspace=tmp_path)

    await agent.ainvoke("Why is cooling energy rising in the data center?")

    assert_requests_provider_valid(llm.calls)


def _optimization_factory(schema):
    if schema.__name__ == "ProblemSpec":
        return schema(
            title="Test problem",
            description_nl="Minimize x.",
            decision_variables=[
                {
                    "name": "x",
                    "type": "continuous",
                    "domain": "x >= 0",
                    "description": "the variable",
                }
            ],
            parameters=[],
            objective={
                "sense": "minimize",
                "expression_nl": "x",
                "tags": ["linear"],
            },
            constraints=[],
            status="VERIFIED",
            notes={
                "verifier": "verified",
                "feasibility": "feasible",
                "user": "",
                "assumptions": "",
            },
        )
    if schema.__name__ == "SolverSpec":
        return schema(solver="ipopt", library="pyomo")
    raise AssertionError(f"unexpected structured schema: {schema.__name__}")


async def test_optimization_agent_role_sequences(tmpdir):
    pytest.importorskip("sympy")
    from ursa.agents.optimization_agent import OptimizationAgent

    llm = RecordingChatModel(structured_factory=_optimization_factory)
    agent = OptimizationAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke({"user_input": "Minimize x subject to x >= 0."})

    assert_requests_provider_valid(llm.calls)


@pytest.mark.parametrize(
    "acquisition_cls", [ArxivAgent, OSTIAgent, WebSearchAgent]
)
async def test_acquisition_agent_role_sequences(acquisition_cls, tmpdir):
    llm = RecordingChatModel()
    agent = acquisition_cls(
        llm=llm, workspace=tmpdir, download=False, summarize=True
    )
    (agent.database_path / "cached_item.txt").write_text(
        "Cached source text about mechanical resonators."
    )

    await agent.ainvoke({
        "query": "mechanical resonators",
        "context": "Summarize the cached source.",
    })

    assert_requests_provider_valid(llm.calls)


async def test_materials_project_agent_role_sequences(tmpdir, monkeypatch):
    pytest.importorskip("mp_api")
    from ursa.agents.mp_agent import MaterialsProjectAgent

    class FakeDoc:
        def __init__(self, material_id, metadata):
            self.material_id = material_id
            self._metadata = metadata

        def dict(self):
            return self._metadata

    class FakeSummary:
        def search(self, **kwargs):
            return [FakeDoc("mp-001", {"formula": "GaInO3", "band_gap": 1.8})]

    class FakeMaterials:
        def __init__(self):
            self.summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ursa.agents.mp_agent.MPRester", FakeMPRester)

    llm = RecordingChatModel()
    agent = MaterialsProjectAgent(
        llm=llm, max_results=1, summarize=True, workspace=tmpdir
    )

    await agent.ainvoke({
        "query": {
            "elements": ["Ga", "In"],
            "band_gap_min": 1.5,
            "band_gap_max": 2.5,
        },
        "context": "Highlight stability and band gaps.",
    })

    assert_requests_provider_valid(llm.calls)


async def test_planning_agent_role_sequences(tmpdir):
    llm = RecordingChatModel(structured_factory=_plan_factory)
    agent = PlanningAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke({"messages": [HumanMessage(content="make a plan")]})

    assert_requests_provider_valid(llm.calls)


@pytest.mark.xfail(
    reason=(
        "DeepReviewAgent appends a fresh SystemMessage per debate phase into "
        "the accumulated history, so requests from the second phase onward "
        "carry mid-conversation system messages, which langchain-anthropic "
        "rejects; see upstream issue #294"
    ),
    strict=True,
)
async def test_deep_review_agent_role_sequences(tmpdir):
    llm = RecordingChatModel()
    agent = DeepReviewAgent(llm=llm, workspace=tmpdir, max_iterations=1)

    await agent.ainvoke({"question": "How can cooling usage be reduced?"})

    assert_requests_provider_valid(llm.calls)


def test_invariant_rejects_assistant_final_requests():
    good = [SystemMessage(content="s"), HumanMessage(content="h")]
    bad = [SystemMessage(content="s"), AIMessage(content="a")]

    assert_requests_provider_valid([good])

    with pytest.raises(AssertionError, match="assistant turn"):
        assert_requests_provider_valid([good, bad])


def test_invariant_rejects_empty_requests():
    with pytest.raises(AssertionError, match="no LLM calls"):
        assert_requests_provider_valid([])

    with pytest.raises(AssertionError, match="empty message list"):
        assert_requests_provider_valid([[]])


def test_invariant_rejects_mid_conversation_system_messages():
    leading_prefix_ok = [
        SystemMessage(content="s1"),
        SystemMessage(content="s2"),
        HumanMessage(content="h"),
    ]
    mid_list_system = [
        SystemMessage(content="s1"),
        HumanMessage(content="h"),
        SystemMessage(content="s2"),
        HumanMessage(content="h2"),
    ]

    assert_requests_provider_valid([leading_prefix_ok])

    with pytest.raises(AssertionError, match="after the leading"):
        assert_requests_provider_valid([mid_list_system])


async def test_execution_agent_role_sequences(tmpdir):
    llm = RecordingChatModel(
        structured_factory=lambda schema: schema(
            is_complete=True, reason="Work complete."
        )
    )
    agent = ExecutionAgent(llm=llm, workspace=tmpdir)

    await agent.ainvoke("say hello")

    assert_requests_provider_valid(llm.calls)
