from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ursa.workflows import (
    HypothesisOrchestratorWorkflow,
    SpawnInvestigatorInput,
)
from ursa.workflows.hypothesis_orchestrator import (
    _InvestigatorRegistry,
    _slug,
    _timestamp,
)

ARTIFACT = """# Hypothesis Space

### H1: Expected improvement is best

- detail one

### H2: Upper confidence bound is best

- detail two

### H3: Probability of improvement is best
"""

AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["unused"])


class StubWorkflow(HypothesisOrchestratorWorkflow):
    """Deterministic subclass: fixed hypotheses + a recording orchestrator."""

    def __init__(self, **kwargs: Any):
        super().__init__(fake_llm(), **kwargs)
        self.spawn_calls: list[tuple[str, str]] = []

    def build_hypotheses(self, query, context, config):
        self._last_hypothesis_space = ARTIFACT
        return self._parse_hypotheses(ARTIFACT, query)

    def _build_orchestrator(self, registry):
        tools = self._make_spawn_tools(registry)
        spawn = next(t for t in tools if t.name == "spawn_investigator")
        listing = next(t for t in tools if t.name == "list_investigators")
        workflow = self

        class _FakeOrchestrator:
            def invoke(self, prompt, config=None):
                # Investigation phase: one call per hypothesis label.
                for label in ("H1", "H2", "H3"):
                    workflow.spawn_calls.append((label, "investigate"))
                    spawn.func(hypothesis_label=label, task="investigate")
                # Review phase: reuse the SAME labels.
                for label in ("H1", "H2", "H3"):
                    workflow.spawn_calls.append((label, "review"))
                    spawn.func(hypothesis_label=label, task="review")
                return {"messages": [f"listing:\n{listing.func()}"]}

        return _FakeOrchestrator()


class RecordingInvestigator:
    def __init__(self):
        self.tasks: list[str] = []

    def invoke(self, task):
        self.tasks.append(task)
        return {"messages": [f"done: {task}"]}


def test_parse_hypotheses_from_artifact():
    wf = StubWorkflow()
    hs = wf._parse_hypotheses(ARTIFACT, "which acquisition function")
    assert [h.index for h in hs] == [1, 2, 3]
    assert hs[0].statement == "Expected improvement is best"


def test_max_hypotheses_caps_parse():
    wf = StubWorkflow(max_hypotheses=2)
    hs = wf._parse_hypotheses(ARTIFACT, "q")
    assert len(hs) == 2


def test_slug_and_timestamp_format():
    assert _slug("Expected Improvement is BEST!!") == "expected_improvement_is_best"
    assert _slug("") == "hypothesis"
    stamp = _timestamp()
    assert re.fullmatch(r"\d{8}_\d{6}", stamp)


def test_registry_mints_valid_persistent_names(monkeypatch):
    created: list[dict[str, Any]] = []

    class FakeExec:
        def __init__(self, llm, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    reg = _InvestigatorRegistry(
        fake_llm(), workspace=None, group=None, run_stamp="20240101_120000"
    )
    a1 = reg.get_or_create("H1")
    a1_again = reg.get_or_create("h1")  # case-insensitive reuse
    a2 = reg.get_or_create("H2")

    assert a1 is a1_again  # same persistent agent reused
    assert a1 is not a2
    names = reg.identities
    assert set(names) == {"h1", "h2"}
    for name in names.values():
        assert AGENT_NAME_RE.fullmatch(name)
        assert name.endswith("20240101_120000")
    # agent_name persistence + timestamp were passed to ExecutionAgent
    assert created[0]["agent_name"].endswith("20240101_120000")


def test_registry_scopes_workspace_and_group(monkeypatch, tmp_path):
    created: list[dict[str, Any]] = []

    class FakeExec:
        def __init__(self, llm, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    reg = _InvestigatorRegistry(
        fake_llm(),
        workspace=tmp_path,
        group="mygroup",
        run_stamp="20240101_120000",
    )
    reg.get_or_create("H1")
    kwargs = created[0]
    assert kwargs["group"] == "mygroup"
    assert Path(kwargs["workspace"]).parent == tmp_path
    assert Path(kwargs["workspace"]).name == kwargs["agent_name"]


def test_spawn_investigator_input_schema():
    model = SpawnInvestigatorInput(hypothesis_label="H1", task="do it")
    assert model.hypothesis_label == "H1"
    assert model.task == "do it"


def test_end_to_end_invoke_reuses_labels(monkeypatch):
    investigators: dict[str, RecordingInvestigator] = {}

    class FakeExec:
        def __init__(self, llm, **kwargs):
            self.name = kwargs.get("agent_name", "orch")

    def fake_get_or_create(self, label):
        key = label.strip().lower()
        if key not in investigators:
            investigators[key] = RecordingInvestigator()
            self._names[key] = f"hyp_{key}_20240101_120000"
        return investigators[key]

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    monkeypatch.setattr(
        _InvestigatorRegistry, "get_or_create", fake_get_or_create
    )

    wf = StubWorkflow()
    result = wf.invoke("which acquisition function is best?")

    # Three hypotheses parsed.
    assert len(result["hypotheses"]) == 3
    # One persistent investigator per label (no duplicates despite 2 phases).
    assert set(investigators) == {"h1", "h2", "h3"}
    # Each investigator was called twice: investigate then review.
    for inv in investigators.values():
        assert inv.tasks == ["investigate", "review"]
    # Identities surfaced and recorded.
    assert set(result["investigator_identities"]) == {"h1", "h2", "h3"}
    # Six spawn calls total, labels reused across phases.
    assert wf.spawn_calls == [
        ("H1", "investigate"),
        ("H2", "investigate"),
        ("H3", "investigate"),
        ("H1", "review"),
        ("H2", "review"),
        ("H3", "review"),
    ]
    assert "listing:" in result["final"]


def test_string_and_mapping_inputs_normalize():
    wf = StubWorkflow()
    assert wf._normalize_inputs("hi") == {"query": "hi"}
    assert wf._normalize_inputs({"query": "hi"}) == {"query": "hi"}
