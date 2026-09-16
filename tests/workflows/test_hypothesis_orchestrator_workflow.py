from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ursa import security
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


@pytest.fixture(autouse=True)
def _isolate_ursa_cache(monkeypatch, tmp_path):
    """Keep visualization run artifacts out of the developer's real cache.

    The workflow records environment runs under `URSA_CACHE_DIR` by default, so
    without this fixture every test invocation would litter `~/.cache/ursa`.
    """
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")


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

    def invoke(self, task, config=None):
        self.tasks.append(task)
        return {"messages": [f"done: {task}"]}

    async def ainvoke(self, task, config=None):
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
    assert (
        _slug("Expected Improvement is BEST!!")
        == "expected_improvement_is_best"
    )
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


def _recorded_events(cache_root: Path) -> list[dict[str, Any]]:
    """Read every event written by the run under a monkeypatched cache root."""
    runs = cache_root / "ursa" / "default" / "environment_runs"
    run_dirs = [p for p in runs.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"expected exactly one run dir, got {run_dirs}"
    events_file = run_dirs[0] / "events.jsonl"
    return [
        json.loads(line)
        for line in events_file.read_text().splitlines()
        if line.strip()
    ]


def test_visualization_records_delegation_events(monkeypatch, tmp_path):
    """The run must stream topology + per-investigator delegation events."""
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
    wf.invoke("which acquisition function is best?")

    events = _recorded_events(tmp_path)
    types = [e.get("event_type") for e in events]

    # Topology is declared up front so the dashboard can draw the graph
    # before any slow investigation work begins.
    assert "topology_declared" in types
    assert "orchestration_started" in types

    # Every hypothesis produced a started/completed pair for both phases
    # (3 hypotheses x 2 phases = 6 delegations).
    assert types.count("delegation_started") == 6
    assert types.count("delegation_completed") == 6
    assert "delegation_failed" not in types

    targets = {
        e["target"]["name"]
        for e in events
        if e.get("event_type") == "delegation_started"
    }
    assert targets == {"H1", "H2", "H3"}

    completed = [
        e for e in events if e.get("event_type") == "delegation_completed"
    ]
    # Custom emit kwargs are nested under "payload" by the recorder.
    assert all("elapsed_seconds" in e["payload"] for e in completed)
    assert {e["payload"]["hypothesis_label"] for e in completed} == {
        "H1",
        "H2",
        "H3",
    }


def test_visualization_records_failed_delegation(monkeypatch, tmp_path):
    """An investigator error is recorded, not swallowed silently."""

    class FakeExec:
        def __init__(self, llm, **kwargs):
            self.name = kwargs.get("agent_name", "orch")

    class Boom:
        def invoke(self, task, config=None):
            raise RuntimeError("investigator exploded")

    def fake_get_or_create(self, label):
        self._names[label.strip().lower()] = "hyp_x_20240101_120000"
        return Boom()

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    monkeypatch.setattr(
        _InvestigatorRegistry, "get_or_create", fake_get_or_create
    )

    wf = StubWorkflow()
    # The error is recorded and then re-raised so the caller/LLM sees it.
    with pytest.raises(RuntimeError, match="investigator exploded"):
        wf.invoke("q")

    events = _recorded_events(tmp_path)
    failed = [e for e in events if e.get("event_type") == "delegation_failed"]
    assert failed, "expected delegation_failed events to be recorded"
    assert failed[0]["level"] == "error"
    assert "investigator exploded" in failed[0]["payload"]["error"]


def test_visualize_false_writes_no_run_directory(monkeypatch, tmp_path):
    """Opting out of visualization must not touch the cache at all."""

    class FakeExec:
        def __init__(self, llm, **kwargs):
            self.name = kwargs.get("agent_name", "orch")

    def fake_get_or_create(self, label):
        self._names[label.strip().lower()] = "hyp_x_20240101_120000"
        return RecordingInvestigator()

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    monkeypatch.setattr(
        _InvestigatorRegistry, "get_or_create", fake_get_or_create
    )

    wf = StubWorkflow(visualize=False)
    wf.invoke("q")

    assert wf.orchestrator_run_id is None
    assert not (tmp_path / "ursa" / "default" / "environment_runs").exists()


def test_format_result_returns_only_final_answer():
    """The dashboard must get the final write-up, not the whole state."""
    wf = StubWorkflow()
    state = {
        "final": "  the verdict  ",
        "query": "q",
        "hypotheses": [{"index": 1}],
        "hypothesis_space_markdown": "# noisy artifact",
        "investigator_identities": {"h1": "hyp_h1_20240101_120000"},
        "orchestrator_result": {"messages": ["bulky"]},
    }
    assert wf.format_result(state) == "the verdict"


def test_format_result_falls_back_to_summary():
    wf = StubWorkflow()
    assert wf.format_result({"summary": "short summary"}) == "short summary"
    # An empty `final` should not mask a usable summary.
    assert (
        wf.format_result({"final": "", "summary": "short summary"})
        == "short summary"
    )


def test_format_result_raises_without_any_response():
    wf = StubWorkflow()
    with pytest.raises(ValueError, match="without a response"):
        wf.format_result({})


def test_format_result_is_used_by_the_dashboard_formatter(monkeypatch):
    """Regression: format_result must not raise, or the dashboard dumps state.

    `ursa_dashboard.adapters._format_agent_result` swallows ValueError/KeyError
    and falls back to `str(result)`, which is exactly the messy full-state dump
    this formatting is meant to avoid.
    """
    from ursa_dashboard.adapters import _format_agent_result

    wf = StubWorkflow()
    state = {"final": "the verdict", "hypothesis_space_markdown": "# noisy"}
    shown = _format_agent_result(wf, state)
    assert shown == "the verdict"
    assert "noisy" not in shown


def _registry_with_artifacts(tmp_path, monkeypatch, labels=("H1", "H2")):
    """Registry whose investigators each own a real workspace artifact.

    ExecutionAgent is stubbed out (the fake LLM cannot bind_tools); only the
    registry's workspace/symlink bookkeeping is under test here.
    """

    class FakeExec:
        def __init__(self, llm, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    run_ws = tmp_path / "run"
    run_ws.mkdir(parents=True, exist_ok=True)
    registry = _InvestigatorRegistry(
        fake_llm(),
        workspace=run_ws,
        group=None,
        run_stamp="20250101_000000",
    )
    for label in labels:
        registry.get_or_create(label)
        ws = registry.workspace_for(label)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "evidence.md").write_text(f"EVIDENCE FROM {label}\n")
    return registry


def test_link_peers_exposes_other_investigator_workspaces(
    tmp_path, monkeypatch
):
    """The review phase must be able to read peers' real artifacts, not just the
    summary text pasted into the prompt."""
    registry = _registry_with_artifacts(tmp_path, monkeypatch)
    exposed = registry.link_peers("H1")

    assert exposed == {"h2": "peers/h2"}
    own = registry.workspace_for("H1")
    peer_file = own / exposed["h2"] / "evidence.md"
    assert peer_file.read_text().strip() == "EVIDENCE FROM H2"
    # An investigator never gets a link to itself.
    assert not (own / "peers" / "h1").exists()


def test_peer_workspaces_are_read_only(tmp_path, monkeypatch):
    """Writes through a peer symlink must resolve outside the agent's own
    workspace so the write-tool sandbox refuses them."""
    registry = _registry_with_artifacts(tmp_path, monkeypatch)
    exposed = registry.link_peers("H1")
    own = registry.workspace_for("H1").resolve()

    target = (own / exposed["h2"] / "evidence.md").resolve()
    assert not target.is_relative_to(own)


def test_link_peers_is_idempotent_and_not_cyclic_for_listings(
    tmp_path, monkeypatch
):
    registry = _registry_with_artifacts(
        tmp_path, monkeypatch, labels=("H1", "H2", "H3")
    )
    for _ in range(3):
        exposed = registry.link_peers("H1")
    assert exposed == {"h2": "peers/h2", "h3": "peers/h3"}

    registry.link_peers("H2")
    own = registry.workspace_for("H1")
    # Mutual links create a cycle on disk, but glob must not walk into it.
    entries = {str(p.relative_to(own)) for p in own.glob("**/*")}
    assert entries == {"evidence.md", "peers", "peers/h2", "peers/h3"}


def test_link_peers_noop_without_workspace(monkeypatch):
    class FakeExec:
        def __init__(self, llm, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        "ursa.workflows.hypothesis_orchestrator.ExecutionAgent", FakeExec
    )
    registry = _InvestigatorRegistry(
        fake_llm(), workspace=None, group=None, run_stamp="20250101_000000"
    )
    registry.get_or_create("H1")
    registry.get_or_create("H2")
    assert registry.link_peers("H1") == {}
    assert registry.workspace_for("H1") is None


def test_peer_access_note_states_read_only_and_paths():
    note = HypothesisOrchestratorWorkflow._peer_access_note({"h2": "peers/h2"})
    assert "READ-ONLY" in note
    assert "peers/h2/" in note
    assert "H2" in note
    # Must warn about the two non-obvious behaviors.
    assert "list_workspace_files" in note
    assert "grep -r" in note


def test_peer_access_note_empty_without_peers():
    assert HypothesisOrchestratorWorkflow._peer_access_note({}) == ""


def test_with_peer_access_appends_note_and_tolerates_stub_registry(
    tmp_path, monkeypatch
):
    wf = StubWorkflow()
    registry = _registry_with_artifacts(tmp_path, monkeypatch)
    task = wf._with_peer_access(registry, "H1", "Review the others.")
    assert task.startswith("Review the others.")
    assert "READ-ONLY" in task

    class _NoPeerRegistry:
        pass

    # Registries without peer support must degrade to the original task.
    assert wf._with_peer_access(_NoPeerRegistry(), "H1", "task") == "task"

    class _BrokenRegistry:
        def link_peers(self, label):
            raise OSError("no symlink privileges")

    assert wf._with_peer_access(_BrokenRegistry(), "H1", "task") == "task"


def test_orchestrator_prompt_instructs_read_only_peer_review():
    from ursa.workflows.hypothesis_orchestrator import _ORCHESTRATOR_PROMPT

    assert "peers/<label>/" in _ORCHESTRATOR_PROMPT
    assert "READ-ONLY" in _ORCHESTRATOR_PROMPT
