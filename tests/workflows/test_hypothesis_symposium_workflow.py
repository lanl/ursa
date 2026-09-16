from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableConfig

from tests.conftest import FakeChatModel
from ursa import security
from ursa.workflows import (
    Hypothesis,
    HypothesisInvestigation,
    HypothesisSymposiumWorkflow,
)

ARTIFACT = """# Hypothesis Space

### H1: The pump is failing

- **Relative likelihood:** 0.5
- detail one

### H2: The sensor is miscalibrated

- **Relative likelihood:** 0.3
- detail two

### H3: The controller has a bug

- **Relative likelihood:** 0.2
"""


@pytest.fixture(autouse=True)
def _isolate_ursa_cache(monkeypatch, tmp_path):
    """Keep visualization run artifacts out of the developer's real cache.

    The symposium stage records environment runs under `URSA_CACHE_DIR` by
    default, so without this fixture tests would litter `~/.cache/ursa`.
    """
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "ursa")


def fake_llm() -> FakeListChatModel:
    return FakeListChatModel(responses=["unused"])


def tool_llm() -> FakeChatModel:
    """A fake chat model that supports bind_tools, needed when a real
    ChatAgent symposium member is actually constructed."""
    return FakeChatModel(messages=iter([]))


class StubWorkflow(HypothesisSymposiumWorkflow):
    """Deterministic subclass: overrides every scientific hook.

    Demonstrates the hook seam (a Plan D layer would override the same hooks)
    while keeping the fan-out / quarantine / fan-in plumbing under test.
    """

    def __init__(self, *, boom_indices: set[int] | None = None, **kwargs: Any):
        self.boom_indices = boom_indices or set()
        self.investigated: list[int] = []
        self.symposium_calls = 0
        super().__init__(fake_llm(), **kwargs)

    def build_hypotheses(self, state, config) -> list[Hypothesis]:
        state["hypothesis_space_markdown"] = ARTIFACT
        return self._parse_hypotheses(ARTIFACT, state.get("query", ""))

    def investigate_hypothesis(
        self, hypothesis: Hypothesis, query: str, config: RunnableConfig
    ) -> HypothesisInvestigation:
        self.investigated.append(hypothesis.index)
        if hypothesis.index in self.boom_indices:
            raise RuntimeError(f"branch {hypothesis.index} blew up")
        return HypothesisInvestigation(
            index=hypothesis.index,
            statement=hypothesis.statement,
            findings=f"evidence for H{hypothesis.index}",
            ok=True,
        )

    async def run_symposium(self, state, config) -> dict[str, Any]:
        self.symposium_calls += 1
        return {"final": f"verdict::{state.get('evidence_digest', '')[:20]}"}

    async def update_hypothesis_space(self, state, config) -> dict[str, Any]:
        # Confirm the final update sees ALL new information.
        info = self._final_new_information(state)
        assert "investigation" in info
        assert "verdict::" in info or state.get("synthesis")
        return {
            "hypothesis_space_markdown": ARTIFACT + "\n<updated>",
            "summary": "updated",
        }


def test_parses_and_fans_out_over_every_hypothesis():
    wf = StubWorkflow()
    result = wf.invoke("why is the plant offline?")

    assert sorted(wf.investigated) == [1, 2, 3]
    investigations = result["investigations"]
    assert len(investigations) == 3
    assert {i["index"] for i in investigations} == {1, 2, 3}
    assert result["failures"] == []


def test_converge_digest_orders_and_includes_all_branches():
    wf = StubWorkflow()
    result = wf.invoke("why?")
    digest = result["evidence_digest"]
    # All three hypotheses appear, ordered by index.
    assert (
        digest.index("### H1") < digest.index("### H2") < digest.index("### H3")
    )
    assert "evidence for H2" in digest


def test_failing_branch_is_quarantined_not_fatal():
    wf = StubWorkflow(boom_indices={2})
    result = wf.invoke("why?")

    # The run completes; the two healthy branches survive.
    ok = {i["index"] for i in result["investigations"]}
    assert ok == {1, 3}
    failures = result["failures"]
    assert len(failures) == 1
    assert failures[0]["index"] == 2
    assert failures[0]["ok"] is False
    assert "RuntimeError" in failures[0]["error"]

    # The digest records the failed branch and a summary note.
    digest = result["evidence_digest"]
    assert "### H2 (FAILED)" in digest
    assert "1 of 3 investigation branch(es) failed" in digest


def test_pipeline_runs_symposium_and_updates_space_with_all_information():
    wf = StubWorkflow()
    result = wf.invoke("why?")

    assert wf.symposium_calls == 1
    assert result["symposium_result"]["final"].startswith("verdict::")
    assert result["synthesis"].startswith("verdict::")
    assert result["hypothesis_space_markdown"].endswith("<updated>")
    assert result["summary"] == "updated"


@pytest.mark.asyncio
async def test_async_path_uses_ainvestigate_and_completes():
    wf = StubWorkflow(boom_indices={3})
    result = await wf.ainvoke("why?")
    assert {i["index"] for i in result["investigations"]} == {1, 2}
    assert {f["index"] for f in result["failures"]} == {3}
    assert result["synthesis"].startswith("verdict::")


def test_no_hypotheses_routes_straight_to_converge():
    class EmptyWorkflow(StubWorkflow):
        def build_hypotheses(self, state, config):
            return []

        def _parse_hypotheses(self, artifact, query):  # pragma: no cover
            return []

    wf = EmptyWorkflow()
    # Empty fan-out means no branches; converge/symposium/update still run.
    result = wf.invoke("why?")
    assert result["investigations"] == []
    assert wf.symposium_calls == 1
    assert result["hypothesis_space_markdown"].endswith("<updated>")


def test_default_symposium_builds_one_member_per_hypothesis():
    """Design A: each hypothesis becomes a dedicated symposium member whose
    role/prompt assign that hypothesis as its expertise."""
    wf = HypothesisSymposiumWorkflow(tool_llm())
    hypotheses = wf._parse_hypotheses(ARTIFACT, "why?")
    assert len(hypotheses) == 3

    symposium = wf._build_default_symposium(hypotheses)
    members = list(symposium.config.members)
    assert len(members) == 3
    # Names are stable, unique, and filesystem/tool-safe.
    names = [m.name for m in members]
    assert names[0].startswith("h1_")
    assert len(set(names)) == 3
    for m in names:
        assert all(c.isalnum() or c in "._-" for c in m)
    # Each member's role + prompt carry its own hypothesis and instruct it to
    # investigate only that one.
    for idx, member in enumerate(members, start=1):
        assert f"H{idx}" in member.role
        assert member.prompt is not None
        assert f"H{idx}" in member.prompt
        assert "other members own those" in member.prompt
        assert member.reviewer is True


def test_default_symposium_falls_back_to_generalist_without_hypotheses():
    wf = HypothesisSymposiumWorkflow(tool_llm())
    symposium = wf._build_default_symposium([])
    members = list(symposium.config.members)
    assert len(members) == 1
    assert members[0].name == "generalist"


def test_workspace_is_threaded_into_default_symposium(tmp_path):
    """The dashboard/caller-provided workspace must reach the symposium members
    instead of falling back to the shared cache directory."""
    ws = tmp_path / "run_ws"
    wf = HypothesisSymposiumWorkflow(tool_llm(), workspace=str(ws))
    assert wf.workspace == ws

    hypotheses = wf._parse_hypotheses(ARTIFACT, "why?")
    symposium = wf._build_default_symposium(hypotheses)
    # The symposium workspace is anchored under the provided run workspace, not
    # under ~/.cache/ursa/.../workspaces/.
    assert Path(symposium.workspace) == ws
    assert ".cache" not in str(symposium.workspace)


def test_workspace_absent_uses_environment_default():
    wf = HypothesisSymposiumWorkflow(tool_llm())
    assert wf.workspace is None
    symposium = wf._build_default_symposium(
        wf._parse_hypotheses(ARTIFACT, "why?")
    )
    # No explicit workspace -> BaseEnvironment default path is used.
    assert symposium.workspace is not None


def test_format_result_prefers_detailed_symposium_answer():
    """Only the final write-up is returned, not the whole bulky state."""
    wf = StubWorkflow()
    state = {
        "symposium_result": {"final": "  detailed verdict  "},
        "summary": "short summary",
        "hypothesis_space_markdown": "# noisy artifact",
    }
    assert wf.format_result(state) == "detailed verdict"


def test_format_result_falls_back_to_summary():
    wf = StubWorkflow()
    # No final text at all.
    assert wf.format_result({"summary": "short summary"}) == "short summary"
    # Present-but-empty final must not mask the summary.
    assert (
        wf.format_result({
            "symposium_result": {"final": ""},
            "summary": "short summary",
        })
        == "short summary"
    )
    # A non-mapping symposium_result must not raise AttributeError.
    assert (
        wf.format_result({"symposium_result": None, "summary": "short summary"})
        == "short summary"
    )


def test_format_result_raises_without_any_response():
    wf = StubWorkflow()
    with pytest.raises(ValueError, match="without a response"):
        wf.format_result({})


def test_default_symposium_uses_short_display_name():
    """The symposium environment name surfaces on the dashboard Runs page, so it
    must stay short/readable instead of being derived from the class name."""
    wf = HypothesisSymposiumWorkflow(tool_llm())
    assert wf.name == "hypothesis_symposium"
    symposium = wf._build_default_symposium(
        wf._parse_hypotheses(ARTIFACT, "why?")
    )
    assert symposium.name == "hypothesis_symposium"


def test_symposium_display_name_is_overridable():
    wf = HypothesisSymposiumWorkflow(tool_llm(), name="protein_folding")
    symposium = wf._build_default_symposium(
        wf._parse_hypotheses(ARTIFACT, "why?")
    )
    assert symposium.name == "protein_folding"
