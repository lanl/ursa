"""Regression coverage for settings passed to lazily created workflow agents."""

import pytest
from langchain_core.tools import tool

from ursa import security
from ursa.workflows import (
    Hypothesis,
    HypothesisOrchestratorWorkflow,
    HypothesisSymposiumWorkflow,
)
from ursa_dashboard.worker_main import _agent_with_tools_targets


@tool
def external_lookup(query: str) -> str:
    """Look up external evidence."""
    return query


@tool
def search_notes(query: str) -> str:
    """Search local research notes."""
    return query


@pytest.fixture(
    params=[HypothesisSymposiumWorkflow, HypothesisOrchestratorWorkflow]
)
def workflow_class(request, monkeypatch, tmp_path, chat_model):
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    security.group_root_dir("research").mkdir(parents=True)
    security.group_config_file("research").write_text(
        "allowed_base_urls: ['https://models.example.test/v1']\n"
    )
    monkeypatch.setitem(
        chat_model.__dict__, "base_url", "https://models.example.test/v1"
    )
    return request.param


def test_hypothesis_artifacts_stay_in_each_workspace(
    workflow_class, chat_model, tmp_path
):
    workflows = [
        workflow_class(
            chat_model,
            workspace=tmp_path / name,
            group="research",
            enable_metrics=False,
        )
        for name in ("first", "second")
    ]
    try:
        first, second = [wf.hypothesizer for wf in workflows]
        filename = first.experience_filename
        path = first._write_hypothesis_space(filename, "first run's evidence")
        assert path.is_relative_to(tmp_path / "first")
        assert first.group == second.group == "research"
        assert second._read_existing_hypothesis_space(filename) == ""
    finally:
        for wf in workflows:
            wf.hypothesizer.close()


def test_hypothesis_parsing_preserves_details_limits_and_fallback(
    workflow_class, chat_model, tmp_path
):
    wf = workflow_class(
        chat_model,
        workspace=tmp_path,
        group="research",
        max_hypotheses=1,
        enable_metrics=False,
    )
    try:
        artifact = "### H1: First cause\nEvidence\n\n### H2: Second cause\nMore"
        assert wf._parse_hypotheses(artifact, "why?") == [
            Hypothesis(1, "First cause", "Evidence")
        ]
        assert wf._parse_hypotheses("Unstructured notes", "why?") == [
            Hypothesis(1, "why?", "Unstructured notes")
        ]
        assert (
            wf._artifact_text({
                "hypothesis_space_markdown": artifact,
                "summary": "Short summary",
            })
            == artifact
        )
        assert (
            wf._artifact_text({"summary": "Short summary"}) == "Short summary"
        )
    finally:
        wf.hypothesizer.close()


async def test_lazy_children_receive_common_settings_and_mcp_tools(
    workflow_class, chat_model, embedding_model, tmp_path, monkeypatch
):
    rag_calls = []

    def build_rag_tools(**kwargs):
        rag_calls.append(kwargs)
        return [search_notes]

    async def discover_tools(client):
        return [external_lookup], {external_lookup.name: "evidence"}

    monkeypatch.setattr("ursa.rag.tools.build_rag_tools", build_rag_tools)
    monkeypatch.setattr(
        "ursa.agents.base.load_mcp_tools_with_sources", discover_tools
    )
    wf = workflow_class(
        chat_model,
        workspace=tmp_path / "run",
        group="research",
        use_web=True,
        rag_tools=["notes"],
        rag_tool_embedding=embedding_model,
        rag_tool_return_k=3,
        tokens_before_summarize=1234,
        messages_to_keep=7,
        enable_metrics=False,
    )
    assert _agent_with_tools_targets(wf) == [wf]
    sources = await wf.add_mcp_tools(object())
    assert sources == {"external_lookup": "evidence"}

    if isinstance(wf, HypothesisSymposiumWorkflow):
        symposium = wf._get_symposium([Hypothesis(1, "A cause")])
        children = [*symposium.members.values(), symposium.organizer]
    else:
        registry = wf._build_registry()
        children = [
            wf._build_orchestrator(registry),
            registry.get_or_create("H1"),
        ]
    try:
        assert wf.hypothesizer.tokens_before_summarize == 1234
        for agent in children:
            assert agent.tokens_before_summarize == 1234
            assert agent.messages_to_keep == 7
            assert agent.tools["external_lookup"] is external_lookup
            assert agent.tools["search_notes"] is search_notes
            assert any(
                "search" in name and name != "search_notes"
                for name in agent.tools
            )
        assert len(rag_calls) == len(children)
        assert all(call["embedding"] is embedding_model for call in rag_calls)
        assert all(
            call["group"] == "research" and call["return_k"] == 3
            for call in rag_calls
        )
    finally:
        for agent in [wf.hypothesizer, *children]:
            await agent.aclose()
            agent.close()
