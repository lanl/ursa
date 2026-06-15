import pytest

from ursa.agents.hypothesizer_agent import HypothesizerAgent


def test_hypothesizer_extracts_text_from_structured_response_blocks(
    chat_model,
    tmp_path,
) -> None:
    agent = HypothesizerAgent(llm=chat_model, workspace=tmp_path)
    structured_content = [
        {
            "id": "rs_123",
            "summary": [],
            "type": "reasoning",
            "content": [],
        },
        {
            "type": "text",
            "text": "# Hypothesis Space\\n\\n### H1: Leaky sprinkler",
        },
    ]
    expected = "# Hypothesis Space\n\n### H1: Leaky sprinkler"

    polluted_string = repr(structured_content)

    assert HypothesizerAgent._response_text(structured_content) == expected
    assert HypothesizerAgent._response_text(polluted_string) == expected
    assert (
        HypothesizerAgent._response_text(
            "# Hypothesis Space\\n\\n### H1: Leaky sprinkler"
        )
        == expected
    )
    assert (
        agent.format_result({"hypothesis_space_markdown": structured_content})
        == expected
    )


@pytest.mark.asyncio
async def test_hypothesizer_agent_offloads_hypothesis_space(
    chat_model,
    tmp_path,
) -> None:
    agent = HypothesizerAgent(llm=chat_model, workspace=tmp_path)

    result = await agent.ainvoke(
        "Why is cooling energy rising in the edge data center?"
    )

    assert result["experience_filename"] == "hypothesis_space.md"
    assert result["summary"]
    assert result["hypothesis_space_markdown"].startswith("#")
    assert "### H1" in result["hypothesis_space_markdown"]
    assert agent.format_result(result) == result["hypothesis_space_markdown"]
    assert result["revision_history"]

    artifact = agent.den / "experiences" / "hypothesis_space.md"
    assert artifact.exists()
    artifact_text = artifact.read_text(encoding="utf-8")
    assert "Why is cooling energy rising" in artifact_text
    assert "URSA hypothesis metadata" not in artifact_text
    assert "Experience artifact:" not in artifact_text
    assert "Last updated:" not in artifact_text


def test_hypothesizer_sanitizes_polluted_existing_artifact(
    chat_model,
    tmp_path,
) -> None:
    agent = HypothesizerAgent(llm=chat_model, workspace=tmp_path)
    artifact = agent.den / "experiences" / "hypothesis_space.md"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        repr([
            {"type": "reasoning", "content": []},
            {"type": "text", "text": "# Hypothesis Space\n\n### H1: Clean"},
        ]),
        encoding="utf-8",
    )

    assert agent._read_existing_hypothesis_space("hypothesis_space.md") == (
        "# Hypothesis Space\n\n### H1: Clean"
    )


@pytest.mark.asyncio
async def test_hypothesizer_agent_updates_existing_experience(
    chat_model,
    tmp_path,
) -> None:
    agent = HypothesizerAgent(llm=chat_model, workspace=tmp_path)

    first = await agent.ainvoke(
        "Initial question about an unexpected sensor shift"
    )
    second = await agent.ainvoke({
        "query": first["query"],
        "new_information": "New evidence: the shift began immediately after a firmware update.",
        "context": "Execution agent inspected logs and found no ambient temperature change.",
        "revision_history": first["revision_history"],
    })

    artifact = agent.den / "experiences" / "hypothesis_space.md"
    artifact_text = artifact.read_text(encoding="utf-8")

    follow_up = agent.format_query(
        "Additional evidence: the vendor released a calibration note.",
        state=second,
    )

    assert second["summary"]
    assert len(second["revision_history"]) == 2
    assert follow_up["query"] == first["query"]
    assert follow_up["new_information"] == (
        "Additional evidence: the vendor released a calibration note."
    )
    assert follow_up["revision_history"] == second["revision_history"]
    assert "firmware update" in artifact_text
    assert "ambient temperature" in artifact_text
