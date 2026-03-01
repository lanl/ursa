from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from ursa.agents.summarizing_agent import SummarizingAgent
from ursa.prompt_library.summarizing_prompts import (
    SYSTEM_MAP_PROMPT,
    SYSTEM_REDUCE_PROMPT,
    SYSTEM_REWRITE_PROMPT,
)

# ----------------------------
# Test doubles
# ----------------------------


@dataclass
class _LLMCall:
    system: str
    user: str
    tags: tuple[str, ...]


class FakeLLM:
    """
    Deterministic LLM stub that returns stage-appropriate outputs.

    The SummarizingAgent calls llm.invoke([SystemMessage, HumanMessage], config=...).
    We detect stage from the system prompt.
    """

    def __init__(
        self, *, reduce_output: str, rewrite_output: Optional[str] = None
    ):
        self.calls: list[_LLMCall] = []
        self._reduce_output = reduce_output
        self._rewrite_output = rewrite_output or ""

    def invoke(
        self,
        messages: list[Any],
        config: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ):
        system = getattr(messages[0], "content", "")
        user = getattr(messages[1], "content", "")
        tags = tuple((config or {}).get("tags", []))
        self.calls.append(_LLMCall(system=system, user=user, tags=tags))

        if system == SYSTEM_MAP_PROMPT:
            # Compact bullet notes
            return SimpleNamespace(text="- note one\n- note two\n- note three")

        if system == SYSTEM_REDUCE_PROMPT:
            # Final structured output (may intentionally contain forbidden words to trigger rewrite)
            return SimpleNamespace(text=self._reduce_output)

        if system == SYSTEM_REWRITE_PROMPT:
            # Rewrite output to scrub forbidden meta-language
            return SimpleNamespace(text=self._rewrite_output)

        # If prompts change, make failures obvious.
        raise AssertionError(
            f"Unexpected system prompt passed to LLM: {system!r}"
        )


def _stub_read_file_from_workspace(filename: str, state: dict[str, Any]) -> str:
    """
    Tool stub for ursa.tools.read_file.

    The agent passes tool_state = {"workspace": <input_dir>}.
    The agent passes filename as a relative path under that workspace.
    """
    workspace = Path(state["workspace"])
    p = workspace / filename
    return p.read_text(encoding="utf-8")


# ----------------------------
# Fixtures
# ----------------------------


@pytest.fixture(autouse=True)
def stub_read_file_tool(monkeypatch):
    """
    Replace the agent's imported read_file tool with a deterministic local reader.

    Note: we patch the symbol in the agent module, not ursa.tools itself,
    mirroring your existing tests pattern.
    """
    monkeypatch.setattr(
        "ursa.agents.summarizing_agent.read_file",
        _stub_read_file_from_workspace,
    )


# ----------------------------
# Helpers
# ----------------------------


def _write(tmp_path: Path, rel: str, text: str) -> None:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def _minimal_good_summary() -> str:
    # A reduce-stage output that already satisfies the structure.
    return (
        "Executive Summary: This is a short executive summary.\n\n"
        "Required Actions:\n"
        "- Eng: Do the thing this week\n"
        "- PM: Confirm scope in 30 days\n\n"
        "Main Synthesis: " + ("This is a paragraph. " * 120).strip()
    )


def _forbidden_reduce_summary() -> str:
    # Intentionally includes forbidden meta-language to trigger rewrite.
    return (
        "Executive Summary: This document explains the text above.\n\n"
        "Required Actions:\n"
        "- Eng: Do the thing this week\n\n"
        "Main Synthesis: The first document says X and the second file says Y."
    )


def _clean_rewrite_summary() -> str:
    return (
        "Executive Summary: This is a unified summary without referencing inputs.\n\n"
        "Required Actions:\n"
        "- Eng: Validate the key assumptions this week\n"
        "- PM: Align stakeholders in 30 days\n\n"
        "Main Synthesis: " + ("Unified synthesis paragraph. " * 80).strip()
    )


# ----------------------------
# Tests
# ----------------------------


def test_summarizing_agent_empty_dir_returns_empty_summary(tmp_path: Path):
    llm = FakeLLM(reduce_output=_minimal_good_summary())
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    result = agent.invoke({"input_docs_dir": str(tmp_path)})

    assert result["summary"] == ""
    assert result["selected_files"] == []
    assert result["skipped_files"] == []
    assert llm.calls == []


def test_summarizing_agent_selects_files_filters_hidden_and_extensions(
    tmp_path: Path,
):
    # Eligible
    _write(tmp_path, "a.txt", "alpha")
    _write(tmp_path, "b.md", "bravo")
    # Ineligible extensions
    _write(tmp_path, "c.json", '{"x":1}')
    # Hidden path should be skipped
    _write(tmp_path, ".hidden/secret.txt", "nope")

    llm = FakeLLM(reduce_output=_minimal_good_summary())
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    result = agent.invoke({
        "input_docs_dir": str(tmp_path),
        "allowed_extensions": (".txt", ".md"),
        "recurse": True,
    })

    # We can't directly see selection order from agent output except via selected_files
    assert sorted(result["selected_files"]) == ["a.txt", "b.md"]
    assert all(".hidden" not in s for s in result["selected_files"])
    assert "c.json" not in result["selected_files"]

    # Should produce a summary (non-empty) given two small docs.
    assert (result["summary"] or "").strip()


def test_summarizing_agent_max_files_enforces_deterministic_cap(tmp_path: Path):
    _write(tmp_path, "a.txt", "a")
    _write(tmp_path, "b.txt", "b")
    _write(tmp_path, "c.txt", "c")

    llm = FakeLLM(reduce_output=_minimal_good_summary())
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    result = agent.invoke({
        "input_docs_dir": str(tmp_path),
        "allowed_extensions": (".txt",),
        "max_files": 2,
    })

    # Sorted case-insensitive: a, b, c -> first two selected, remaining skipped
    assert result["selected_files"] == ["a.txt", "b.txt"]
    assert result["skipped_files"] == ["c.txt"]


def test_summarizing_agent_non_strict_skips_tool_error(
    tmp_path: Path, monkeypatch
):
    _write(tmp_path, "a.txt", "alpha")
    _write(tmp_path, "b.txt", "bravo")

    def bad_read_file(filename: str, state: dict[str, Any]) -> str:
        if filename == "a.txt":
            return "[Error] failed to read"
        return _stub_read_file_from_workspace(filename, state)

    monkeypatch.setattr(
        "ursa.agents.summarizing_agent.read_file", bad_read_file
    )

    llm = FakeLLM(reduce_output=_minimal_good_summary())
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    result = agent.invoke({
        "input_docs_dir": str(tmp_path),
        "allowed_extensions": (".txt",),
        "strict": False,
    })

    assert result["selected_files"] == ["a.txt", "b.txt"]
    # a.txt is skipped internally; selected_files lists discovered, not successfully read
    assert (result["summary"] or "").strip()
    # Only b.txt content is summarized; LLM should still have been called.
    assert llm.calls


def test_summarizing_agent_strict_raises_on_tool_error(
    tmp_path: Path, monkeypatch
):
    _write(tmp_path, "a.txt", "alpha")

    def bad_read_file(filename: str, state: dict[str, Any]) -> str:
        return "[Error] failed to read"

    monkeypatch.setattr(
        "ursa.agents.summarizing_agent.read_file", bad_read_file
    )

    llm = FakeLLM(reduce_output=_minimal_good_summary())
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    with pytest.raises(RuntimeError):
        agent.invoke({
            "input_docs_dir": str(tmp_path),
            "allowed_extensions": (".txt",),
            "strict": True,
        })


def test_summarizing_agent_rewrite_pass_triggers_and_scrubs_forbidden_refs(
    tmp_path: Path,
):
    _write(tmp_path, "a.txt", "alpha " * 50)
    _write(tmp_path, "b.txt", "bravo " * 50)

    llm = FakeLLM(
        reduce_output=_forbidden_reduce_summary(),
        rewrite_output=_clean_rewrite_summary(),
    )
    agent = SummarizingAgent(llm=llm, workspace=tmp_path)

    result = agent.invoke({
        "input_docs_dir": str(tmp_path),
        "allowed_extensions": (".txt",),
        # Keep chunks small so we always exercise map->reduce with multiple calls
        "chunk_size_chars": 200,
        "chunk_overlap_chars": 50,
        "max_chunks_per_file": 20,
        "reduce_batch_size": 2,
    })

    summary = result["summary"]
    assert "Executive Summary" in summary
    assert "Required Actions" in summary
    assert "Main Synthesis" in summary

    # Rewrite should remove common forbidden tokens that appear in the bad reduce output.
    lowered = summary.lower()
    assert "document" not in lowered
    assert "the text above" not in lowered
    assert "first document" not in lowered

    # Ensure the rewrite stage was actually invoked.
    assert any(call.system == SYSTEM_REWRITE_PROMPT for call in llm.calls)


def test_summarizing_agent_emits_events(tmp_path: Path):
    _write(tmp_path, "a.txt", "alpha " * 20)

    llm = FakeLLM(reduce_output=_minimal_good_summary())

    events: list[str] = []

    def on_event(name: str, data: dict[str, Any]) -> None:
        events.append(name)

    agent = SummarizingAgent(llm=llm, workspace=tmp_path)
    result = agent.invoke({
        "input_docs_dir": str(tmp_path),
        "allowed_extensions": (".txt",),
        "on_event": on_event,
    })

    assert (result["summary"] or "").strip()
    # Spot-check that major lifecycle events were emitted
    assert "start" in events
    assert "discover_done" in events
    assert "chunking_done" in events
    assert "reduce_done" in events
    assert "done" in events
