from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import BaseTool

from ursa.cli.config import UrsaConfig
from ursa.rag import persistence
from ursa.rag.persistence import (
    normalize_rag_tool_names,
    resolve_ingest_source,
)
from ursa.rag.tools import build_rag_tools, rag_tool_name


def test_rag_agent_import_does_not_trigger_cli_security_cycle():
    from ursa.agents.rag_agent import RAGAgent

    assert RAGAgent.__name__ == "RAGAgent"


def test_normalize_rag_tool_names_accepts_comma_separated_values():
    assert normalize_rag_tool_names("docs, policy") == ["docs", "policy"]
    assert normalize_rag_tool_names(["docs,policy", "notes"]) == [
        "docs",
        "policy",
        "notes",
    ]
    assert UrsaConfig(rag_tools="docs,policy").rag_tools == ["docs", "policy"]


def test_rag_subcommands_accept_config_after_subcommand(tmp_path: Path):
    from ursa.cli import build_parser, resolve_config

    source = tmp_path / "docs"
    source.mkdir()
    config_file = tmp_path / "special_group_config.yaml"
    config_file.write_text(
        "\n".join([
            "llm_model:",
            "  model: openai:gpt-test",
            "  base_url: https://models.example.test/v1",
            "emb_model:",
            "  model: openai:text-embedding-test",
            "  base_url: https://embeddings.example.test/v1",
        ]),
        encoding="utf-8",
    )

    parser = build_parser()
    cfg = parser.parse_args([
        "rag-ingest",
        str(source),
        "--name",
        "new-rag-agent",
        "--group",
        "special-group",
        "--config",
        str(config_file),
    ])
    resolved = resolve_config(cfg)

    assert resolved.llm_model.model == "openai:gpt-test"
    assert resolved.llm_model.base_url == "https://models.example.test/v1"
    assert resolved.emb_model is not None
    assert resolved.emb_model.model == "openai:text-embedding-test"
    assert resolved.emb_model.base_url == "https://embeddings.example.test/v1"

    cfg = parser.parse_args([
        "rag-query",
        "--name",
        "new-rag-agent",
        "--group",
        "special-group",
        "--config",
        str(config_file),
        "What",
        "is",
        "indexed?",
    ])
    resolved = resolve_config(cfg)

    assert resolved.llm_model.model == "openai:gpt-test"
    assert resolved.emb_model is not None
    assert resolved.emb_model.model == "openai:text-embedding-test"


def test_resolve_ingest_source_validates_without_copying(tmp_path: Path):
    rag_root = tmp_path / "rag"
    file_source = tmp_path / "source.txt"
    file_source.write_text("Persistent RAG document text", encoding="utf-8")

    assert resolve_ingest_source(file_source) == file_source.resolve()
    assert not (rag_root / "database" / "source.txt").exists()

    source_dir = tmp_path / "docs"
    nested = source_dir / "nested" / "policy.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("Policy document", encoding="utf-8")

    assert resolve_ingest_source(source_dir) == source_dir.resolve()
    assert not (rag_root / "database" / "nested" / "policy.md").exists()

    with pytest.raises(FileNotFoundError):
        resolve_ingest_source(tmp_path / "missing")


def test_build_persistent_rag_agent_uses_external_ingest_source(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(persistence, "RAG_AGENTS_DIR", tmp_path / "ursa_rag")
    source = tmp_path / "docs"
    source.mkdir()
    captured = {}

    class FakeRAGAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("ursa.agents.RAGAgent", FakeRAGAgent)

    persistence.build_persistent_rag_agent(
        name="docs",
        group="default",
        llm=SimpleNamespace(),
        create=True,
        database_path=source,
    )

    assert (
        Path(captured["workspace"])
        == tmp_path / "ursa_rag" / "default" / "docs"
    )
    assert Path(captured["database_path"]) == source
    assert not any((captured["workspace"] / "database").iterdir())


def test_rag_ingest_passes_source_path_without_copying(
    monkeypatch, tmp_path: Path
):
    from jsonargparse import Namespace

    from ursa.cli.rag_management import handle_rag_command

    monkeypatch.setattr(persistence, "RAG_AGENTS_DIR", tmp_path / "ursa_rag")
    source = tmp_path / "source.md"
    source.write_text("hello", encoding="utf-8")
    captured = {}

    class FakeAgent:
        def invoke(self, payload):
            captured["payload"] = payload
            return {"rag_metadata": {"num_results": 0}}

    monkeypatch.setattr(
        "ursa.cli.rag_management._init_models",
        lambda *args, **kwargs: (SimpleNamespace(), SimpleNamespace()),
    )

    def fake_build_persistent_rag_agent(**kwargs):
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr(
        "ursa.cli.rag_management.build_persistent_rag_agent",
        fake_build_persistent_rag_agent,
    )

    args = Namespace(
        subcommand="rag-ingest",
        **{
            "rag-ingest": Namespace(
                source=str(source),
                name="docs",
                group="default",
                return_k=10,
                chunk_size=1000,
                chunk_overlap=200,
            )
        },
    )

    assert handle_rag_command(args)
    assert captured["database_path"] == source.resolve()
    assert not any(
        (tmp_path / "ursa_rag" / "default" / "docs" / "database").iterdir()
    )


def test_build_rag_tools_wraps_persisted_agent(monkeypatch, capsys):
    class FakeRAG:
        def invoke(self, payload):
            assert payload["context"] == "What is in the docs?"
            assert payload["query"] == "What is in the docs?"
            return {"summary": "The docs say hello."}

    def fake_builder(**kwargs):
        assert kwargs["name"] == "policy-docs"
        assert kwargs["group"] == "default"
        return FakeRAG()

    monkeypatch.setattr(
        "ursa.rag.tools.build_persistent_rag_agent", fake_builder
    )
    tools = build_rag_tools(
        names=["policy-docs"],
        group="default",
        llm=SimpleNamespace(),
    )

    assert len(tools) == 1
    tool = tools[0]
    assert isinstance(tool, BaseTool)
    assert tool.name == rag_tool_name("policy-docs")
    assert (
        tool.invoke({"query": "What is in the docs?"}) == "The docs say hello."
    )
    assert (
        "[Request to policy-docs]: What is in the docs?"
        in capsys.readouterr().out
    )


def test_rag_group_copies_regular_agent_group_config(
    monkeypatch, tmp_path: Path
):
    agent_groups = tmp_path / "ursa_agents"
    rag_groups = tmp_path / "ursa_rag"
    regular_group = agent_groups / "science"
    regular_group.mkdir(parents=True)
    group_config = regular_group / "group.yaml"
    group_config.write_text(
        "allowed_base_urls:\n  - https://models.example.test\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(persistence, "RAG_AGENTS_DIR", rag_groups)
    monkeypatch.setattr(persistence, "AGENT_GROUPS_DIR", agent_groups)

    path = persistence.ensure_rag_agent_dir("science", "docs")

    assert path == rag_groups / "science" / "docs"
    copied_config = rag_groups / "science" / "group.yaml"
    assert copied_config.read_text(encoding="utf-8") == group_config.read_text(
        encoding="utf-8"
    )


def test_rag_group_errors_when_regular_agent_group_is_missing(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(persistence, "RAG_AGENTS_DIR", tmp_path / "ursa_rag")
    monkeypatch.setattr(
        persistence, "AGENT_GROUPS_DIR", tmp_path / "ursa_agents"
    )

    with pytest.raises(ValueError, match="Group 'missing' does not exist"):
        persistence.ensure_rag_agent_dir("missing", "docs")


def test_rag_agent_dir_uses_separate_cache_root(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(persistence, "RAG_AGENTS_DIR", tmp_path / "ursa_rag")
    path = persistence.ensure_rag_agent_dir("default", "docs")
    assert path == tmp_path / "ursa_rag" / "default" / "docs"
    assert (path / "database").is_dir()
    assert (path / "summaries").is_dir()
    assert (path / "vectorstore").is_dir()
    assert persistence.list_rag_agent_names("default") == ["docs"]


def test_agent_with_tools_binds_rag_tools(monkeypatch, chat_model):
    from langchain_core.tools import tool

    from ursa.agents import ChatAgent

    @tool
    def fake_rag(query: str) -> str:
        """Query fake RAG."""
        return f"fake: {query}"

    captured = {}

    def fake_build_rag_tools(**kwargs):
        captured.update(kwargs)
        return [fake_rag]

    monkeypatch.setattr("ursa.rag.tools.build_rag_tools", fake_build_rag_tools)
    agent = ChatAgent(llm=chat_model, rag_tools="docs")

    assert "fake_rag" in agent.tools
    assert captured["names"] == ("docs",)
    assert captured["group"] == "default"
