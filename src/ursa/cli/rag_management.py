"""CLI helpers for persistent URSA RAG agents."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from ursa.cli.config import ModelConfig, UrsaConfig
from ursa.rag.persistence import (
    build_persistent_rag_agent,
    delete_rag_agent,
    ensure_rag_agent_dir,
    list_rag_agent_names,
    require_rag_agent_dir,
    resolve_ingest_source,
    save_rag_agent,
    show_rag_agent,
    validate_rag_agent_name,
)
from ursa.security import (
    enforce_group_base_url_policy,
    enforce_model_group_policy,
)

RAG_COMMANDS = {
    "rag-ingest",
    "rag-query",
    "list-rag-agents",
    "show-rag-agent",
    "delete-rag-agent",
    "save-rag-agent",
}


def add_rag_subcommands(subparsers) -> None:
    from jsonargparse import ArgumentParser

    ingest = ArgumentParser()
    ingest.add_argument("source", help="File or directory to ingest")
    ingest.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Path to a YAML/JSON file with URSA model and embedding configuration.",
    )
    ingest.add_argument(
        "--name", required=True, help="Persistent RAG agent name"
    )
    ingest.add_argument("--group", default="default", help="RAG group name")
    ingest.add_argument("--return-k", type=int, default=10)
    ingest.add_argument("--chunk-size", type=int, default=1000)
    ingest.add_argument("--chunk-overlap", type=int, default=200)
    subparsers.add_subcommand(
        "rag-ingest",
        ingest,
        help="Create/update a persistent RAG agent from a file or directory.",
        dest="subcommand",
    )

    query = ArgumentParser()
    query.add_argument(
        "query", nargs="*", help="Query text. Omit for REPL mode."
    )
    query.add_argument(
        "--config",
        default=None,
        type=Path,
        help="Path to a YAML/JSON file with URSA model and embedding configuration.",
    )
    query.add_argument(
        "--name", required=True, help="Persistent RAG agent name"
    )
    query.add_argument("--group", default="default", help="RAG group name")
    query.add_argument("--return-k", type=int, default=10)
    subparsers.add_subcommand(
        "rag-query",
        query,
        help="Query a persistent RAG agent.",
        dest="subcommand",
    )

    list_cmd = ArgumentParser()
    list_cmd.add_argument("--group", default="default", help="RAG group name")
    subparsers.add_subcommand(
        "list-rag-agents",
        list_cmd,
        help="List persistent RAG agents in a group.",
        dest="subcommand",
    )

    show = ArgumentParser()
    show.add_argument("name", help="Persistent RAG agent name")
    show.add_argument("--group", default="default", help="RAG group name")
    subparsers.add_subcommand(
        "show-rag-agent",
        show,
        help="Show details for a persistent RAG agent.",
        dest="subcommand",
    )

    delete = ArgumentParser()
    delete.add_argument("name", help="Persistent RAG agent name")
    delete.add_argument("--group", default="default", help="RAG group name")
    subparsers.add_subcommand(
        "delete-rag-agent",
        delete,
        help="Delete a persistent RAG agent.",
        dest="subcommand",
    )

    save = ArgumentParser()
    save.add_argument("name", help="Persistent RAG agent name")
    save.add_argument("--group", default="default", help="RAG group name")
    subparsers.add_subcommand(
        "save-rag-agent",
        save,
        help="Save a timestamped copy of a persistent RAG agent.",
        dest="subcommand",
    )


def _model_config_from_namespace(
    args: Namespace,
) -> tuple[ModelConfig, ModelConfig | None]:
    data = args.as_dict() if hasattr(args, "as_dict") else vars(args)
    cfg = UrsaConfig.model_validate(data, extra="ignore")
    return cfg.llm_model, cfg.emb_model


def _init_models(
    root_args: Namespace,
    cmd_args: Namespace,
    config: UrsaConfig | None = None,
):
    llm_model, emb_model = (
        (config.llm_model, config.emb_model)
        if config is not None
        else _model_config_from_namespace(root_args)
    )
    group = getattr(cmd_args, "group", "default") or "default"
    enforce_group_base_url_policy(llm_model.base_url, group)
    llm = init_chat_model(**llm_model.kwargs)
    enforce_model_group_policy(llm, group)
    embedding = None
    if emb_model:
        enforce_group_base_url_policy(emb_model.base_url, group)
        embedding = init_embeddings(**emb_model.kwargs)
        enforce_model_group_policy(embedding, group)
    return llm, embedding


def handle_rag_command(
    args: Namespace, config: UrsaConfig | None = None
) -> bool:
    command = (
        args.get("subcommand", None)
        if hasattr(args, "get")
        else getattr(args, "subcommand", None)
    )
    if command not in RAG_COMMANDS:
        return False
    cmd_args = args.get(command, args) if hasattr(args, "get") else args

    if command == "list-rag-agents":
        names = list_rag_agent_names(cmd_args.group)
        if names:
            print("\n".join(names))
        else:
            print(f"No RAG agents found in group '{cmd_args.group}'")
        return True

    if command == "show-rag-agent":
        show_rag_agent(cmd_args.name, cmd_args.group)
        return True

    if command == "delete-rag-agent":
        delete_rag_agent(cmd_args.name, cmd_args.group)
        return True

    if command == "save-rag-agent":
        save_rag_agent(cmd_args.name, cmd_args.group)
        return True

    llm, embedding = _init_models(args, cmd_args, config)

    if command == "rag-ingest":
        name = validate_rag_agent_name(cmd_args.name)
        rag_root = ensure_rag_agent_dir(cmd_args.group, name)
        source = resolve_ingest_source(Path(cmd_args.source))
        agent = build_persistent_rag_agent(
            name=name,
            group=cmd_args.group,
            llm=llm,
            embedding=embedding,
            create=True,
            return_k=cmd_args.return_k,
            chunk_size=cmd_args.chunk_size,
            chunk_overlap=cmd_args.chunk_overlap,
            database_path=source,
        )
        # Trigger the RAG graph. The read/ingest nodes index anything not already stored.
        result = agent.invoke({
            "context": "Ingest documents.",
            "query": "Ingest documents.",
        })
        print(f"RAG agent: {name}")
        print(f"Group: {cmd_args.group}")
        print(f"Path: {rag_root}")
        print(f"Source: {source}")
        # print("Raw documents copied: no")
        metadata = (
            result.get("rag_metadata", {}) if isinstance(result, dict) else {}
        )
        if metadata:
            print(
                f"Indexed/retrieved results: {metadata.get('num_results', 0)}"
            )
        return True

    if command == "rag-query":
        query_text = " ".join(cmd_args.query).strip()
        require_rag_agent_dir(cmd_args.group, cmd_args.name)
        agent = build_persistent_rag_agent(
            name=cmd_args.name,
            group=cmd_args.group,
            llm=llm,
            embedding=embedding,
            create=False,
            return_k=cmd_args.return_k,
        )
        if query_text:
            result = agent.invoke({"context": query_text, "query": query_text})
            print(
                result.get("summary", result)
                if isinstance(result, dict)
                else result
            )
            return True

        print(
            f"Entering RAG query loop for '{cmd_args.name}'. Press Ctrl-D or enter blank line to exit."
        )
        while True:
            try:
                query_text = input("rag> ").strip()
            except EOFError:
                print()
                break
            if not query_text:
                break
            result = agent.invoke({"context": query_text, "query": query_text})
            print(
                result.get("summary", result)
                if isinstance(result, dict)
                else result
            )
        return True

    return False
