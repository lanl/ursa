"""Tool wrappers for persisted URSA RAG collections."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_core.tools import BaseTool, StructuredTool

from ursa.rag.persistence import (
    build_persistent_rag_agent,
    normalize_rag_tool_names,
    validate_rag_group_name,
)

logger = logging.getLogger(__name__)


def rag_tool_name(rag_agent_name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", rag_agent_name).strip("_")
    if not safe:
        safe = "rag"
    if safe[0].isdigit():
        safe = f"rag_{safe}"
    return f"query_rag_{safe}"


def build_rag_tool(
    *,
    name: str,
    group: str,
    llm: BaseChatModel,
    embedding: Embeddings | None = None,
    return_k: int = 10,
) -> BaseTool:
    """Build a LangChain tool that queries one persisted RAG collection."""
    group = validate_rag_group_name(group)
    rag_agent = build_persistent_rag_agent(
        name=name,
        group=group,
        llm=llm,
        embedding=embedding,
        create=False,
        return_k=return_k,
    )

    def query_rag(query: str) -> str:
        """Query the persisted RAG collection and return its summary."""
        logger.info(f"[Request to {name}]: {query}")
        result = rag_agent.invoke({"context": query, "query": query})
        summary = result.get("summary") if isinstance(result, dict) else None
        if summary:
            return str(summary)
        return str(result)

    return StructuredTool.from_function(
        func=query_rag,
        name=rag_tool_name(name),
        description=(
            f"Query the persisted URSA RAG collection '{name}' in group "
            f"'{group}'. Use this for questions about documents ingested into "
            "that collection. Input should be a focused natural-language query."
        ),
    )


def build_rag_tools(
    *,
    names: str | Sequence[str] | None,
    group: str,
    llm: BaseChatModel,
    embedding: Embeddings | None = None,
    return_k: int = 10,
) -> list[BaseTool]:
    return [
        build_rag_tool(
            name=name,
            group=group,
            llm=llm,
            embedding=embedding,
            return_k=return_k,
        )
        for name in normalize_rag_tool_names(names)
    ]
