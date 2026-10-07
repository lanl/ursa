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

    def _summarize(result: object) -> str:
        summary = result.get("summary") if isinstance(result, dict) else None
        if summary:
            return str(summary)
        return str(result)

    def query_rag(query: str) -> str:
        """Query the persisted RAG collection and return its summary."""
        logger.info(f"[Request to {name}]: {query}")
        result = rag_agent.invoke({"context": query, "query": query})
        return _summarize(result)

    async def aquery_rag(query: str) -> str:
        """Async query the persisted RAG collection and return its summary.

        Providing an async implementation lets LangGraph's ``ToolNode`` await the
        RAG agent directly on the running event loop (via ``ainvoke``) instead of
        dispatching the synchronous ``invoke`` into a worker thread. Running the
        RAG agent's sync SQLite-backed graph from an executor thread while the
        parent agent's event loop owns async SQLite/Chroma resources is what
        triggers the ``bad value(s) in fds_to_keep`` and lock errors.
        """
        logger.info(f"[Request to {name}]: {query}")
        result = await rag_agent.ainvoke({"context": query, "query": query})
        return _summarize(result)

    return StructuredTool.from_function(
        func=query_rag,
        coroutine=aquery_rag,
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
