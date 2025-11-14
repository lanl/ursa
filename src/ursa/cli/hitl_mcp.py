# ursa/cli/hitl_mcp.py
# pip install "mcp[cli]" fastapi uvicorn

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

# MCP SDK
from mcp.server.fastmcp import Context, FastMCP  # convenience wrapper
from mcp.server.session import ServerSession

# ---- Your dependency type (the thing that runs work) ----
# Import the HITL type just for type hints; the CLI will construct it.
from ursa.cli.hitl import HITL  # noqa: F401

# ---- App-scoped state injection via MCP lifespan ----


@dataclass
class AppCtx:
    hitl: HITL  # provided by CLI before server starts


# will be populated by CLI via set_hitl()
_HITL_SINGLETON: HITL | None = None


def set_hitl(h: HITL) -> None:
    """Called by the CLI to inject the already-constructed HITL instance."""
    global _HITL_SINGLETON
    _HITL_SINGLETON = h


@asynccontextmanager
async def lifespan(server: FastMCP):
    if _HITL_SINGLETON is None:
        raise RuntimeError(
            "HITL not initialized. CLI must call set_hitl(hitl) before serving."
        )
    yield AppCtx(hitl=_HITL_SINGLETON)


# ---- Build the MCP server and register tools ----

mcp = FastMCP(
    name="URSA Server",
    lifespan=lifespan,
    # description="URSA agents exposed as MCP tools (arxiv, plan, execute, web, recall, hypothesize, chat).",
)


# Each tool is a thin shim to your HITL methods. The type hints become the tool's JSON Schema.
@mcp.tool(
    description="Search for papers on arXiv and summarize in the query context."
)
def arxiv(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_arxiv(query)


@mcp.tool(description="Build a step-by-step plan to solve the user's problem.")
def plan(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_planner(query)


@mcp.tool(
    description="Execute a ReAct agent that can write/edit code & run commands."
)
def execute(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_executor(query)


@mcp.tool(description="Search the web and summarize results in context.")
def web(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_websearcher(query)


@mcp.tool(description="Recall prior execution steps from memory (RAG).")
def recall(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_rememberer(query)


@mcp.tool(description="Deep reasoning to propose an approach.")
def hypothesize(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_hypothesizer(query)


@mcp.tool(description="Direct chat with the hosted LLM.")
def chat(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return ctx.request_context.lifespan_context.hitl.run_chatter(query)


# Optional: a quick ping/health tool (some clients call this)
@mcp.tool(description="Liveness check.")
def ping(input: str = "ok") -> str:
    return "pong"


# ---- ASGI app that serves the MCP Streamable HTTP endpoint ----
# This is a complete ASGI app; uvicorn can serve it directly.
mcp_http_app = mcp.streamable_http_app()
