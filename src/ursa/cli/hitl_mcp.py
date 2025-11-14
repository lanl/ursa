# ursa/cli/hitl_mcp.py
# pip install "mcp[cli]" fastapi uvicorn

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass

# MCP SDK
from mcp.server.fastmcp import Context, FastMCP  # convenience wrapper
from mcp.server.session import ServerSession

from ursa.cli.hitl import HITL

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


async def _heartbeat(ctx: Context, label: str, interval: float = 5.0) -> None:
    """
    Periodically stream bytes so HTTP clients don't hit read timeouts.
    - If the client provided a progressToken, report structured progress.
    - Otherwise, send lightweight log messages.
    """
    tick = 0
    try:
        while True:
            tick += 1
            seconds = int(tick * interval)
            # Try structured progress first (no-op if client didn't send a token)
            try:
                ctx.report_progress(tick, message=f"{label}… t={seconds}s")
            except Exception:
                # Fallback: log line still streams bytes
                ctx.info(f"{label}… t={seconds}s")
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        with contextlib.suppress(Exception):
            ctx.info(f"{label}… finishing")
        raise


async def _run_with_keepalive(
    ctx: Context[ServerSession, AppCtx], label: str, fn, *args, **kwargs
):
    """
    Run a blocking HITL function in a worker thread while emitting keepalive bytes.
    """
    hb = asyncio.create_task(_heartbeat(ctx, label, interval=5.0))
    try:
        # Run the (likely blocking) HITL method in a thread so the loop can stream logs/progress.
        return await asyncio.to_thread(fn, *args, **kwargs)
    finally:
        hb.cancel()
        with contextlib.suppress(Exception):
            await hb


def _hitl(ctx: Context[ServerSession, AppCtx]) -> HITL:
    return ctx.request_context.lifespan_context.hitl


# Each tool is a thin shim to your HITL methods. The type hints become the tool's JSON Schema.
@mcp.tool(
    description="Search for papers on arXiv and summarize in the query context."
)
async def arxiv(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "ArxivAgent processing", _hitl(ctx).run_arxiv, query
    )


@mcp.tool(description="Build a step-by-step plan to solve the user's problem.")
async def plan(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "PlanningAgent processing", _hitl(ctx).run_planner, query
    )


@mcp.tool(
    description="Execute a ReAct agent that can write/edit code & run commands."
)
async def execute(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "ExecuteAgent processing", _hitl(ctx).run_executor, query
    )


@mcp.tool(description="Search the web and summarize results in context.")
async def web(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "WebSearchAgent processing", _hitl(ctx).run_websearcher, query
    )


@mcp.tool(description="Recall prior execution steps from memory (RAG).")
async def recall(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "RecallAgent processing", _hitl(ctx).run_rememberer, query
    )


@mcp.tool(description="Deep reasoning to propose an approach.")
async def hypothesize(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "HypothesizerAgent processing", _hitl(ctx).run_hypothesizer, query
    )


@mcp.tool(description="Direct chat with the hosted LLM.")
async def chat(query: str, ctx: Context[ServerSession, AppCtx]) -> str:
    return await _run_with_keepalive(
        ctx, "ChatAgent processing", _hitl(ctx).run_chatter, query
    )


# Optional: a quick ping/health tool (some clients call this)
@mcp.tool(description="Liveness check.")
def ping(_: str = "ok") -> str:
    return "pong"


# ---- ASGI app that serves the MCP Streamable HTTP endpoint ----
# This is a complete ASGI app; uvicorn can serve it directly.
mcp_http_app = mcp.streamable_http_app()
