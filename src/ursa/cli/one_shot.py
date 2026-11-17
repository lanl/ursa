# ruff: noqa: TID251

"""Render one command-line response without loading the interactive interface."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.markdown import Markdown as RichMarkdown

from ursa.cli.tui.helpers import _route_prompt
from ursa.util.tqdm_lock import install_thread_only_tqdm_lock

if TYPE_CHECKING:
    from ursa.cli.runtime import HITL


def run_once(hitl: HITL, prompt: str, *, stdout: Any = None) -> str:
    """Run one routed prompt and render its event stream to standard output."""
    from ursa.cli.callbacks import HITLLogEventHandler

    install_thread_only_tqdm_lock()
    output = stdout or sys.stdout
    console = Console(file=output)
    handler = HITLLogEventHandler(console=console, workspace=hitl.workspace)
    agent, routed_prompt = _route_prompt(hitl, prompt)

    async def invoke() -> str:
        try:
            response = await hitl.run_agent(
                agent, routed_prompt, callbacks=[handler]
            )
            if handler.emitted_any:
                console.print()
            if console.is_terminal:
                console.print(RichMarkdown(response))
            else:
                print(response, file=output, flush=True)  # noqa: T201
            return response
        finally:
            await hitl.aclose()

    return asyncio.run(invoke())
