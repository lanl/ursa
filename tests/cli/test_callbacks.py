from __future__ import annotations

# ruff: noqa: TID251
import asyncio
import io

import pytest
from rich.console import Console

from ursa.cli.callbacks import AgentLogEventHandler, HITLLogEventHandler
from ursa.util.events import DEFAULT_EVENT_NAME
from ursa.util.rendering import event_artifact


def test_hitl_log_event_handler_redirects_to_agent_handler(tmp_path) -> None:
    with pytest.warns(
        DeprecationWarning,
        match=(
            "HITLLogEventHandler is deprecated; "
            "use AgentLogEventHandler instead"
        ),
    ):
        handler = HITLLogEventHandler(
            console=Console(file=io.StringIO()),
            workspace=tmp_path,
        )

    assert isinstance(handler, AgentLogEventHandler)


def test_log_event_handler_renders_agent_and_tool_events(tmp_path) -> None:
    output = io.StringIO()
    handler = AgentLogEventHandler(
        console=Console(
            file=output,
            force_terminal=False,
            force_interactive=False,
            color_system=None,
            width=80,
        ),
        workspace=tmp_path,
    )

    async def emit() -> None:
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "PlanningAgent",
                "stage": "reflect_result",
                "message": "Plan needs another pass",
                "approved": False,
                "reason": "Need one more concrete step.",
            },
            run_id="agent-run",
        )
        await handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "tool": "edit_code",
                "stage": "edit",
                "phase": "end",
                "message": "File updated",
                "artifact": event_artifact(
                    "--- app.py\n+++ app.py\n-old\n+new",
                    "text/x-diff",
                    metadata={"title": "Edit diff", "path": "repo/app.py"},
                ),
            },
            run_id="tool-run",
        )

    asyncio.run(emit())

    rendered = output.getvalue()
    assert "Plan needs another pass" in rendered
    assert "Need one more concrete step." in rendered
    assert "Edit diff" in rendered
    assert "-old" in rendered
    assert "+new" in rendered


def test_log_event_handler_renders_named_agent_artifacts(tmp_path) -> None:
    output = io.StringIO()
    handler = AgentLogEventHandler(
        console=Console(file=output, force_terminal=False, color_system=None),
        workspace=tmp_path,
    )

    asyncio.run(
        handler.on_custom_event(
            DEFAULT_EVENT_NAME,
            {
                "agent": "dummy_bot_3000",
                "tool": "run_command",
                "stage": "execute",
                "phase": "end",
                "message": "Command finished",
                "query": "python first_10_integers.py",
                "artifacts": [
                    event_artifact(
                        "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n",
                        "text/plain",
                        metadata={"title": "stdout"},
                    )
                ],
            },
            run_id="named-command-tool-run",
        )
    )

    rendered = output.getvalue()
    assert "stdout" in rendered
    assert "1" in rendered
    assert "10" in rendered
