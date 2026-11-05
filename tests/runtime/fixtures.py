"""Importable child-process fixtures for runtime tests."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

from ursa.runtime import AgentContext, AgentMessage


class StatefulAsyncAgent:
    def __init__(self, config: dict[str, Any], context: AgentContext) -> None:
        self.config = config
        self.context = context
        self.state: dict[str, Any] = {"calls": 0, "received": []}

    async def ainvoke(self, inputs: Any, config=None) -> Any:
        self.state["calls"] += 1
        if isinstance(inputs, dict) and "announce" in inputs:
            await self.context.send_message(str(inputs["announce"]), "started")
        if isinstance(inputs, dict) and "sleep" in inputs:
            await asyncio.sleep(float(inputs["sleep"]))
        if isinstance(inputs, dict) and "send" in inputs:
            await self.context.send_message(
                str(inputs["send"]), inputs.get("payload")
            )
        callbacks = list((config or {}).get("callbacks") or ())
        for callback in callbacks:
            await callback.on_custom_event(
                "test", {"calls": self.state["calls"]}, run_id="run"
            )
        return {
            "calls": self.state["calls"],
            "label": self.config.get("label"),
            "pid": os.getpid(),
        }

    async def on_message(self, message: AgentMessage) -> None:
        if self.config.get("raise_on_message"):
            raise RuntimeError("message handler failed")
        self.state["received"].append({
            "sender": message.sender,
            "payload": message.payload,
        })

    async def received(self) -> list[dict[str, Any]]:
        return list(self.state["received"])


def stateful_factory(
    config: dict[str, Any], context: AgentContext
) -> StatefulAsyncAgent:
    return StatefulAsyncAgent(config, context)


def tool_describing_factory(config: dict[str, Any], context: AgentContext):
    """Expose a real URSA tool through the production metadata adapter."""
    del config
    from ursa.runtime.agent_runtime import _UrsaProcessAgent
    from ursa.tools.run_command_tool import run_command

    return _UrsaProcessAgent(
        SimpleNamespace(tools={"run_command": run_command}),
        context,
        "root",
    )


class InterruptibleProcessAgent:
    def __init__(self, context: AgentContext) -> None:
        self.context = context

    @staticmethod
    def format_query(prompt: str, state=None):
        return {"prompt": prompt, "interrupted": False}

    @staticmethod
    def _prepend_system_message(query, _instruction: str):
        return {**query, "interrupted": True}

    async def ainvoke(self, inputs, config=None):
        del config
        if inputs["prompt"] == "block":
            await self.context.send_message("host", "started")
            await asyncio.Event().wait()
        await self.context.send_message(
            "host",
            {
                "processed": inputs["prompt"],
                "interrupted": inputs["interrupted"],
            },
        )
        return inputs

    @staticmethod
    def format_result(result) -> str:
        return f"{result['prompt']}:{result['interrupted']}"


async def interruptible_ursa_factory(config, context: AgentContext):
    del config
    from ursa.runtime.agent_runtime import _UrsaProcessAgent

    return _UrsaProcessAgent(
        InterruptibleProcessAgent(context), context, "root"
    )
