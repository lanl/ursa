from pathlib import Path
from types import SimpleNamespace

from ursa.util.events import DEFAULT_EVENT_NAME


class FakeAgent:
    description = "A configured test agent."
    config = {"mode": "test"}


class FakeHITL:
    model = SimpleNamespace(model_name="test-model")
    embedding = None
    group = "default"
    config = SimpleNamespace(mcp_servers={})
    agents = {"chat": FakeAgent(), "plan": FakeAgent()}

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.calls = []

    async def run_agent(self, name, prompt, callbacks=None):
        self.calls.append((name, prompt))
        return "Finished"


async def emit_event(handler, payload=None, **details):
    await handler.on_custom_event(
        DEFAULT_EVENT_NAME,
        details if payload is None else payload,
    )
