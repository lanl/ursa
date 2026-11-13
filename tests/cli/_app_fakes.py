from pathlib import Path
from types import SimpleNamespace

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
)
from ursa.util.events import DEFAULT_EVENT_NAME


class FakeAgent:
    description = "A configured test agent."
    config = {"mode": "test"}


class FakeHITL:
    model = SimpleNamespace(model_name="test-model")
    embedding = None
    group = "default"
    agent_name = None
    agents = {"chat": FakeAgent(), "plan": FakeAgent()}

    def __init__(self, workspace: Path):
        self.workspace = workspace
        inference_providers = {
            "openai": InferenceProviderConfig(
                base_url="https://api.openai.com/v1",
                api_key={"env": "OPENAI_API_KEY"},
            )
        }
        self.config = SimpleNamespace(
            tokens_before_summarize=50000,
            mcp_servers={},
            inference_providers=inference_providers,
            llm_model=ChatModelConfig(
                model="openai:test-model", inference_provider="openai"
            ).resolve_inference_provider(inference_providers),
            emb_model=EmbModelConfig(
                model="openai:test-embedding", inference_provider="openai"
            ).resolve_inference_provider(inference_providers),
            agent_name=None,
        )
        self.calls = []
        self.model_changes = []
        self.inference_provider = "openai"
        self.embedding_inference_provider = "openai"
        self.closed = False

    async def run_agent(self, name, prompt, callbacks=None):
        self.calls.append((name, prompt))
        return "Finished"

    async def aclose(self):
        self.closed = True

    async def reconfigure_model(self, model_name, inference_provider):
        self.model_changes.append((model_name, inference_provider))
        self.config.llm_model.model = model_name
        self.inference_provider = inference_provider

    async def reconfigure_models(
        self,
        chat_config,
        embedding_config,
    ):
        self.model_changes.append((chat_config, embedding_config))
        self.config.llm_model = chat_config.resolve_inference_provider(
            self.config.inference_providers
        )
        self.model = SimpleNamespace(model_name=self.config.llm_model.model)
        self.config.emb_model = None
        self.embedding = None
        if embedding_config is not None:
            self.config.emb_model = embedding_config.resolve_inference_provider(
                self.config.inference_providers
            )
            self.embedding = SimpleNamespace(model=self.config.emb_model.model)
        self.inference_provider = chat_config.inference_provider
        self.embedding_inference_provider = (
            embedding_config.inference_provider if embedding_config else None
        )


async def emit_event(handler, payload=None, **details):
    await handler.on_custom_event(
        DEFAULT_EVENT_NAME,
        details if payload is None else payload,
    )


async def start_turn(app, prompt):
    """Start a real persisted turn without invoking a model."""
    from ursa.cli.tui.event_cards.card_stream import CardStream

    stream = CardStream(app.card_store, app.hitl.workspace, prompt)
    await stream.start()
    app.current_turn = stream
    app._reset_conversation_auto_follow(app.card_viewport.conversation)
    app.card_viewport.follow_tail = True
    await app.card_viewport.refresh()
    return app.card_viewport.turns[stream.turn_id]


async def emit_card(app, payload):
    """Wait for persistence and its viewport projection before asserting."""
    await app.add_turn_event(app.current_turn, payload)
    await app.card_viewport.refresh()
