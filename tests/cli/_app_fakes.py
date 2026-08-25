from pathlib import Path
from types import SimpleNamespace

from ursa.cli.config import InferenceProviderConfig
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
        self.config = SimpleNamespace(
            mcp_servers={},
            inference_providers={
                "openai": InferenceProviderConfig(
                    base_url="https://api.openai.com/v1",
                    api_key={"env": "OPENAI_API_KEY"},
                )
            },
            llm_model=SimpleNamespace(
                model="openai:test-model",
                base_url="https://api.openai.com/v1",
            ),
            emb_model=SimpleNamespace(
                model="openai:test-embedding",
                base_url="https://api.openai.com/v1",
            ),
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
        chat_model,
        chat_inference_provider,
        embedding_model,
        embedding_inference_provider,
    ):
        self.model_changes.append((
            chat_model,
            chat_inference_provider,
            embedding_model,
            embedding_inference_provider,
        ))
        self.config.llm_model.model = chat_model
        self.config.emb_model = (
            SimpleNamespace(
                model=embedding_model,
                base_url=self.config.inference_providers[
                    embedding_inference_provider
                ].base_url,
            )
            if embedding_model is not None
            else None
        )
        self.inference_provider = chat_inference_provider
        self.embedding_inference_provider = embedding_inference_provider


async def emit_event(handler, payload=None, **details):
    await handler.on_custom_event(
        DEFAULT_EVENT_NAME,
        details if payload is None else payload,
    )
