from pathlib import Path
from types import SimpleNamespace

from ursa.cli.config import (
    ChatModelConfig,
    EmbModelConfig,
    InferenceProviderConfig,
)


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
        self.model_changes = []
        self.inference_provider = "openai"
        self.embedding_inference_provider = "openai"
        self.closed = False

    async def aclose(self):
        self.closed = True

    @property
    def initialized_model(self):
        return self.model

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
