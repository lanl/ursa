from types import SimpleNamespace

import pytest
from dotenv import load_dotenv
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel

from tests.agents.utils import (
    ToolReadyFakeChatModel,
    message_stream,
    message_with_usage,
)
from ursa.util.has_extra import has_extra


def extraskip(extra: str, package: str = "ursa-ai") -> None:
    """Skip the calling module unless its declared extra is available.

    Call before module-level imports, like ``pytest.importorskip``. Unknown
    extras raise ValueError so typos cannot silently skip tests.
    """
    if not has_extra(extra, package=package):
        pytest.skip(
            f"Tests require {package}[{extra}]", allow_module_level=True
        )


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv():
    load_dotenv()


@pytest.fixture
def fake_keyring(monkeypatch):
    """Install a password lookup without accessing the system keyring."""

    def install(get_password):
        monkeypatch.setattr(
            "ursa.util.secrets.get_keyring",
            lambda: SimpleNamespace(get_password=get_password),
        )

    return install


def _fake_structured(schema):
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        if "steps" in schema.model_fields:
            return schema(
                steps=[
                    {
                        "name": "Stub step",
                        "description": "Stubbed plan step for tests.",
                        "requires_code": False,
                        "expected_outputs": ["stub"],
                        "success_criteria": ["stub"],
                    }
                ]
            )
        return schema()

    annotations = getattr(schema, "__annotations__", {}) or {}
    if "is_safe" in annotations and "reason" in annotations:
        return {
            "is_safe": True,
            "reason": "Stubbed safety check for tests",
        }
    return {key: "stub" for key in annotations}


class FakeChatModel(ToolReadyFakeChatModel):
    @property
    def model_name(self) -> str:
        return "fake-chat"

    model = model_name

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = " ".join(
            msg.content
            for msg in messages
            if hasattr(msg, "content") and isinstance(msg.content, str)
        ).lower()
        if "latex" in text or "\\documentclass" in text:
            content = "\\documentclass{article}\n\\begin{document}\nStub\n\\end{document}"
        else:
            content = "ok"

        message = message_with_usage(content=content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def with_structured_output(self, schema, **kwargs):
        output = _fake_structured(schema)

        async def ainvoke(messages):
            return output

        return SimpleNamespace(invoke=lambda messages: output, ainvoke=ainvoke)


class FakeEmbeddingModel(FakeEmbeddings):
    @property
    def model_name(self) -> str:
        return "fake-embeddings"

    model = model_name


@pytest.fixture(autouse=True)
def _stub_model_init(monkeypatch, request):
    if request.node.get_closest_marker("real_llm"):
        return

    def fake_init_chat_model(*args, **kwargs):
        return FakeChatModel(messages=message_stream("ok"))

    def fake_init_embeddings(*args, **kwargs):
        return FakeEmbeddingModel(size=12)

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.setattr(
        "langchain.embeddings.init_embeddings", fake_init_embeddings
    )
    monkeypatch.setattr(
        "ursa.agents.rag_agent.init_embeddings",
        fake_init_embeddings,
        raising=False,
    )


@pytest.fixture(scope="function")
def chat_model():
    model = FakeChatModel(messages=message_stream("ok"))
    model._testing_only_kwargs = {
        "model": "fake:chat",
        "max_tokens": 3000,
        "temperature": 0.0,
    }
    return model


@pytest.fixture(scope="function")
def embedding_model():
    model = FakeEmbeddingModel(size=12)
    model._testing_only_kwargs = {
        "model": "fake:embeddings",
        "model_provider": "fake",
    }
    return model
