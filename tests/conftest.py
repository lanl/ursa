from importlib import metadata

import pytest
from dotenv import load_dotenv
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import BaseModel


def extraskip(extra: str, package: str = "ursa-ai") -> None:
    """Skip the calling test module unless an extra's requirements are installed.

    Like ``pytest.importorskip``, call this before module-level imports.
    Extras aren't recorded as installed, so check their direct dependencies
    using distribution metadata, including versions and environment markers.
    Base-package dependencies are not checked. Unknown extras raise ValueError
    so typos cannot silently skip tests.
    """
    extra = canonicalize_name(extra)
    reason = f"Tests require {package}[{extra}]"
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        pytest.skip(
            f"{reason}: {package} is not installed", allow_module_level=True
        )

    extras = {
        canonicalize_name(name)
        for name in dist.metadata.get_all("Provides-Extra", [])
    }
    if extra not in extras:
        raise ValueError(f"Unknown extra {extra!r} for {package}")

    for raw_requirement in dist.requires or []:
        requirement = Requirement(raw_requirement)
        marker = requirement.marker
        if (
            marker is None
            or not marker.evaluate({"extra": extra})
            or marker.evaluate({"extra": ""})
        ):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            pytest.skip(
                f"{reason}: {requirement.name} is not installed",
                allow_module_level=True,
            )
        if not requirement.specifier.contains(installed, prereleases=True):
            pytest.skip(
                f"{reason}: {requirement.name}{requirement.specifier} required, "
                f"found {installed}",
                allow_module_level=True,
            )


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv():
    load_dotenv()


def _message_stream(content: str):
    while True:
        yield AIMessage(content=content)


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


class FakeChatModel(GenericFakeChatModel):
    @property
    def model_name(self) -> str:
        return "fake-chat"

    @property
    def model(self) -> str:
        return "fake-chat"

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

        message = AIMessage(
            content=content,
            usage_metadata={
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self

    def with_structured_output(self, schema, **kwargs):
        output = _fake_structured(schema)

        class _Runner:
            def invoke(self, messages):
                return output

            async def ainvoke(self, messages):
                return output

        return _Runner()


class FakeEmbeddingModel(FakeEmbeddings):
    @property
    def model_name(self) -> str:
        return "fake-embeddings"

    @property
    def model(self) -> str:
        return "fake-embeddings"


@pytest.fixture(autouse=True)
def _stub_model_init(monkeypatch, request):
    if request.node.get_closest_marker("real_llm"):
        return

    def fake_init_chat_model(*args, **kwargs):
        return FakeChatModel(messages=_message_stream("ok"))

    def fake_init_embeddings(*args, **kwargs):
        return FakeEmbeddingModel(size=12)

    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model", fake_init_chat_model
    )
    monkeypatch.setattr(
        "langchain.embeddings.init_embeddings", fake_init_embeddings
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.init_chat_model", fake_init_chat_model, raising=False
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.init_embeddings", fake_init_embeddings, raising=False
    )
    monkeypatch.setattr(
        "ursa.cli.config.init_chat_model",
        fake_init_chat_model,
        raising=False,
    )
    monkeypatch.setattr(
        "ursa.cli.config.init_embeddings",
        fake_init_embeddings,
        raising=False,
    )
    monkeypatch.setattr(
        "ursa.agents.rag_agent.init_embeddings",
        fake_init_embeddings,
        raising=False,
    )


@pytest.fixture(scope="function")
def chat_model():
    model = FakeChatModel(messages=_message_stream("ok"))
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
