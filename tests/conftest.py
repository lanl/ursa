import os

import pytest
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


@pytest.fixture(scope="session", autouse=True)
def _load_dotenv():
    load_dotenv()
    if not os.path.exists(".env") and os.path.exists("env.txt"):
        load_dotenv("env.txt")


def bind_kwargs(func, **kwargs):
    """Bind kwargs so that tests can recreate the model"""
    model = func(**kwargs)
    model._testing_only_kwargs = kwargs
    return model


@pytest.fixture(scope="function")
def chat_model():
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model_name = os.getenv("URSA_TEST_CHAT_MODEL", "openai:gpt-5-nano")
    if base_url and ":" not in model_name:
        # OpenAI-compatible endpoints require the OpenAI provider in init_chat_model.
        model_name = f"openai:{model_name}"

    kwargs = {
        "model": model_name,
        "max_tokens": int(os.getenv("URSA_TEST_MAX_TOKENS", "3000")),
        "temperature": float(os.getenv("URSA_TEST_TEMPERATURE", "0.0")),
    }
    if base_url:
        kwargs["base_url"] = base_url
    if api_key := os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key

    return bind_kwargs(
        init_chat_model,
        **kwargs,
    )


@pytest.fixture(scope="function")
def embedding_model():
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    model_name = os.getenv(
        "URSA_TEST_EMBED_MODEL", "openai:text-embedding-3-small"
    )
    if base_url and ":" not in model_name:
        model_name = f"openai:{model_name}"

    kwargs = {
        "model": model_name,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if api_key := os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key

    return bind_kwargs(
        init_embeddings,
        **kwargs,
    )
