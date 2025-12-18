import pytest
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


@pytest.fixture(scope="function")
def chat_model():
    return init_chat_model("openai:gpt-5-nano", max_tokens=200, temperature=0.0)


@pytest.fixture(scope="function")
def embedding_model():
    return init_embeddings("openai:text-embedding-3-small")
