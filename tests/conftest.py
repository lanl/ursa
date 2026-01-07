import pytest
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings


def bind_kwargs(func, **kwargs):
    """Bind kwargs so that tests can recreate the model"""
    model = func(**kwargs)
    model._testing_only_kwargs = kwargs
    return model


@pytest.fixture(scope="function")
def chat_model():
    return bind_kwargs(
        init_chat_model,
        model="openai:gpt-5-nano",
        max_tokens=200,
        temperature=0.0,
    )


@pytest.fixture(scope="function")
def embedding_model():
    return bind_kwargs(
        init_embeddings,
        model="openai:text-embedding-3-small",
    )
