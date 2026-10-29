"""Resolve context limits from local metadata, including bound models."""

from types import SimpleNamespace

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from ursa.util.model_context import (
    model_context_window,
    summarization_threshold,
)


@pytest.mark.parametrize(
    "wrapper,fields",
    [
        (ChatOpenAI, {"model_name": "gpt-4o"}),
        (ChatAnthropic, {"model": "claude-sonnet-4-5-20250929"}),
        (ChatGoogleGenerativeAI, {"model": "gemini-2.5-pro"}),
    ],
)
def test_provider_catalog_profiles(wrapper, fields):
    model = wrapper.model_construct(**fields, profile=None)
    # Exercise real catalog loading without credentials or endpoint requests.
    model._set_model_profile()
    window = model.profile["max_input_tokens"]
    assert window > 0
    assert model_context_window(model.bind(extra_headers={})) == window
    assert summarization_threshold(model) == (window * 7 + 9) // 10
    assert summarization_threshold(model.bind(extra_headers={})) == (
        summarization_threshold(model)
    )


def test_custom_endpoint_explicit_profile():
    model = ChatOpenAI.model_construct(
        model_name="custom-alias",
        openai_api_base="https://inference.example/v1",
        profile={"max_input_tokens": 32768},
    )
    assert summarization_threshold(model) == 22938


@pytest.mark.parametrize("window", [None, 0, -1, True, "8192", 8192.0])
def test_unknown_or_invalid_context_uses_fallback(window):
    model = SimpleNamespace(
        profile={"max_input_tokens": window}, max_tokens=8192
    )
    assert model_context_window(model) is None
    assert summarization_threshold(model) == 50000
    assert summarization_threshold(model, fallback=1234) == 1234


def test_bound_model_override_invalidates_profile():
    model = ChatOpenAI.model_construct(
        model_name="gpt-4o", profile={"max_input_tokens": 128000}
    )
    assert summarization_threshold(model.bind(model="gpt-4o")) == 89600
    assert summarization_threshold(model.bind(model="custom-alias")) == 50000


def test_ollama_allocation_overrides_profile_and_respects_bound_options():
    model = ChatOllama.model_construct(
        model="llama3.1",
        num_ctx=8192,
        profile={"max_input_tokens": 131072},
    )
    assert summarization_threshold(model) == 5735
    bound = model.bind(options={"num_ctx": 4096}).with_config(tags=["test"])
    assert model_context_window(bound) == 4096
    assert summarization_threshold(bound) == 2868
    assert (
        summarization_threshold(bound.bind(options={"num_ctx": 8192})) == 5735
    )
    # Other bound options replace num_ctx as well; server defaults are unknown.
    assert (
        summarization_threshold(model.bind(options={"temperature": 0})) == 50000
    )


def test_missing_metadata_does_not_treat_output_limit_as_context():
    assert summarization_threshold(SimpleNamespace(max_tokens=8192)) == 50000
