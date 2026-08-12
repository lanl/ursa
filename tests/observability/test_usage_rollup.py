"""Acceptance tests for the 165 usage-rollup repair.

Pre-registered before the fix: reasoning and cached counts must be
captured from langchain's standardized usage_metadata detail containers
and from the raw side-channel namings that several providers use, with
exact non-double-counting semantics. The guard tests pin the paths that
already work today.
"""

import tempfile
import uuid
from collections.abc import Iterator
from types import SimpleNamespace

from langchain_core.language_models.fake_chat_models import (
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from ursa.agents.chat_agent import ChatAgent
from ursa.observability.metrics_charts import extract_llm_token_stats
from ursa.observability.timing import PerLLMTimer

_RAW_CC = {
    "prompt_tokens": 100,
    "completion_tokens": 80,
    "total_tokens": 180,
    "completion_tokens_details": {"reasoning_tokens": 30},
    "prompt_tokens_details": {"cached_tokens": 20},
}

_STANDARDIZED = {
    "input_tokens": 100,
    "output_tokens": 80,
    "total_tokens": 180,
    "input_token_details": {"cache_read": 20},
    "output_token_details": {"reasoning": 30},
}


def _roll(message=None, llm_output=None, messages=None):
    timer = PerLLMTimer()
    run_id = uuid.uuid4()
    timer.on_llm_start({}, [], run_id=run_id, metadata={"model": "m"})
    msgs = messages if messages is not None else [message]
    generations = [[ChatGeneration(message=m) for m in msgs]]
    timer.on_llm_end(
        LLMResult(generations=generations, llm_output=llm_output),
        run_id=run_id,
    )
    return timer.samples[-1]["metrics"].get("usage_rollup") or {}


def test_standardized_details_captured():
    # Issue 165: reasoning and cache_read arriving only in the
    # standardized usage_metadata details (the Gemini, OpenAI Responses,
    # and streaming shapes) must be captured.
    roll = _roll(message=AIMessage("x", usage_metadata=dict(_STANDARDIZED)))

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_responses_raw_naming_captured():
    # Defensive: the Responses-API raw container names.
    roll = _roll(
        message=AIMessage("x"),
        llm_output={
            "token_usage": {
                "input_tokens": 100,
                "output_tokens": 80,
                "total_tokens": 180,
                "output_tokens_details": {"reasoning_tokens": 30},
                "input_tokens_details": {"cached_tokens": 20},
            }
        },
    )

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_anthropic_cache_captured_exactly():
    # Anthropic shape with BOTH carriers: cached must be exactly the
    # cache-read count; cache_creation stays out of cached_tokens
    # (pricing treats cached as the cache-read discount).
    roll = _roll(
        message=AIMessage(
            "x",
            usage_metadata={
                "input_tokens": 120,
                "output_tokens": 80,
                "total_tokens": 200,
                "input_token_details": {
                    "cache_read": 20,
                    "cache_creation": 5,
                },
            },
            response_metadata={
                "usage": {
                    "input_tokens": 95,
                    "output_tokens": 80,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                }
            },
        )
    )

    assert roll.get("cached_tokens") == 20


def test_anthropic_raw_thinking_captured():
    # Current Anthropic SDKs expose thinking counts in the raw usage the
    # adapter forwards; the defensive synonym must capture them.
    roll = _roll(
        message=AIMessage(
            "x",
            response_metadata={
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 80,
                    "output_tokens_details": {"thinking_tokens": 30},
                }
            },
        )
    )

    assert roll.get("reasoning_tokens") == 30


def test_service_tier_prefixed_details_captured():
    # langchain-openai prefixes detail keys with the service tier; the
    # bare tier bucket key must NOT be counted as reasoning.
    roll = _roll(
        message=AIMessage(
            "x",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 80,
                "total_tokens": 180,
                "input_token_details": {"flex_cache_read": 20},
                "output_token_details": {"flex_reasoning": 30, "flex": 999},
            },
        )
    )

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_two_generations_sum_reasoning():
    # Per-message summation is preserved: two generations each carrying
    # standardized reasoning must sum.
    roll = _roll(
        messages=[
            AIMessage("a", usage_metadata=dict(_STANDARDIZED)),
            AIMessage("b", usage_metadata=dict(_STANDARDIZED)),
        ]
    )

    assert roll.get("reasoning_tokens") == 60
    assert roll.get("cached_tokens") == 40


def _standardized_stream() -> Iterator[AIMessage]:
    while True:
        yield AIMessage("ok", usage_metadata=dict(_STANDARDIZED))


class _ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_end_to_end_chart_surface_nonzero():
    # The exact aggregation surface from the issue's screenshot must see
    # the counts once extraction captures them.
    with tempfile.TemporaryDirectory() as tmp:
        agent = ChatAgent(
            llm=_ToolReadyFakeChatModel(messages=_standardized_stream()),
            workspace=tmp,
        )
        agent.invoke({"messages": [], "query": "hello"})
        payload = agent.telemetry.to_json(
            include_raw_snapshot=False, include_raw_records=False
        )

    totals, _samples = extract_llm_token_stats(payload)
    assert totals.get("reasoning_tokens") == 30
    assert totals.get("cached_tokens") == 20


def test_raw_chat_completions_llm_output_pinned():
    # Guard (green before and after): the raw Chat Completions rescue.
    roll = _roll(
        message=AIMessage("x"), llm_output={"token_usage": dict(_RAW_CC)}
    )

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_raw_response_metadata_pinned():
    # Guard (green before and after): raw usage in response_metadata.
    roll = _roll(
        message=AIMessage("x", response_metadata={"token_usage": dict(_RAW_CC)})
    )

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_attr_object_coercion_pinned():
    # Guard (green before and after): attribute-object usage coercion.
    roll = _roll(
        message=AIMessage("x"),
        llm_output={
            "token_usage": SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=80,
                total_tokens=180,
                completion_tokens_details=SimpleNamespace(reasoning_tokens=30),
                prompt_tokens_details=SimpleNamespace(cached_tokens=20),
            )
        },
    )

    assert roll.get("reasoning_tokens") == 30
    assert roll.get("cached_tokens") == 20


def test_both_carriers_no_double_count():
    # Guard: with the standardized carrier AND the raw side-channel both
    # present (the real ChatOpenAI shape), counts appear exactly once.
    roll = _roll(
        message=AIMessage("x", usage_metadata=dict(_STANDARDIZED)),
        llm_output={"token_usage": dict(_RAW_CC)},
    )

    assert roll.get("reasoning_tokens") == 30, "double-counted reasoning"
    assert roll.get("cached_tokens") == 20, "double-counted cached"


def test_empty_details_clean():
    # Guard: empty detail containers stay zero without crashing.
    roll = _roll(
        message=AIMessage(
            "x",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 80,
                "total_tokens": 180,
                "input_token_details": {},
                "output_token_details": {},
            },
        )
    )

    assert roll.get("reasoning_tokens") == 0
    assert roll.get("cached_tokens") == 0


def test_dual_names_single_dict_no_double():
    # Guard: one dict carrying the same count under two namings yields
    # the count once.
    roll = _roll(
        message=AIMessage(
            "x",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 80,
                "total_tokens": 180,
                "completion_tokens_details": {"reasoning_tokens": 30},
                "output_token_details": {"reasoning": 30},
            },
        )
    )

    assert roll.get("reasoning_tokens") == 30
