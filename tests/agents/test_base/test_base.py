import json
from pathlib import Path
from typing import Annotated, Any, Mapping, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

# LangChain core bits
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# Your project imports
from ursa.agents.base import BaseAgent


# --- Tiny offline model that triggers LLM callbacks and returns usage ---
class TinyCountingModel(BaseChatModel):
    """
    Offline fake chat model:
      - Pretends to be "openai:o3" (so it matches pricing.json keys)
      - Emits small token usage numbers (4 in, 5 out)
      - Never calls any external APIs
    """

    model: str = "openai:o3"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        # Simulate a small response with usage metadata
        usage = {"input_tokens": 4, "output_tokens": 5, "total_tokens": 9}
        # Also mirror provider-style fields in response_metadata for robustness
        response_metadata = {
            "model": self.model,
            "token_usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }
        ai = AIMessage(
            content="ok",
            response_metadata=response_metadata,
            usage_metadata=usage,
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])


class TestState(TypedDict, total=False):
    messages: Annotated[list, add_messages]


# --- Minimal agent under test (subclasses BaseAgent and makes one LLM call) ---
class TestAgent(BaseAgent):
    def __init__(
        self,
        llm,
        checkpointer=None,
        enable_metrics=False,
        metrics_dir="ursa_metrics",
        autosave_metrics=True,
        **kwargs,
    ):
        super().__init__(
            llm,
            checkpointer,
            enable_metrics,
            metrics_dir,
            autosave_metrics,
            **kwargs,
        )
        self.graph = self._build_graph()
        self._action = self.graph

    def _run_impl(self, state: TestState):
        # Make one LLM call with callbacks + metadata wired in via build_config()
        cfg = self.build_config(tags=["TestAgent"])
        _ = self.llm.invoke(state["messages"], config=cfg)
        # Return something that looks like your agents' shape
        return {"messages": [AIMessage(content="done")]}

    def _build_graph(self):
        builder = StateGraph(TestState)
        builder.add_node(
            "run_impl",
            self._wrap_node(self._run_impl, "run_impl", "test"),
        )
        builder.set_entry_point("run_impl")
        builder.set_finish_point("run_impl")

        graph = builder.compile()
        return graph

    def _invoke(
        self, inputs: Mapping[str, Any], recursion_limit: int = 100000, **_
    ):
        config = self.build_config(
            recursion_limit=recursion_limit, tags=["graph"]
        )
        return self._action.invoke(inputs, config)


@pytest.fixture
def pricing_file(tmp_path: Path) -> Path:
    """
    Create a tiny pricing.json that matches the fake model ("openai:o3").
    Prices are per 1K tokens.
      input:  $0.002 / 1K
      output: $0.008 / 1K
    """
    data = {
        "_note": "Test pricing file for unit test",
        "openai:o3": {
            "input_per_1k": 0.002,
            "output_per_1k": 0.008,
            "cached_input_multiplier": 0.25,
        },
    }
    p = tmp_path / "pricing.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


def _read_single_metrics_file(metrics_dir: Path) -> dict:
    files = sorted(metrics_dir.glob("*.json"))
    assert files, f"No metrics JSON found in {metrics_dir}"
    # pick the newest
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def test_base_agent_metrics_and_pricing(
    tmp_path: Path, monkeypatch, pricing_file: Path
):
    """
    End-to-end: BaseAgent.invoke() triggers telemetry + pricing using a fake chat model.
    Asserts:
      - llm_events present with 1 call
      - usage_rollup matches fake counts
      - costs computed from pricing.json
      - totals sane
    """
    # Make a dedicated metrics dir for this test
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Point loader at our local pricing.json
    monkeypatch.setenv("URSA_PRICING_JSON", str(pricing_file))

    # Instantiate agent with metrics enabled and autosave on
    agent = TestAgent(
        llm=TinyCountingModel(),
        enable_metrics=True,
        autosave_metrics=True,
        metrics_dir=str(metrics_dir),
    )

    # Run once (no network); prints telemetry and writes JSON
    out = agent.invoke("hello tiny model")
    assert isinstance(out, dict)
    assert "messages" in out

    # Read the latest metrics JSON
    payload = _read_single_metrics_file(metrics_dir)

    # Basic structure present
    assert "llm_events" in payload
    assert isinstance(payload["llm_events"], list)
    assert len(payload["llm_events"]) >= 1

    # Pick the last event
    ev = payload["llm_events"][-1]
    assert ev["ok"] is True
    assert ev["metadata"]["model"] == "openai:o3"

    # Usage rollup should reflect our fake numbers
    roll = ev["metrics"]["usage_rollup"]
    assert roll["input_tokens"] == 4
    assert roll["output_tokens"] == 5
    assert roll["total_tokens"] == 9
    assert roll["prompt_tokens"] == 4
    assert roll["completion_tokens"] == 5
    assert roll["reasoning_tokens"] == 0
    assert roll["cached_tokens"] == 0

    # Costs block computed and totals correct:
    # input: 4 * 0.002 / 1000 = 0.000008
    # output: 5 * 0.008 / 1000 = 0.000040
    # total = 0.000048
    assert "costs" in payload
    total_usd = payload["costs"]["total_usd"]
    by_model = payload["costs"]["by_model_usd"]
    assert pytest.approx(total_usd, rel=1e-9, abs=1e-9) == 0.000048
    assert pytest.approx(by_model["openai:o3"], rel=1e-9, abs=1e-9) == 0.000048

    # Per-event cost details should be annotated as computed
    assert ev["metrics"]["cost_source"] == "computed"
    cd = ev["metrics"]["cost_details"]["components_usd"]
    assert pytest.approx(cd["input_cost"], rel=1e-9, abs=1e-9) == 0.000008
    assert pytest.approx(cd["output_cost"], rel=1e-9, abs=1e-9) == 0.000040
    assert pytest.approx(cd["total_cost"], rel=1e-9, abs=1e-9) == 0.000048


def test_metrics_toggle_off(tmp_path: Path, monkeypatch, pricing_file: Path):
    """
    When metrics=False, callbacks are disabled and render() prints nothing.
    No JSON should be saved to metrics_dir.
    """
    metrics_dir = tmp_path / "metrics_off"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Still set pricing, but it shouldn't be used
    monkeypatch.setenv("URSA_PRICING_JSON", str(pricing_file))

    agent = TestAgent(
        llm=TinyCountingModel(),
        enable_metrics=False,  # <-- disable metrics
        autosave_metrics=True,  # ignored when metrics disabled
        metrics_dir=str(metrics_dir),
    )

    _ = agent.invoke("hello")
    # No files should be created
    files = list(metrics_dir.glob("*.json"))
    assert files == []
