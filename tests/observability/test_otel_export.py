"""Pre-registered acceptance tests for the #259 Tier 1 OTel bridge repair.

Each test pins one audited defect (D-numbers from the Tier 1 map). On the
unfixed tree every non-characterization test fails for its documented
reason; the repair is done when they flip green with no test edits.

Two tests are labeled CHARACTERIZATION: they pin already-correct behavior
(env-var resolution) and are green before and after the repair.
"""

import json
import subprocess
import sys
import textwrap
from datetime import datetime
from types import MappingProxyType

import pytest

otel = pytest.importorskip("opentelemetry")

from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E402
    OTLPSpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

import ursa.observability.timing as timing  # noqa: E402

_EXPORTERS = []


class RecordingExporter(OTLPSpanExporter):
    """Real exporter (real constructor/config resolution), capture-only."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured = []
        _EXPORTERS.append(self)

    def export(self, spans):
        self.captured.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS


def _payload():
    return {
        "context": {
            "agent": "test_agent",
            "thread_id": "t",
            "run_id": "run-1",
            "started_at": "2026-08-07T00:00:00+00:00",
            "ended_at": "2026-08-07T00:00:05+00:00",
        },
        "tables": {},
        "totals": {},
        "llm_events": [],
    }


@pytest.fixture(autouse=True)
def _patched_exporter(monkeypatch):
    _EXPORTERS.clear()
    monkeypatch.setattr(timing, "OTLPSpanExporter", RecordingExporter)
    yield
    _EXPORTERS.clear()


def _telemetry():
    return timing.Telemetry(enable=True, output_dir="/tmp/unused-otel-tests")


def test_d3_span_exported_before_save_otel_returns():
    telemetry = _telemetry()
    telemetry._save_otel(_payload(), "http://127.0.0.1:19999/v1/traces", None)

    assert _EXPORTERS, "no exporter constructed"
    assert len(_EXPORTERS[-1].captured) >= 1, (
        "span not exported by the time _save_otel returned (no flush)"
    )


def test_d4_headers_accept_env_style_string_and_mapping():
    telemetry = _telemetry()
    telemetry._save_otel(
        _payload(),
        "http://127.0.0.1:19999/v1/traces",
        "authorization=Bearer abc,x-extra=1",
    )
    telemetry._save_otel(
        _payload(),
        "http://127.0.0.1:19999/v1/traces",
        {"authorization": "Bearer abc"},
    )


def test_d4_headers_garbage_raises_ursa_error():
    telemetry = _telemetry()
    with pytest.raises(ValueError, match="otel_headers"):
        telemetry._save_otel(
            _payload(), "http://127.0.0.1:19999/v1/traces", 12345
        )


def test_d4_headers_accept_nondict_mapping():
    telemetry = _telemetry()
    telemetry._save_otel(
        _payload(),
        "http://127.0.0.1:19999/v1/traces",
        MappingProxyType({"authorization": "Bearer abc"}),
    )


def test_d5b_telemetry_field_endpoint_honored():
    telemetry = _telemetry()
    telemetry.otel_endpoint = "http://127.0.0.1:18888/v1/traces"
    telemetry._save_otel(_payload(), None, None)

    assert _EXPORTERS
    assert "18888" in str(_EXPORTERS[-1]._endpoint)


def test_d6_characterization_env_honored_when_unset(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://127.0.0.1:17777/v1/traces",
    )
    telemetry = _telemetry()
    telemetry.otel_endpoint = None
    telemetry._save_otel(_payload(), None, None)

    assert _EXPORTERS
    assert "17777" in str(_EXPORTERS[-1]._endpoint)


def test_d6_characterization_param_beats_env(monkeypatch):
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "http://127.0.0.1:17777/v1/traces",
    )
    telemetry = _telemetry()
    telemetry._save_otel(_payload(), "http://127.0.0.1:16666/v1/traces", None)

    assert _EXPORTERS
    assert "16666" in str(_EXPORTERS[-1]._endpoint)


def test_d1_inprocess_unavailable_structured_branch(caplog, monkeypatch):
    monkeypatch.setattr(timing, "opentelemetry_available", False)
    telemetry = _telemetry()
    with caplog.at_level("WARNING", logger="ursa.observability.otel"):
        result = telemetry._save_otel(_payload(), None, None)

    assert result == {
        "ok": False,
        "endpoint": None,
        "span_count": 0,
        "reason": "otel-unavailable",
    }
    assert any("ursa-ai[otel]" in r.message for r in caplog.records)


def test_d7_structured_result_and_logging(caplog):
    telemetry = _telemetry()
    with caplog.at_level("INFO", logger="ursa.observability.otel"):
        result = telemetry._save_otel(
            _payload(), "http://127.0.0.1:19999/v1/traces", None
        )

    assert isinstance(result, dict), "structured result expected"
    assert result.get("ok") is True
    assert result.get("span_count", 0) >= 1
    assert "19999" in str(result.get("endpoint"))
    assert any("19999" in record.message for record in caplog.records), (
        "success was not logged"
    )


_SUBPROCESS_RUNNER = textwrap.dedent("""
    import json, sys
    {blocker}
    import ursa.observability.timing as timing

    payload = {{
        "context": {{"agent": "a", "thread_id": "t", "run_id": "r",
                     "started_at": "2026-08-07T00:00:00+00:00",
                     "ended_at": "2026-08-07T00:00:05+00:00"}},
        "tables": {{}}, "totals": {{}}, "llm_events": [],
    }}
    {body}
""")


def _run_subprocess(blocker: str, body: str) -> subprocess.CompletedProcess:
    code = _SUBPROCESS_RUNNER.format(blocker=blocker, body=body)
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )


def test_d1_unavailable_extra_warns_instead_of_silence():
    blocker = textwrap.dedent("""
        import importlib.abc, sys
        class _Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.startswith("opentelemetry.exporter.otlp.proto.http"):
                    raise ImportError("blocked for test")
        sys.meta_path.insert(0, _Block())
        for mod in list(sys.modules):
            if mod.startswith("opentelemetry"):
                del sys.modules[mod]
    """)
    body = textwrap.dedent("""
        import logging
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        assert timing.opentelemetry_available is False
        t = timing.Telemetry(enable=True, output_dir="/tmp/unused")
        t._save_otel(payload, "http://127.0.0.1:19999/v1/traces", None)
    """)
    proc = _run_subprocess(blocker, body)
    assert proc.returncode == 0, proc.stderr
    assert "ursa-ai[otel]" in proc.stderr, (
        "no install-hint warning when the otel extra is unavailable: "
        + proc.stderr
    )


def test_d2_two_endpoints_route_independently():
    body = textwrap.dedent("""
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace.export import SpanExportResult

        exporters = []

        class Rec(OTLPSpanExporter):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.captured = []
                exporters.append(self)

            def export(self, spans):
                self.captured.extend(spans)
                return SpanExportResult.SUCCESS

        timing.OTLPSpanExporter = Rec
        t = timing.Telemetry(enable=True, output_dir="/tmp/unused")
        t._save_otel(payload, "http://127.0.0.1:11111/v1/traces", None)
        t._save_otel(payload, "http://127.0.0.1:22222/v1/traces", None)
        import time
        time.sleep(0.2)
        print(json.dumps([len(e.captured) for e in exporters]))
    """)
    proc = _run_subprocess("", body)
    assert proc.returncode == 0, proc.stderr
    counts = json.loads(proc.stdout.strip().splitlines()[-1])
    assert counts == [1, 1], (
        f"spans misrouted across endpoints: {counts} (expected [1, 1])"
    )


# --- T1.2: true-time span tree (root from run context, child per event) ---

_T2_START = "2026-08-07T00:00:00+00:00"
_T2_END = "2026-08-07T00:00:05+00:00"


def _iso_ns(iso: str) -> int:
    return int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000)


def _epoch(iso: str) -> float:
    return datetime.fromisoformat(iso).timestamp()


def _event(
    ok=True,
    t_start=None,
    t_end=None,
    model="test-model",
    node="agent_node",
    step=3,
):
    """Mirror a PerLLMTimer.samples entry (on_llm_end / on_llm_error)."""
    metrics = (
        {
            "usage_rollup": {
                "input_tokens": 11,
                "output_tokens": 7,
                "total_tokens": 18,
            }
        }
        if ok
        else {"error": "RuntimeError('boom')"}
    )
    return {
        "name": f"llm:{model}",
        "ms": 1000.0,
        "ok": ok,
        "tags": [],
        "metadata": {
            "model": model,
            "langgraph_node": node,
            "langgraph_step": step,
        },
        "metrics": metrics,
        "t_start": t_start,
        "t_end": t_end,
    }


def _payload_with_events(events):
    payload = _payload()
    payload["llm_events"] = list(events)
    return payload


def test_t2_root_span_uses_real_run_timestamps():
    telemetry = _telemetry()
    telemetry._save_otel(_payload(), "http://127.0.0.1:19999/v1/traces", None)

    assert _EXPORTERS
    (root,) = _EXPORTERS[-1].captured
    assert root.start_time == _iso_ns(_T2_START), (
        "root span start is not the run's real started_at"
    )
    assert root.end_time == _iso_ns(_T2_END), (
        "root span end is not the run's real ended_at"
    )


def test_t2_child_span_per_llm_event_within_root():
    base = _epoch(_T2_START)
    events = [
        _event(t_start=base + 1.0, t_end=base + 2.0),
        _event(t_start=base + 2.5, t_end=base + 4.0),
    ]
    telemetry = _telemetry()
    result = telemetry._save_otel(
        _payload_with_events(events), "http://127.0.0.1:19999/v1/traces", None
    )

    captured = _EXPORTERS[-1].captured
    assert len(captured) == 3, (
        f"expected 1 root + 2 children, got {len(captured)}"
    )
    roots = [s for s in captured if s.parent is None]
    assert len(roots) == 1, "expected exactly one root span"
    root = roots[0]
    children = [s for s in captured if s.parent is not None]
    assert all(c.parent.span_id == root.context.span_id for c in children), (
        "children are not parented to the run root span"
    )
    assert sorted(c.start_time for c in children) == [
        int((base + 1.0) * 1_000_000_000),
        int((base + 2.5) * 1_000_000_000),
    ]
    assert sorted(c.end_time for c in children) == [
        int((base + 2.0) * 1_000_000_000),
        int((base + 4.0) * 1_000_000_000),
    ]
    for child in children:
        assert (
            root.start_time
            <= child.start_time
            <= child.end_time
            <= root.end_time
        ), "child span falls outside the root span interval"
    assert result["span_count"] == 3


def test_t2_child_attributes_follow_genai_semconv():
    base = _epoch(_T2_START)
    telemetry = _telemetry()
    telemetry._save_otel(
        _payload_with_events([_event(t_start=base + 1.0, t_end=base + 2.0)]),
        "http://127.0.0.1:19999/v1/traces",
        None,
    )

    children = [s for s in _EXPORTERS[-1].captured if s.parent is not None]
    assert len(children) == 1
    child = children[0]
    assert child.name == "chat test-model"
    attrs = dict(child.attributes)
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["gen_ai.request.model"] == "test-model"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["ursa.langgraph.node"] == "agent_node"
    assert attrs["ursa.langgraph.step"] == 3


def test_t2_error_event_child_has_error_status():
    base = _epoch(_T2_START)
    events = [
        _event(t_start=base + 1.0, t_end=base + 2.0),
        _event(ok=False, t_start=base + 2.0, t_end=base + 3.0),
    ]
    telemetry = _telemetry()
    telemetry._save_otel(
        _payload_with_events(events), "http://127.0.0.1:19999/v1/traces", None
    )

    children = {
        c.start_time: c for c in _EXPORTERS[-1].captured if c.parent is not None
    }
    ok_child = children[int((base + 1.0) * 1_000_000_000)]
    err_child = children[int((base + 2.0) * 1_000_000_000)]
    assert err_child.status.status_code is StatusCode.ERROR, (
        "ok=False event did not produce an ERROR-status span"
    )
    assert ok_child.status.status_code is StatusCode.UNSET


def test_t2_no_events_root_only():
    # Guard (green before and after T1.2): empty llm_events yields exactly
    # one root span and span_count 1.
    telemetry = _telemetry()
    result = telemetry._save_otel(
        _payload(), "http://127.0.0.1:19999/v1/traces", None
    )

    captured = _EXPORTERS[-1].captured
    assert len(captured) == 1
    assert captured[0].parent is None
    assert result["span_count"] == 1


def test_t2_missing_run_timestamps_still_exports():
    # Robustness guard (green before and after T1.2): a degenerate context
    # without timestamps must not crash and still exports a root span.
    payload = _payload()
    del payload["context"]["started_at"]
    del payload["context"]["ended_at"]
    telemetry = _telemetry()
    result = telemetry._save_otel(
        payload, "http://127.0.0.1:19999/v1/traces", None
    )

    assert result["ok"] is True
    assert len(_EXPORTERS[-1].captured) == 1
