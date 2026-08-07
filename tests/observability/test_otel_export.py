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

import pytest

otel = pytest.importorskip("opentelemetry")

from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E402
    OTLPSpanExporter,
)

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
