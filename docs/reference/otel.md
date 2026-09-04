# OpenTelemetry export

URSA can export a per-run trace over OTLP/HTTP. Export is opt-in per
invocation and requires the `otel` extra:

```bash
pip install "ursa-ai[otel]"
```

Without the extra, `save_otel=True` logs a warning carrying this install
hint and the run continues normally.

## Enabling export

Pass `save_otel=True` to `invoke` or `ainvoke`. The endpoint and headers
can ride along on the same call:

```python
agent.invoke(
    inputs,
    save_otel=True,
    otel_endpoint="http://127.0.0.1:4318/v1/traces",
    otel_headers={"authorization": "Bearer TOKEN"},
)
```

`otel_headers` accepts a mapping or an env-style string such as
`"authorization=Bearer TOKEN,x-extra=1"`. Any other type raises a
`ValueError`. A malformed non-empty string parses to an empty header set
rather than failing, matching the OpenTelemetry SDK's liberal parsing of
its own headers variables.

## Endpoint resolution order

| Priority | Source | Notes |
|---|---|---|
| 1 | `otel_endpoint=` on `invoke`/`ainvoke` | Used exactly as given. |
| 2 | The `Telemetry.otel_endpoint` field | Set it on `agent.telemetry`. |
| 3 | `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | Standard SDK variable, used verbatim. |
| 4 | `OTEL_EXPORTER_OTLP_ENDPOINT` | Base URL; the SDK appends `/v1/traces`. |
| 5 | SDK default | `http://localhost:4318/v1/traces`. |

When no explicit headers are given, the SDK's standard
`OTEL_EXPORTER_OTLP_HEADERS` and `OTEL_EXPORTER_OTLP_TRACES_HEADERS`
variables apply as well.

### Example: local OpenTelemetry Collector

```python
# The SDK default already points at a local collector's OTLP/HTTP port.
agent.invoke(inputs, save_otel=True)
```

### Example: MLflow tracking server

MLflow ingests OTLP over HTTP on its tracking port and routes spans to an
experiment through a header:

```python
agent.invoke(
    inputs,
    save_otel=True,
    otel_endpoint="http://127.0.0.1:5000/v1/traces",
    otel_headers={"x-mlflow-experiment-id": "123456"},
)
```

## What gets exported

Each export produces one trace:

- A root span named with the run id, spanning the run's real
  `started_at` to `ended_at` and carrying the run's time-breakdown and
  token-total attributes.
- One child span per recorded LLM call, placed at the call's real
  wall-clock interval, named `chat <model>` (bare `chat` when the model
  name is unavailable) with `SpanKind.CLIENT` and, when the recorder
  captured them, the attributes `gen_ai.operation.name`,
  `gen_ai.request.model`, `gen_ai.provider.name`,
  `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `ursa.langgraph.node`, and `ursa.langgraph.step`. Calls that failed
  are exported with ERROR span status and the error text as the status
  description.

Attribute names follow the OpenTelemetry GenAI semantic conventions,
which are in Development status and maintained in the
[semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)
repository; names may still evolve upstream.

The export path builds a private, per-call tracer provider. URSA never
touches the global OpenTelemetry provider, so it cannot interfere with
other OTel users in the same process, such as MLflow or chromadb.

## Semantics and limits

- A successful export means the spans were handed to the exporter and
  flushed within a 10 second timeout. Delivery to the collector is not
  confirmed at this layer; the SDK's exporter logs delivery failures
  itself.
- Rendering the same run twice with `save_otel=True` exports it twice,
  as two distinct traces.
- Timestamps are exported exactly as recorded, without clamping. A
  payload missing one of `started_at`/`ended_at` falls back to the
  export-time clock for that bound, which can distort the root span's
  duration.
- Token counts of zero are omitted rather than exported as `0`.
- LLM calls recorded without timestamps are skipped, with the count
  noted in a debug log. Failed calls carry ERROR status but not the
  `error.type` attribute.
