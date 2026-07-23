# Event logging

URSA reports agent, tool, workflow, and environment progress through structured
[LangChain custom events](https://python.langchain.com/docs/concepts/callbacks/).
Core execution code describes what happened; callbacks and renderers decide how
that information appears.

Use events whenever a developer, user, or interface needs to know that something
happened during a run. Put presentation logic in a callback or renderer instead
of using `print(...)`, `console.print(...)`, or Rich directly in agents and
tools.

See the [Python scripts guide][getting-started-python-scripts] for running
agents, the [CLI guide][getting-started-cli] for URSA's interactive interface,
and the [dashboard guide][getting-started-web-dashboard] for the web interface.

## Event model

### Shared event channel

URSA publishes progress under one
[`DEFAULT_EVENT_NAME`][ursa.util.events.DEFAULT_EVENT_NAME]:

```python
from ursa.util.events import DEFAULT_EVENT_NAME

assert DEFAULT_EVENT_NAME == "ursa_agent_progress"
```

The helpers in `ursa.util.events` add source identity, timestamps, and lifecycle
fields consistently:

- [`AgentEvents`][ursa.util.events.AgentEvents] reports
  progress from agents and workflow nodes.
- [`EnvironmentEvents`][ursa.util.events.EnvironmentEvents]
  reports progress from agent teams and symposia.
- [`ToolEvents`][ursa.util.events.ToolEvents] reports
  tool progress and preserves
  [`ToolRuntime`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime)
  metadata.
- [`emit(...)`][ursa.util.events.ProgressEvents.emit]
  and [`aemit(...)`][ursa.util.events.ProgressEvents.aemit]
  publish one event.
- [`range(...)`][ursa.util.events.ProgressEvents.range]
  emits start, end, and error lifecycle events around an operation.
- [`configure_event_logging(...)`][ursa.util.events.configure_event_logging]
  provides a default console view for scripts.

### Payload fields

Event payloads are small structured dictionaries. A completed tool event may
look like this:

```python
{
    "tool": "write_code",
    "tool_call_id": "call_abc123",
    "stage": "write",
    "phase": "end",
    "message": "File written",
    "monotonic_timestamp_ns": 123456789,
    "elapsed_ms": 42.5,
    "path": "/workspace/example.py",
}
```

| Field | Meaning |
| --- | --- |
| `agent`, `environment`, or `tool` | Event source, added by the corresponding helper. |
| `stage` | Stable machine-readable operation name such as `generate`, `search`, or `write`. |
| `message` | Concise human-readable status. |
| `phase` | Lifecycle state: `start`, `end`, or `error`. Ranges add it automatically. |
| `monotonic_timestamp_ns` | Monotonic timestamp used to order events. |
| `elapsed_ms` | Duration added to terminal range events. |
| `error`, `error_type` | Failure details. |
| `artifact` | One MIME-typed artifact. |
| `artifacts` | Multiple MIME-typed artifacts. Standard rendering places them side by side. |

Additional fields may contain strings, numbers, booleans, lists, or dictionaries.
Keep payloads serializable, bounded, and free of secrets.

### MIME-typed artifacts

Artifacts carry output that benefits from richer presentation while keeping the
event envelope renderer-independent. Create one with
[`event_artifact(...)`][ursa.util.rendering.event_artifact]:

```python
from ursa.util.rendering import event_artifact

artifact = event_artifact(
    diff,
    "text/x-diff",
    metadata={"title": "Edit diff", "path": str(path)},
)
events.emit(
    "File updated",
    stage="edit",
    phase="end",
    artifact=artifact,
)
```

Artifact `content` may be any serializable value. Metadata values are strings,
integers, or floats; `title` and `path` are well-known entries used by the
standard renderer.

Use [`file_artifact(...)`][ursa.util.rendering.file_artifact]
for files that were read or written. It emits the path,
not the entire file. The console renderer dereferences small text files when it
displays the event and applies syntax highlighting. Missing, binary, oversized,
or undecodable files fall back to displaying their path.

```python
from ursa.util.rendering import file_artifact

span.update(artifact=file_artifact(path, title="File written"))
```

Commands emit separate non-empty `text/plain` artifacts for stdout and stderr.
When both streams contain text, the standard formatter renders their panels
side by side.

Built-in rendering supports file references, JSON, Markdown, diffs, and plain
text. Interfaces can call
[`register_artifact_renderer(...)`][ursa.util.rendering.register_artifact_renderer]
to add or replace a MIME renderer.

## Formatting events

URSA provides two complementary ways to format and consume events. Scripts can
install the standard Python logging formatter, while applications and LangGraph
workflows can attach callbacks and choose their own presentation. Both consume
the same structured event payloads and MIME-typed artifacts.

### Configure logging in scripts

URSA uses Python's standard logging system. See the
[Python Logging HOWTO][logging-howto] for logger, handler, level, and
configuration concepts.

Call
[`configure_event_logging()`][ursa.util.events.configure_event_logging]
once before invoking an agent:

```python
from ursa.util.events import configure_event_logging

configure_event_logging()
result = agent.invoke(inputs)
```

URSA events are shown at `INFO` while unrelated libraries remain at `WARNING`,
avoiding noisy dependency logs such as HTTP request traces. A typical summary is:

```text
[ursa] write_code write/end: File written (path=workspace/example.py)
```

Rich artifact rendering is enabled by default. Disable artifact bodies while
retaining event summaries with:

```python
configure_event_logging(rich=False)
```

You can also supply a standard
[`logging.Formatter`][]
through `formatter=` or change the URSA logging level through `level=`.

### Format events with LangGraph callbacks

LangGraph applications should consume structured callbacks instead of parsing
console output. Attach a callback through the underlying
[LangChain runnable configuration](https://python.langchain.com/docs/concepts/runnables/):

```python
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from ursa.util.events import DEFAULT_EVENT_NAME


class MyProgressHandler(BaseCallbackHandler):
    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id,
        **kwargs,
    ) -> None:
        if name != DEFAULT_EVENT_NAME or not isinstance(data, dict):
            return

        source = (
            data.get("agent")
            or data.get("environment")
            or data.get("tool")
            or "ursa"
        )
        self.handle_progress(source, data)


handler = MyProgressHandler()
result = agent.invoke(inputs, config={"callbacks": [handler]})
```

Use the same configuration with `await agent.ainvoke(...)`. Pass callbacks at
the top-level invocation when possible so nested agents and tools share one
event stream.

The CLI uses
[`HITLLogEventHandler`][ursa.cli.callbacks.HITLLogEventHandler]. The
default
[`EventLoggingHandler`][ursa.util.events.EventLoggingHandler]
forwards structured events into Python logging, and
[`EventConsoleFormatter`][ursa.util.rendering.EventConsoleFormatter]
provides the standard human-readable view.

## Using events

Events are the default way to expose progress from core URSA code. If a user,
developer, or interface needs to know that something happened during a run,
emit an event. If code needs to decide how that event looks, use a callback or
renderer.

Before adding or changing event-producing code:

- Emit events instead of printing progress from agents, tools, and workflows.
- Accept `config: RunnableConfig | None = None` in event-emitting graph nodes.
- Create tool events with
  [`ToolEvents.from_runtime(...)`][ursa.util.events.ToolEvents.from_runtime].
- Use
  [`events.range(...)`][ursa.util.events.ProgressEvents.range]
  for operations with a clear start and finish.
- Choose stable `stage` values and concise messages.
- Emit `phase="error"` on handled failure paths.
- Keep payloads small, structured, safe, and serializable.
- Represent displayable output with MIME-typed artifacts.
- Keep Rich and other presentation logic in interfaces and renderers.
- Pass callbacks through `config={"callbacks": [...]}`.
- Test important event payloads and failure paths.

### Build events into an agent

Agent graph nodes should accept the active
[`RunnableConfig`](https://reference.langchain.com/python/langchain-core/runnables/#langchain_core.runnables.RunnableConfig)
and call
[`self.events(config)`][ursa.agents.base.BaseAgent.events]:

```python
from langchain_core.runnables import RunnableConfig


class MyAgent(BaseAgent):
    def generation_node(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        events = self.events(config)

        with events.range(
            "generate",
            "Drafting answer",
            done="Answer drafted",
            error="Answer generation failed",
        ) as span:
            answer = self.llm.invoke(state["messages"])
            span.update(result_chars=len(str(answer)))

        return {"messages": [answer]}
```

Use
[`await events.aemit(...)`][ursa.util.events.ProgressEvents.aemit]
and
[`async with events.range(...)`][ursa.util.events.ProgressEvents.range]
in async graph nodes.

### Build events into a tool

Tools construct events from their
[`ToolRuntime`](https://reference.langchain.com/python/langchain/tools/#langchain.tools.ToolRuntime),
which preserves the runnable configuration, tool call ID, and owning agent or
environment metadata:

```python
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents
from ursa.util.rendering import file_artifact


@tool
def read_custom_file(
    path: str,
    runtime: ToolRuntime[AgentContext],
) -> str:
    events = ToolEvents.from_runtime("read_custom_file", runtime)
    target = runtime.context.workspace / path

    with events.range(
        "read",
        "Reading file",
        done="File read",
        error="File read failed",
        path=str(target),
    ) as span:
        text = target.read_text()
        span.update(artifact=file_artifact(target, title="File read"))

    return text
```

If a tool handles an error instead of allowing it to escape a range, emit the
error event explicitly:

```python
events.emit(
    "Invalid file path",
    stage="read",
    phase="error",
    path=path,
    error="Path is outside the workspace",
)
return "Failed: invalid file path"
```

### Build an interface or renderer

Interfaces should interpret event fields and artifacts without requiring core
execution code to know about UI details:

```python
class UiProgressHandler(BaseCallbackHandler):
    def on_custom_event(self, name, data, *, run_id, **kwargs):
        if name != DEFAULT_EVENT_NAME or not isinstance(data, dict):
            return

        if "agent" in data:
            self.render_agent_event(data)
        elif "tool" in data:
            self.render_tool_event(data)
```

Use
[`render_event_artifact(...)`][ursa.util.rendering.render_event_artifact]
for one artifact and
[`render_event_artifacts(...)`][ursa.util.rendering.render_event_artifacts]
for a collection. The latter guarantees an equal-width, side-by-side layout.
For application-specific presentation, use the structured artifact content
directly or register another MIME renderer.

### Name stages and messages

Use stable snake_case stage names. Request/result pairs work well for one-off
events:

- `generate` / `generate_result`
- `reflect` / `reflect_result`
- `search` / `search_result`
- `summarize` / `summarize_result`

Use one stage with lifecycle phases for scoped work:

- `write` with `phase=start|end|error`
- `edit` with `phase=start|end|error`
- `execute` with `phase=start|end|error`
- `download` with `phase=start|end|error`

Messages should be short and readable:

```python
events.emit("Searching Web", stage="search", query=query)
events.emit(
    "Web search complete",
    stage="search_result",
    result_chars=len(result),
)
```

### Keep payloads safe and useful

Include paths, query strings, result sizes, return codes, boolean status fields,
correlation IDs, and elapsed time when they help consumers.

Do not include secrets, credentials, unbounded model responses, or UI markup.
Use artifacts for intentional display output and counts or short previews for
diagnostic metadata.

### Test event behavior

Core tests should assert structured payloads rather than rendered strings. Test
the event name, source, stage, phase, message, useful metadata, artifacts, and
handled failure events. Rendering tests should separately cover callbacks,
formatters, MIME renderers, and multi-artifact layout.

## API reference

See the generated [utility API reference][utility-api-reference], under
"Event logging," for event helpers, artifact constructors, and rendering APIs.
