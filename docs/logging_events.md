# URSA Event-Based Logging Guide

URSA uses structured LangChain custom events for progress logging. New development should emit events from core agents/tools/workflows and render those events in interfaces, callbacks, dashboards, or applications.

The goal is simple:

> Core URSA code reports what happened; UI/application code decides how to display, store, or react to it.

Avoid adding progress `print(...)`, `console.print(...)`, or Rich UI rendering inside agents and tools. Use events instead.

## Core concepts

URSA progress events are LangChain custom events emitted under one shared event name:

```python
from ursa.util.events import DEFAULT_EVENT_NAME

DEFAULT_EVENT_NAME == "ursa_agent_progress"
```

Use the helpers in `ursa.util.events`:

```python
from ursa.util.events import (
    AgentEvents,
    ToolEvents,
    configure_event_logging,
    DEFAULT_EVENT_NAME,
)
```

Main helpers:

- `AgentEvents`: emit events from agents/workflow nodes.
- `ToolEvents`: emit events from tools using `ToolRuntime`.
- `events.emit(...)`: emit a single event.
- `events.aemit(...)`: async version of `emit`.
- `events.range(...)`: context manager that emits start/end/error lifecycle events.
- `configure_event_logging()`: simple console logging setup for scripts and examples.

## Event payload shape

All progress payloads should be small, structured dictionaries.

Agent event example:

```python
{
    "agent": "PlanningAgent",
    "stage": "generate_result",
    "message": "Drafted plan",
    "monotonic_timestamp_ns": 123456789,
    "steps": [...],
}
```

Tool event example:

```python
{
    "tool": "write_code",
    "tool_call_id": "call_abc123",
    "stage": "write",
    "phase": "end",
    "message": "File written",
    "path": "/workspace/example.py",
    "elapsed_ms": 42.5,
}
```

Recommended fields:

| Field | Use |
| --- | --- |
| `agent` or `tool` | Source of the event. Added automatically by `AgentEvents`/`ToolEvents`. |
| `stage` | Required stable machine-readable stage, e.g. `generate`, `search`, `write`. |
| `message` | Required concise human-readable status. |
| `phase` | Optional lifecycle phase: `start`, `end`, or `error`. Automatically added by `events.range(...)`. |
| `monotonic_timestamp_ns` | Added automatically for ordering. |
| `elapsed_ms` | Added automatically to terminal range events. |
| `error`, `error_type` | Use on failure events. |

Extra fields are allowed, but keep them simple and safe: strings, numbers, booleans, lists, or dictionaries.

## Using events in scripts and examples

For normal scripts and examples, enable console rendering once at startup:

```python
from ursa.util.events import configure_event_logging

configure_event_logging()

result = agent.invoke(inputs)
```

This causes URSA progress events to be logged in a compact readable form, for example:

```text
[ursa] write_code write/end: File written (path=workspace/example.py)
```

Use this in examples where developers should see progress without writing a custom callback.

## Subscribing to events in applications

Applications should attach callbacks through the LangChain runnable config.

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
        tags=None,
        metadata=None,
        **kwargs,
    ) -> None:
        if name != DEFAULT_EVENT_NAME or not isinstance(data, dict):
            return

        source = data.get("agent") or data.get("tool") or "ursa"
        print(f"{source}: {data.get('message')}")

agent.invoke(inputs, config={"callbacks": [MyProgressHandler()]})
```

Async invocation uses the same pattern:

```python
await agent.ainvoke(inputs, config={"callbacks": [MyProgressHandler()]})
```

Applications embedding URSA workflows should prefer callbacks over stdout capture. Callbacks are structured, safer, and easier to test.

## Building events into new agents

Agent graph nodes that emit events should accept an optional `RunnableConfig` argument and call `self.events(config)`.

```python
from langchain_core.runnables import RunnableConfig

class MyAgent(BaseAgent):
    def generation_node(
        self,
        state: dict,
        config: RunnableConfig | None = None,
    ) -> dict:
        events = self.events(config)
        events.emit("Drafting answer", stage="generate")

        answer = self.llm.invoke(state["messages"])

        events.emit(
            "Answer drafted",
            stage="generate_result",
            preview=str(answer)[:2000],
        )
        return {"messages": [answer]}
```

For work with a clear beginning and end, use `events.range(...)`:

```python
with events.range(
    "generate",
    "Drafting answer",
    done="Answer drafted",
    error="Answer generation failed",
) as span:
    answer = self.llm.invoke(state["messages"])
    span.update(result_chars=len(str(answer)))
```

This emits:

1. `phase="start"` when entering the block.
2. `phase="end"` when the block succeeds.
3. `phase="error"` with error details if an exception escapes.

For async code:

```python
await events.aemit("Starting async work", stage="start")

async with events.range(
    "download",
    "Downloading data",
    done="Download complete",
    error="Download failed",
):
    await download_data()
```

## Building events into new tools

Tools should use `ToolEvents.from_runtime(...)`. This preserves the runnable config and includes the LangGraph `tool_call_id` when available.

```python
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from ursa.agents.base import AgentContext
from ursa.util.events import ToolEvents

@tool
def read_custom_file(path: str, runtime: ToolRuntime[AgentContext]) -> str:
    events = ToolEvents.from_runtime("read_custom_file", runtime)

    with events.range(
        "read",
        "Reading file",
        done="File read",
        error="File read failed",
        path=path,
    ):
        text = runtime.context.workspace.joinpath(path).read_text()

    return text
```

If validation fails before the main work begins, emit an explicit error event:

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

## Building new interfaces and UIs

Interfaces should consume events; they should not require agents/tools to render UI themselves.

Recommended approach for a new UI:

1. Implement a callback handler.
2. Filter on `DEFAULT_EVENT_NAME`.
3. Interpret structured fields such as `agent`, `tool`, `stage`, `phase`, `message`, `path`, `query`, and `elapsed_ms`.
4. Render UI-specific details in the interface layer only.

Example skeleton:

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

Existing examples:

- CLI/HITL uses `HITLLogEventHandler` from `ursa.cli.callbacks`.
- Dashboard adapters attach the same handler when the target agent supports `invoke(..., config=...)`.
- Simple scripts use `configure_event_logging()`.

## Building applications that embed URSA workflows

When an external application runs URSA agents or workflows:

```python
handler = MyProgressHandler()
result = agent.invoke(inputs, config={"callbacks": [handler]})
```

For multi-agent workflows, pass callbacks at the top-level invocation when possible. URSA's base config and LangChain callback propagation allow nested agents/tools to emit into the same stream when config is passed correctly.

Applications can use events to:

- update progress bars,
- stream status messages,
- record audit logs,
- collect telemetry,
- show tool activity,
- display recoverable errors,
- trigger application-specific side effects.

Do not parse console logs if callbacks are available.

## Naming conventions

Use stable snake_case `stage` names. Prefer pairs for request/result style stages:

- `generate` / `generate_result`
- `reflect` / `reflect_result`
- `search` / `search_result`
- `summarize` / `summarize_result`
- `critique` / `critique_result`

Use `events.range(...)` with phases for scoped operations:

- `write` with `phase=start|end|error`
- `edit` with `phase=start|end|error`
- `execute` with `phase=start|end|error`
- `download` with `phase=start|end|error`
- `ingest` with `phase=start|end|error`

Good event messages are short and readable:

```python
events.emit("Searching Web", stage="search", query=query)
events.emit("Web search complete", stage="search_result", result_chars=len(result))
```

Avoid unstable or overly specific stage names:

```python
# Avoid
events.emit("Doing thing", stage="step_7_for_bob_debug_temp")
```

## Payload guidelines

Do include:

- file paths,
- query strings,
- result sizes,
- return codes,
- boolean status fields,
- short previews,
- IDs needed to correlate events,
- elapsed time from event ranges.

Do not include:

- secrets,
- API keys,
- credentials,
- full command outputs,
- full file contents,
- very large LLM responses,
- UI-specific formatting instructions.

Use summaries instead of large payloads:

```python
events.emit(
    "Command finished",
    stage="execute",
    returncode=result.returncode,
    stdout_chars=len(stdout),
    stderr_chars=len(stderr),
)
```

If a preview is useful, truncate it:

```python
events.emit(
    "Draft ready",
    stage="generate_result",
    preview=text[:2000],
)
```

## Migration pattern

Old pattern:

```python
print("PlanningAgent: generating . . .")
plan = structured_llm.invoke(messages)
print("PlanningAgent: Plan approved")
```

New pattern:

```python
events = self.events(config)
events.emit("Drafting plan", stage="generate")
plan = structured_llm.invoke(messages)
events.emit(
    "Drafted plan",
    stage="generate_result",
    steps=[step.model_dump() for step in plan.steps],
)
```

Old tool pattern:

```python
console.print("Writing file:", filename)
path.write_text(code)
console.print("File written:", path)
```

New tool pattern:

```python
events = ToolEvents.from_runtime("write_code", runtime)
with events.range(
    "write",
    "Writing file",
    done="File written",
    error="Failed to write file",
    filename=filename,
    path=str(path),
):
    path.write_text(code)
```

## Testing event behavior

Prefer testing structured payloads over rendered text.

Good tests assert:

- correct event name,
- expected `agent` or `tool`,
- expected `stage`, `phase`, and `message`,
- useful metadata fields,
- error events on failure paths,
- callback forwarding through `config`.

Rendering tests are still useful for UI handlers, but core agent/tool tests should focus on event payloads.

## Development checklist

Before merging new URSA agent/tool/workflow/interface code:

- [ ] Core logic emits events instead of printing progress.
- [ ] Event-emitting graph nodes accept `config: RunnableConfig | None = None`.
- [ ] Tools use `ToolEvents.from_runtime(...)`.
- [ ] Long-running operations use `events.range(...)` where appropriate.
- [ ] Events include stable `stage` values and concise `message` text.
- [ ] Error paths emit `phase="error"` or use range error handling.
- [ ] Payloads are small, structured, and safe.
- [ ] Large outputs are summarized with counts or truncated previews.
- [ ] UI rendering lives in callbacks/interfaces, not agents/tools.
- [ ] Applications pass callbacks through `config={"callbacks": [...]}`.
- [ ] Tests cover important emitted events or callback behavior.

## Rule of thumb

If a developer, user, or UI needs to know that something happened during a run, emit an event. If something needs to decide how that event looks, write a callback or renderer.

Keep URSA execution code observable, structured, and interface-independent.
