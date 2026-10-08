# Getting LLM History

This document describes how to retrieve all content sent to and returned from an
LLM when it is used by an agent.

## Retrieving history for an ExecutionAgent

```python
from pathlib import Path

from ursa.agents import ExecutionAgent
from ursa.util.traced import TracedChatOpenAI

# OpenAI model with reasoning abilities
llm = TracedChatOpenAI(
    model="gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
)

# Ollama models can be traced in the same way with TracedChatOllama.
executor = ExecutionAgent(llm=llm)
executor.invoke("Write a Python script to print the first 10 positive integers.")

# Omit indent for minified JSON.
llm.save_messages(Path("messages.json"), indent=2)
```

## Retrieving history for planning and execution

`PlanningExecutionAgent` owns one graph containing adapted planner and executor
child-agent nodes. Pass the traced model once; every planning, execution,
review, and recap call uses that model and appears in the same history. Add a
checkpointer or persistent `agent_name` when graph state must survive across
processes or sessions.

```python
import tempfile
from pathlib import Path

from ursa.workflows import PlanningExecutionAgent
from ursa.util.traced import TracedChatOpenAI

llm = TracedChatOpenAI(
    model="gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
)

workspace = Path(tempfile.mkdtemp())
agent = PlanningExecutionAgent(llm=llm, workspace=workspace)
agent.invoke(
    "Write a Python script of fewer than 10 lines to compute pi "
    "using Monte Carlo and the standard library only. Plan at most two steps."
)
llm.save_messages(Path("messages.json"), indent=2)
agent.close()
```

The parent agent owns the workspace, thread, callbacks, telemetry, checkpointer,
and store. Its planner and executor nodes inherit those resources; do not
construct or inject separate child agents.
