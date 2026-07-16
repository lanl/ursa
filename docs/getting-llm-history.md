# Getting LLM History

This document describes how to all of the content going into/from the LLM even
when used in an agent.

## Retrieving history for ExecutionAgent-only workflow

```python
import tempfile
from pathlib import Path

from ursa.agents import ExecutionAgent
from ursa.util.traced import TracedChatOpenAI

# OpenAI model with reasoning abilities
llm = TracedChatOpenAI(
    model="gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
)

## Ollama model with reasoning abilities
# from ursa.util.traced import TracedChatOpenAI
# llm = "ollama-nemotron-nano": TracedChatOllama(
#   model="nemotron-3-nano:4b", reasoning=True
# )

## Ollama Model without reasoning abilities
# llm TracedChatOllama(model="nemotron-mini:4b")

executor = ExecutionAgent(llm=llm)
executor.invoke(
    "Write a python script to print the first 10 positive integer."
)
# Save messages to json. Omit indent arg for minified json
llm.save_messages(Path("messages.json"), indent=2)
```

## Retrieving history for plan-execute workflow

```python
import tempfile
from pathlib import Path

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.util.traced import TracedChatOpenAI
from ursa.workflows import PlanningExecutorWorkflow

llm = TracedChatOpenAI(
  model="gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
)

workspace = Path(tempfile.mkdtemp())
planner = PlanningAgent(llm=llm, workspace=workspace)
executor = ExecutionAgent(llm=llm, workspace=workspace)
workflow = PlanningExecutorWorkflow(planner=planner, executor=executor)
workflow(
    "Write a python script <10 lines to compute Pi "
    "using Monte Carlo; use standard lib only."
    "Plan at most two steps."
)
llm.save_messages(Path(f"messages.json"), indent=2)
```
