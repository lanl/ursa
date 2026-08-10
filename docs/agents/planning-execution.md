# Planning and Execution Agent

`PlanningExecutionAgent` combines planning and tool-assisted execution in one
persistent `BaseAgent` graph.

```text
PlanningExecutionAgent
├── planner subgraph
└── executor subgraph
```

The parent owns the model, graph identity, thread, workspace, checkpointer,
store, runtime context, callbacks, and telemetry. The native subgraphs inherit
those resources and use separate checkpoint namespaces (`planner` and
`executor`).

## Basic use

```python
from langchain.chat_models import init_chat_model

from ursa.agents import PlanningExecutionAgent

model = init_chat_model("openai:o4-mini")
agent = PlanningExecutionAgent(
    llm=model,
    workspace="plan_execute_workspace",
    enable_metrics=True,
)

result = agent.invoke(
    "Compare two algorithms for computing Fibonacci numbers and benchmark them."
)
print(result["messages"][-1].text)
agent.close()
```

Pass the model only once. Do not construct and inject separate `PlanningAgent`
and `ExecutionAgent` instances.

## Persistence

Configure persistence on the parent exactly as for any other `BaseAgent`:

```python
from pathlib import Path

from ursa.agents import PlanningExecutionAgent
from ursa.util import Checkpointer

workspace = Path("plan_execute_workspace")
checkpointer = Checkpointer.from_workspace(workspace)
agent = PlanningExecutionAgent(
    llm=model,
    workspace=workspace,
    checkpointer=checkpointer,
    thread_id="analysis-1",
)
```

A checkpoint database for a run contains the parent namespace plus isolated
`planner` and `executor` namespaces. The nested graphs do not open independent
checkpoint resources.

## State boundaries

The parent replaces planner messages before each execution step. Earlier work
is passed to the next step as explicit summaries rather than by leaking the raw
planner or executor transcript. Starting a new request on the same thread also
resets invocation-specific task, plan, and step state.

## Interrupt and resume

The parent accepts LangGraph `Command` inputs, so an interrupted tool can resume
through the same agent and thread:

```python
from langgraph.types import Command

config = {"configurable": {"thread_id": "approval-run"}}
paused = agent.invoke("Perform an operation that requires approval", config=config)
completed = agent.invoke(Command(resume="approved"), config=config)
```

Place an interrupt before a non-idempotent side effect. Code executed before an
interrupt may run again when the node resumes.

## Think, plan, execute, and update

Use `ThinkPlanningExecutionAgent` to maintain a durable hypothesis space around
planning and execution. It embeds URSA's hypothesizer behavior as a native
subgraph and uses the same parent model, workspace, callbacks, thread, and
persistence resources as the planner and executor:

```python
from ursa.workflows import ThinkPlanningExecutionAgent

agent = ThinkPlanningExecutionAgent(llm=model, workspace="experiment")
state = agent.invoke("Form competing hypotheses, plan a test, and execute it.")
print(state["hypothesis_space_markdown"])
```

The graph updates `experiences/hypothesis_space.md` from the user request,
passes the full resulting hypothesis space to the planner, executes the plan
step by step, and sends the execution summaries back through the hypothesizer
as new evidence. The final state and `format_result()` expose the revised
hypothesis space.

On a persistent thread, a follow-up request is incorporated into the existing
hypothesis space before a fresh plan is produced. Invocation-specific plan,
step, and executor state is reset, while the original question, durable
hypothesis artifact, and revision history are retained. The follow-up execution
results then produce another update to the same artifact.
