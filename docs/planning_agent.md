# PlanningAgent Documentation

`PlanningAgent` generates a structured plan for a user request, optionally reflects on that plan, and regenerates it until the reflection budget is exhausted or the plan is approved.

Use it when you want an ordered, model-generated plan before handing work to another agent such as `ExecutionAgent`.

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import PlanningAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = PlanningAgent(llm=llm)

state = agent.invoke("Design a workflow to compare two simulation outputs.")
print(agent.format_result(state))
```

The parsed plan is available as `state["plan"]`.

```python
plan = state["plan"]
for step in plan.steps:
    print(step.name, step.requires_code)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for plan generation and reflection. |
| `max_reflection_steps` | `int` | `1` | Maximum number of reflection passes before ending unless the plan is approved sooner. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent`, including workspace, persistence, group, thread/checkpoint, and metrics options. |

## Structured output

Planning uses structured output with this schema:

```python
class PlanStep(BaseModel):
    name: str
    description: str
    requires_code: bool
    expected_outputs: list[str]
    success_criteria: list[str]

class Plan(BaseModel):
    steps: list[PlanStep]
```

`Plan.__str__()` formats the plan as Markdown-like text with each step's title, code requirement, description, expected outputs, and success criteria. `agent.format_result(state)` returns that formatted string.

## Graph behavior

The current graph has two nodes:

- `generate` — produces a structured `Plan` using the planner prompt.
- `reflect` — critiques the current plan using the reflection prompt.

The workflow is:

1. `generate` creates or regenerates a `Plan` and stores it in `state["plan"]`.
2. If `reflection_steps > 0`, the graph routes to `reflect`.
3. `reflect` reviews the plan and decrements `reflection_steps`.
4. If the reflection text contains `[APPROVED]`, the graph finishes.
5. Otherwise, the graph routes back to `generate` for another pass.
6. When the reflection budget reaches zero, the graph finishes with the latest plan.

If a provider returns an empty reflection message, the implementation treats it as `[APPROVED]`.

## State model

`PlanningState` includes:

- `plan` — parsed `Plan` object.
- `messages` — LangChain message history used by generation/reflection.
- `reflection_steps` — remaining reflection passes.

## Custom reflection budget

```python
from langchain.chat_models import init_chat_model
from ursa.agents import PlanningAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = PlanningAgent(llm=llm, max_reflection_steps=3)

state = agent.invoke("Plan a validation study for a new materials model.")
print(agent.format_result(state))
```

Higher reflection budgets can improve plan quality but increase latency and token usage.
