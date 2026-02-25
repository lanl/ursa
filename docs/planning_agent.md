# PlanningAgent Documentation

`PlanningAgent` generates structured multi-step plans for a task and optionally performs reflection loops.

## Basic Usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import PlanningAgent

llm = init_chat_model("openai:gpt-5.2")
agent = PlanningAgent(llm=llm)

result = agent.invoke("Find a city with at least 10 vowels in its name.")
plan = result["plan"]
print(plan.steps[0].name)
```

## Output Schema

- `plan`: structured `Plan` object with `steps`
- `messages`: message history for generation/reflection

## Advanced Usage

### Customizing Reflection Steps

```python
agent = PlanningAgent(llm=llm, max_reflection_steps=3)
```

`max_reflection_steps` defaults to `1`.

## Graph Shape

- `generate`: creates a structured plan (`Plan`)
- `reflect`: critiques and requests regeneration if needed

Routing ends when reflection budget is exhausted or reflection marks the plan with `[APPROVED]`.
