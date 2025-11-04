# PlanningAgent Documentation

`PlanningAgent` is a class that implements a multi-step planning approach for complex problem solving. It uses a state machine architecture to generate plans, reflect on them, and formalize the final solution.

## Basic Usage

```python
from ursa.agents import PlanningAgent

# Initialize the agent
agent = PlanningAgent()

# Run a planning task
result = agent.invoke("Find a city with at least 10 vowels in its name.")

# Access the final plan
plan_steps = result["plan_steps"]
```

## Parameters

When initializing `PlanningAgent`, you can customize its behavior with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | BaseChatModel | `init_chat_model("openai:gpt-5-mini")` | The LLM model to use for planning |
| `**kwargs` | `dict` | `{}` | Additional parameters passed to the base agent |

## Features

### Multi-step Planning

The agent follows a three-stage planning process:

1. **Generation**: Creates an initial plan to solve the problem
2. **Reflection**: Critically evaluates and improves the plan
3. **Formalization**: Structures the final plan as a JSON object

### Structured Output

The final output includes:

- `messages`: The conversation history
- `plan_steps`: A structured list of steps to solve the problem

## Advanced Usage

### Customizing Reflection Steps

You can adjust how many reflection iterations the agent performs:

```python
# Initialize with custom reflection steps
initial_state = {
    "messages": [HumanMessage(content="Your complex problem here")],
    "reflection_steps": 5  # Default is 3
}

result = agent.invoke(initial_state, {"configurable": {"thread_id": agent.thread_id}})
```

### Streaming Results

You can stream the agent's thinking process:

```python
for event in agent.stream(
    {"messages": [HumanMessage(content="Your problem here")]},
    {"configurable": {"thread_id": agent.thread_id}}
):
    print(event[list(event.keys())[0]]["messages"][-1].content)
```

### Setting a Recursion Limit

For complex planning tasks, you may need to adjust the recursion limit:

```python
result = agent.invoke(
    "Solve this complex problem...", 
    recursion_limit=200  # Default is 100
)
```

## How It Works

1. **State Machine**: The agent uses a directed graph to manage its workflow:
   - `generate` node: Creates or improves the plan
   - `reflect` node: Evaluates the plan for improvements
   - `formalize` node: Structures the final plan as JSON

2. **Termination Conditions**: The planning process ends when either:
   - The agent has completed the specified number of reflection steps
   - The agent explicitly marks the plan as "[APPROVED]"

3. **JSON Output**: The final plan is structured as a JSON array of steps, each containing:
   - A description of the step
   - Any relevant details for executing that step

## Notes

- The agent continues to refine its plan through multiple reflection cycles
- The final output is a structured JSON representation of the solution steps
- You can access the complete conversation history in the `messages` field of the result