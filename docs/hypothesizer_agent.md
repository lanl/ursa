# HypothesizerAgent Documentation

`HypothesizerAgent` is a multi-agent system that iteratively refines solutions to complex problems through a structured debate process. It employs three specialized agents that work together to generate, critique, and provide alternative perspectives on potential solutions.

## Basic Usage

```python
from ursa.agents import HypothesizerAgent

# Initialize the agent
agent = HypothesizerAgent()

# Run the agent with a question
solution = agent.invoke(
    prompt="Find a city with at least 10 vowels in its name.",
    max_iter=3
)

# Print the final solution
print(solution)
```

## Parameters

### Initialization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | `init_chat_model("openai:gpt-5-mini")` | The language model to use for all agents |
| `**kwargs` | `dict` | `{}` | Additional parameters passed to the base agent |

### Run Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | The question or problem to solve |
| `max_iter` | int | 3 | Maximum number of refinement iterations |
| `recursion_limit` | int | 99999 | Maximum recursion depth for the graph |

## How It Works

The system uses a three-agent debate process:

1. **Agent 1 (Hypothesizer)**: Generates initial solutions and refines them based on feedback
2. **Agent 2 (Critic)**: Identifies flaws, assumptions, and areas for improvement
3. **Agent 3 (Competitor/Stakeholder)**: Provides alternative perspectives from different stakeholders

The process iterates through these agents multiple times, with each iteration building on the feedback from previous rounds.

## Features

- **Web Search Integration**: Uses DuckDuckGo to gather information for each agent
- **Iterative Refinement**: Solutions improve through multiple rounds of critique and revision
- **LaTeX Report Generation**: Creates a comprehensive LaTeX document summarizing the process
- **URL Tracking**: Records all websites visited during the research process

## Output

The agent produces:

1. A final refined solution (returned by the `run` method)
2. A LaTeX document with:
   - Executive summary of the iterative process
   - Final solution in full
   - Detailed appendix of all iterations
   - List of websites visited during research
3. A text file containing the full history of all iterations

## Example

```python
from ursa.agents import HypothesizerAgent
from langchain.chat_models import init_chat_model

# Initialize with a specific LLM
agent = HypothesizerAgent(llm=init_chat_model("openai:gpt-5-mini"))

# Run with 5 iterations
result = agent.invoke(
    prompt="What strategies could a small local bookstore use to compete with large online retailers?",
    max_iter=5
)

print(result)
```

## Notes

- Each iteration includes a solution, critique, and competitor perspective
- The agent performs web searches to gather information at each step
- The final LaTeX document provides a comprehensive record of the reasoning process
- Higher `max_iter` values produce more refined solutions but take longer to complete