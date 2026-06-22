# Getting Started - Python Scripts

URSA agents can be used directly from Python. This is useful when you want to build repeatable workflows, integrate URSA with existing scripts, or compose agents programmatically.

## Prerequisites

- URSA is installed in your Python environment.
- You have configured access to an LLM endpoint.
- You have a dedicated workspace for any execution tasks.

## Minimal execution-agent script

Create `run_ursa.py`:

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent

llm = init_chat_model(model="openai:gpt-5.4")
agent = ExecutionAgent(llm=llm)

result = agent.invoke({
    "messages": [
        HumanMessage(
            content="Write and run a Python script that prints the first 10 prime numbers."
        )
    ],
    "workspace": "./ursa-script-workspace",
})

print(result["messages"][-1].content)
```

Run it:

```bash
python run_ursa.py
```

!!! warning "Execution safety"
    `ExecutionAgent` can create files and run shell commands. Use a dedicated workspace and review generated code and commands.

## Use a local or custom endpoint

The Python API uses LangChain chat models, so the same provider packages and endpoint settings apply. For example, with Ollama:

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="ollama:llama3.1",
    base_url="http://localhost:11434",
)
```

For a custom OpenAI-compatible endpoint:

```python
import os
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model="openai:my-model-name",
    base_url="https://my-endpoint.example.com/v1",
    api_key=os.environ["MY_ENDPOINT_API_KEY"],
)
```

## Compose agents with environments

When one agent is not the right shape for the work, URSA environments let you run multiple agents behind one Python object. An [Agent Team](../environments/agent-teams.md) gives a PI delegation tools for specialist members. An [Agent Symposium](../environments/agent-symposia.md) asks multiple members or nested teams to work independently, review one another, revise, and then synthesize a final answer.

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")
symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml",
    llm=llm,
)

result = symposium.invoke("Compare two solution strategies and recommend one.")
print(result["final"])
```

See [Environments](../environments/index.md) for narrative guides and YAML examples.

## Checkpointing and longer examples

Many of the examples in the repository show checkpointing and multi-step workflows. See:

- `examples/single_agent_examples/`
- `examples/two_agent_examples/`
- `examples/environments/`
- [Plan-Execute From YAML](plan-execute-yaml.md)
- [Plan-Execute checkpointing reference](../Plan-Execute-Runner-Checkpointing-Guide.md)

## Where next?

- [Agents overview](../agents/index.md)
- [Configuration](../configuration/index.md)
- [Sandboxing and information control](../best-practices/sandboxing.md)
