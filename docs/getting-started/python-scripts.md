# Getting Started - Python Scripts

URSA agents can be used directly from Python. This is useful when you want to build repeatable workflows, integrate URSA with existing scripts, or compose agents programmatically.

## Set up a Python project

Install URSA in the project environment that will run your script. A separate
`uv tool install ursa` installation provides the `ursa` command, but its
isolated environment is not importable by project scripts.

=== "uv project (recommended)"

    ```bash
    uv init ursa-script
    cd ursa-script
    uv add ursa-ai
    ```

=== "venv + pip — macOS/Linux"

    ```bash
    mkdir ursa-script
    cd ursa-script
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install ursa-ai
    ```

=== "venv + pip — Windows PowerShell"

    ```powershell
    New-Item -ItemType Directory ursa-script
    Set-Location ursa-script
    py -3 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    python -m pip install ursa-ai
    ```

Before continuing, [configure an LLM endpoint][configuration] and choose a
dedicated workspace for execution tasks.

## Minimal execution-agent script

Create `run_ursa.py`:

```python
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent
from ursa.cli.config import UrsaConfig, resolve_ursa_config

config = resolve_ursa_config(UrsaConfig())
llm = config.llm_model.init_chat_model()
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
uv run run_ursa.py
```

!!! warning "Execution safety"
    `ExecutionAgent` can create files and run shell commands. Use a dedicated workspace and review generated code and commands.

## Initialize chat and embedding models from a URSA config

Use the same YAML model configuration in scripts that you use with the TUI and
dashboard. For example, create `config.yaml`:

```yaml
emb_model:
  model: openai:text-embedding-3-large
```

The built-in `openai` inference provider supplies the endpoint and reads
`OPENAI_API_KEY`. Load, resolve, and instantiate both models:

```python
from pathlib import Path

from ursa.cli.config import UrsaConfig, resolve_ursa_config

config = resolve_ursa_config(UrsaConfig.from_file(Path("config.yaml")))

chat_model = config.llm_model.init_chat_model()
embedding_model = (
    config.emb_model.init_embedding()
    if config.emb_model is not None
    else None
)
```

Resolution applies the selected `inference_providers` settings and resolves API
key references in memory. It does not write the secret back to the YAML file.
`UrsaConfig.from_file()` reads the specified file; use the CLI when you need its
full system, user, environment, explicit-file, and command-line precedence.

The resulting objects are ordinary LangChain chat and embedding models and can
be passed to URSA agents, environments, or other LangChain components.

## Connect an MCP server and add its tools to an agent

Follow the [standalone MCP tools example](../examples/mcp_agent_tools/index.md)
to start a local server, configure it, discover its tools, attach them to a
`ChatAgent`, and invoke the agent from Python.

## Use another provider

Keep endpoint and credential settings in the URSA configuration rather than
duplicating them in Python. See [Models and inference
providers][models-and-inference-providers] for hosted, OpenAI-compatible, and
local examples. The resolved model object above uses those same settings.

## Compose agents with environments

When one agent is not the right shape for the work, URSA environments let you
run multiple agents behind one Python object. An [Agent Team][agent-teams] gives
a PI delegation tools for specialist members. An
[Agent Symposium][agent-symposia] asks multiple members or nested teams to work
independently, review one another, revise, and then synthesize a final answer.

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

See [Environments][environments-agents-working-together] for narrative guides
and YAML examples.

## Checkpointing and longer examples

Many of the examples in the repository show checkpointing and multi-step workflows. See:

- `examples/single_agent_examples/`
- `examples/two_agent_examples/`
- `examples/environments/`
- [Plan-Execute From YAML][getting-started-plan-execute-from-yaml]
- [Plan-Execute checkpointing reference][planexecute-runner-checkpointing-resume-guide]

## Where next?

- [Agents overview][agents]
- [Configuration](../configuration/index.md)
- [Sandboxing and information control][sandboxing-and-information-control]
