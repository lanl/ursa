# Persistent Named Agents

Persistent named agents let URSA keep durable agent state on disk and return to it later by name.

## Storage layout

Regular named agents are stored under:

```text
~/.cache/ursa_agents/<group>/<agent_name>/
```

Examples:

```text
~/.cache/ursa_agents/default/lab-assistant/
~/.cache/ursa_agents/chemistry/catalyst-helper/
```

The `default` group is used unless another group is specified.

## Naming rules

Agent names must be simple directory names:

- must not be empty
- must not be `.` or `..`
- must not contain path separators
- may contain letters, numbers, dots, underscores, and hyphens
- must begin with a letter or number

Valid examples:

```text
lab-assistant
project_01
analysis.agent
```

Invalid examples:

```text
../agent
/my/agent
.agent-starts-with-dot
```

## Python scripting

Pass `agent_name` and optionally `group` when constructing an agent:

```python
from ursa.agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    agent_name="lab-assistant",
    group="default",
)
```

The named agent's persistent directory is used as its durable storage location.

You can construct a different agent class with the same `agent_name` later:

```python
from ursa.agents import ExecutionAgent

agent = ExecutionAgent(
    llm=llm,
    agent_name="lab-assistant",
    group="default",
)
```

This lets the same persistent identity be used with different behaviors.

## CLI

Use `--name` to select or create a persistent named agent:

```bash
ursa --name lab-assistant
```

Use `--group` to scope the agent to a group:

```bash
ursa --name catalyst-helper --group chemistry
```

Management commands:

```bash
ursa list-agents --group default
ursa show-agent --name lab-assistant --group default
ursa save-agent --name lab-assistant --group default
ursa copy-agent --name lab-assistant-copy --from lab-assistant --group default
ursa delete-agent --name lab-assistant --group default
```

Sharing commands:

```bash
ursa share-agent --name lab-assistant --group default
ursa share-agent --name lab-assistant --group default --no-checkpoint
ursa import-agent ./ursa_agent_default_lab-assistant_full_YYYYMMDD_HHMMSS.tar.gz --group default
```

`save-agent` creates a timestamped checkpoint copy in the same group.

## Web dashboard

The dashboard uses the same persistent agent store as the CLI:

```text
~/.cache/ursa_agents/<group>/
```

Launch the dashboard for a group:

```bash
ursa-dashboard --group default
```

In the dashboard:

- named agents are listed from the selected group
- selecting a named agent starts a new session for that agent
- sessions are separate from persistent agent names
- the composer can choose the behavior for a message, such as Chat, Execution, or Planning + Execution

The dashboard also has an Agent management settings pane for operations such as save, copy, and delete.

## Persistence plus behaviors

A useful way to think about the new model:

- the **name** is the durable identity and storage location
- the **group** controls storage scope and endpoint policy
- the **agent class or dashboard behavior** controls how URSA acts right now

This makes Chat, Execution, Planning + Execution, Prompting, and related modes more like behaviors that can operate on a persistent named identity.
