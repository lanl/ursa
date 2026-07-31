# Web Dashboard Workflow

The dashboard provides a session-based UI over URSA agents while reusing the same persistent named-agent store as the CLI.

## Launch

Default group:

```bash
ursa-dashboard
```

Specific group:

```bash
ursa-dashboard --group chemistry
```

The dashboard group is selected at launch. Agent lists, agent management, and RAG tool selection are scoped to that group.

## Named agents and sessions

The dashboard distinguishes between:

- a persistent named agent under `~/.cache/ursa_agents/<group>/`
- a dashboard session containing chat messages and run history

Selecting a named agent starts a new session for that agent. Old sessions remain available separately.

Unnamed sessions are also supported for compatibility and ad hoc work.

## Choosing behaviors

The dashboard composer can choose the agent behavior for a message.

Common behaviors include:

- Chat
- Execution
- Prompting
- Planning + Execution

This lets one persistent named agent be operated through different behaviors across a workflow.

## Settings: Agent management

The Settings menu includes Agent management for named agents in the active group.

Typical operations:

- save/checkpoint an agent
- copy an agent
- delete an agent

These operate on the same persistent store as CLI commands such as `save-agent`, `copy-agent`, and `delete-agent`.

## Settings: Agent tools

The Settings menu includes Agent tools for persisted RAG tools.

Use it to:

- view persisted RAG agents in the active dashboard group
- add selected RAG agents as tools
- remove selected RAG tools
- refresh the list
- save the configuration

Selected RAG tools are applied to new Chat, Execution, and Planning + Execution runs.

## Web access default

The dashboard can pass the dashboard web-access default into agents that opt into web tools.

Tool availability still depends on the selected behavior and configuration.

## API credentials

Dashboard model and embedding keys can be saved through Settings using the
operating system credential store. Dashboard persistence contains only an
opaque credential reference and endpoint binding; raw keys are not included in
session or run history. Environment-variable credentials remain available for
headless deployments and automation.

## Practical workflow

1. Create a group if needed:

   ```bash
   ursa create-group chemistry chemistry_group.yaml
   ```

2. Create or select a named agent:

   ```bash
   ursa --name catalyst-helper --group chemistry
   ```

3. Optionally create RAG collections:

   ```bash
   ursa rag-ingest ./papers --name papers --group chemistry
   ```

4. Launch the dashboard:

   ```bash
   ursa-dashboard --group chemistry
   ```

5. Open Settings -> Agent tools and attach RAG collections.
6. Start a session with the named agent.
7. Switch behaviors as needed in the composer.
