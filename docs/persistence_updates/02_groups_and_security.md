# Groups and Endpoint Security

Groups organize persistent agents and can restrict which model endpoints they are allowed to use.

## Storage layout

Groups are stored under:

```text
~/.cache/ursa_agents/<group>/
```

Each non-default group has a config file:

```text
~/.cache/ursa_agents/<group>/group.yaml
```

The default group is:

```text
default
```

The `default` group is unrestricted by the group endpoint whitelist policy.

## Group config format

A non-default group must define allowed model base URLs:

```yaml
allowed_base_urls:
  - https://models.example.org/v1
  - http://127.0.0.1:8000/v1
```

The whitelist is checked against model base URLs. Matching accepts the same normalized URL or same origin.

## CLI group commands

Create a group:

```bash
ursa create-group chemistry chemistry_group.yaml
```

List groups:

```bash
ursa list-groups
```

Show a group:

```bash
ursa show-group chemistry
```

Update a group config:

```bash
ursa update-group chemistry chemistry_group.yaml
```

Delete a group:

```bash
ursa delete-group chemistry
```

The `default` group cannot be deleted.

## Using a group

CLI:

```bash
ursa --name catalyst-helper --group chemistry
```

Python:

```python
agent = ChatAgent(
    llm=llm,
    agent_name="catalyst-helper",
    group="chemistry",
)
```

Dashboard:

```bash
ursa-dashboard --group chemistry
```

The dashboard group is selected at launch. Agent discovery and management are scoped to that group.

## Security behavior

For non-default groups, URSA enforces that the model base URL matches the group's `allowed_base_urls` policy.

This applies in main runtime paths including:

- base agent construction
- CLI/HITL model setup
- RAG agent model and embedding setup
- RAG ingest/query commands

If a URL is not allowed, URSA raises an error and includes the allowed base URLs.

## RAG groups

Persistent RAG agents use a parallel storage root:

```text
~/.cache/ursa_rag/<group>/<rag_agent_name>/
```

For non-default groups, the regular agent group must already exist:

```text
~/.cache/ursa_agents/<group>/group.yaml
```

URSA copies that group config into the RAG group when creating the RAG group if needed.

## Best practices

- Use `default` for unrestricted local experimentation.
- Use named groups for projects that must stay on approved model endpoints.
- Keep group names simple and descriptive.
- Update group configs when endpoint policy changes.
- Launch the dashboard with the same group you use from the CLI for a project.
