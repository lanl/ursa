# Checkpoints, sharing, import, and export

URSA includes commands for saving, copying, sharing, importing, and deleting persistent agent state.

## Save a timestamped checkpoint copy

```bash
ursa save-agent --name catalyst-assistant --group chemistry
```

This creates a timestamped copy that can be used as a restore point or branch source.

## Copy an agent

```bash
ursa copy-agent \
  --name catalyst-assistant-experiment \
  --from catalyst-assistant \
  --group chemistry \
  --from-group chemistry
```

Use copying to branch work without overwriting the original agent.

## Share an agent

```bash
ursa share-agent --name catalyst-assistant --group chemistry
```

This creates a shareable archive in the current working directory.

!!! warning "Review before sharing"
    Persistent state may contain conversation history, generated files, endpoint references, or other sensitive information. Review archives before sharing them.

## Import an agent

```bash
ursa import-agent path/to/shared-agent.tar.gz --group chemistry
```

You can also import compatible SQLite checkpoint databases.

## Delete an agent

```bash
ursa delete-agent --name catalyst-assistant --group chemistry
```

## RAG collection checkpoints

Persistent RAG collections have parallel management commands:

```bash
ursa save-rag-agent papers --group chemistry
ursa delete-rag-agent papers --group chemistry
```

See [RAG collections][persistent-rag-collections].
