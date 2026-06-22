# Persistence

URSA can persist named agents, groups, checkpoints, and RAG collections so work can continue across sessions.

The key concept is that URSA separates **persistent identity** from **behavior**:

```text
Group
└── Named agent
    ├── durable state and checkpoints
    ├── associated workspace context
    └── behavior selected at runtime: chat, execution, planning, prompting, dashboard, etc.
```

A named agent can be operated through different behaviors depending on the interface and workflow.

## Start with a named agent

```bash
ursa --config config.yaml --name my-agent --group default
```

The named agent can store state so you can return to it later.

## Groups

Groups organize persistent agents and define endpoint-security policy through allowed base URLs. Groups are useful for separating projects, teams, endpoint policies, or levels of data sensitivity.

## RAG collections

Persistent RAG collections let you ingest documents once and query them later directly or as tools attached to another URSA agent.

## Persistence topics

- [Named agents](named-agents.md)
- [Groups and endpoint security](groups-and-security.md)
- [RAG collections](rag.md)
- [Checkpoints, sharing, import, and export](checkpoints-and-sharing.md)
- [Quick reference](quick-reference.md)
