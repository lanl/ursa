# Agents

URSA agents are reusable behaviors that can plan, execute, search, reason over documents, refine prompts, or interact with external tools.

The command-line interface exposes common agents by short names, including:

```text
chat
plan
execute
prompt
arxiv
web
hypothesize
```

Additional agents may be available depending on optional dependencies and configuration, such as DSI, LAMMPS, and recall/RAG-related workflows.

## Common agents

- **Chat Agent**: general tool-capable chat behavior.
- **Planning Agent**: decomposes requests into a structured plan.
- **Execution Agent**: writes files, edits code, and runs commands in a workspace.
- **Prompting Agent**: helps refine prompts and task descriptions.
- **ArXiv Agent**: searches and reasons over arXiv literature.
- **Web Search Agent**: performs opt-in web search.
- **Hypothesizer Agent**: generates and evaluates scientific hypotheses.

## Safety note

Agents with tools can access local files, run commands, or make network requests depending on configuration. Web tools are opt in for information-control reasons. See [Sandboxing and information control](../best-practices/sandboxing.md).

## Agent documentation

Use the individual agent pages for current class-level details and examples.
