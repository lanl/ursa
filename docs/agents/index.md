# Agents

URSA agents are reusable behaviors that can chat, plan, execute, search, maintain persistent research artifacts, reason over documents, refine prompts, or interact with external tools.

The interactive command-line interface exposes common agents by short names, including:

- `chat`
- `plan`
- `execute`
- `prompt`
- `arxiv`
- `web`
- `deep_review`
- `hypothesize`

Additional agents may be available depending on optional dependencies and configuration, such as DSI, LAMMPS, and recall/RAG-related workflows.

## Common agents

- **Chat Agent**: general conversational agent with local workspace/file/experience tools and optional web/literature tools.
- **Planning Agent**: decomposes requests into a structured plan and reflects on it before returning the final plan.
- **Execution Agent**: carries out workspace-changing tasks with code, file, command, experience, optional search/RAG/MCP tools, and a structured review-until-complete loop before recap.
- **Prompting Agent**: helps refine prompts and task descriptions for downstream agents.
- **Acquisition Agents**: shared acquire-then-summarize/RAG agents for external sources:
  - **ArXiv Agent**: searches arXiv, downloads PDFs, extracts text, and summarizes or indexes papers.
  - **OSTI Agent**: searches OSTI records, resolves available full text or landing-page content, and summarizes or indexes records.
  - **Web Search Agent**: searches the open web, retrieves HTML/PDF content, and summarizes or indexes acquired pages.
- **Deep Review Agent**: runs an iterative adversarial review with solution, critique, and competitor/stakeholder phases, then synthesizes a final answer/report.
- **Hypothesizer Agent**: maintains a persistent Markdown hypothesis-space artifact that other agents can read from the experiences store.

## Web and external information

Web/search tools are opt-in for information-control reasons. For CLI sessions, enable them with:

```bash
ursa --use-web
```

With `--use-web`, tool-capable agents that support web access, such as `ChatAgent`, `ExecutionAgent`, `DeepReviewAgent`, and `PromptingAgent`, receive web/search tools according to their implementation. Without it, they should rely on local workspace, experience, RAG, MCP, or other explicitly configured tools.

## Composing agents

For work that benefits from multiple roles, URSA environments compose agents behind one `invoke(...)` interface. Use an [Agent Team](../environments/agent-teams.md) when one PI should delegate to specialists and synthesize a coherent answer. Use an [Agent Symposium](../environments/agent-symposia.md) when several agents or nested teams should work independently, review one another, revise, and then produce a final synthesis.

## Safety note

Agents with tools can access local files, run commands, or make network requests depending on configuration. Use dedicated workspaces for execution-heavy workflows and review generated commands/code when appropriate. See [Sandboxing and information control](../best-practices/sandboxing.md).

## Agent documentation

Use the individual agent pages for current class-level details and examples.
