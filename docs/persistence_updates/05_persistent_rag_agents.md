# Persistent RAG Agents and RAG Tools

Persistent RAG agents let URSA ingest documents once and reuse the resulting retrieval collection later.

## Storage layout

RAG agents are stored separately from regular agents:

```text
~/.cache/ursa_rag/<group>/<rag_agent_name>/
```

A collection may contain artifacts such as:

```text
database/
summaries/
vectorstore/
```

Raw source documents are not copied into the RAG cache during ingestion. URSA indexes from the file or directory you provide.

## Ingest documents

Ingest a directory:

```bash
ursa rag-ingest ./papers --name papers
```

Ingest a single file:

```bash
ursa rag-ingest ./notes/report.pdf --name reports
```

Use a group:

```bash
ursa rag-ingest ./papers --name papers --group chemistry
```

Common options:

```bash
ursa rag-ingest <source> \
  --name <rag_agent_name> \
  --group default \
  --return-k 10 \
  --chunk-size 1000 \
  --chunk-overlap 200
```

## Query a RAG collection

One-shot query:

```bash
ursa rag-query --name papers What are the main conclusions?
```

With a group and config file:

```bash
ursa rag-query --name papers --group chemistry --config chemistry_config.yaml What is indexed?
```

Interactive query loop:

```bash
ursa rag-query --name papers
```

Then type questions at:

```text
rag>
```

Use a blank line or Ctrl-D to exit.

## Manage RAG agents

```bash
ursa list-rag-agents --group default
ursa show-rag-agent papers --group default
ursa save-rag-agent papers --group default
ursa delete-rag-agent papers --group default
```

## Use RAG as tools

Persisted RAG agents can be attached as tools to tool-capable URSA agents.

CLI:

```bash
ursa --name lab-assistant --rag-tools papers,lab-notes
```

Python:

```python
from ursa.agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    agent_name="lab-assistant",
    rag_tools=["papers", "lab-notes"],
)
```

The CLI flag is:

```text
--rag-tools
```

The Python argument is:

```python
rag_tools=...
```

When a RAG tool is called, URSA prints the request:

```text
[Request to papers]: What did the documents say about catalyst stability?
```

## Dashboard RAG tool selection

The dashboard can attach persisted RAG agents as tools to new runs.

Workflow:

1. Create or update RAG collections with the CLI.
2. Launch the dashboard for the desired group:

   ```bash
   ursa-dashboard --group chemistry
   ```

3. Open Settings.
4. Open Agent tools.
5. Add or remove persisted RAG agents.
6. Click Save.

Selected RAG tools are attached to new Chat, Execution, and Planning + Execution runs.

The dashboard lists only persisted RAG agents in the active dashboard group.

## RAG groups

RAG groups mirror regular agent groups.

Regular agents:

```text
~/.cache/ursa_agents/<group>/
```

RAG agents:

```text
~/.cache/ursa_rag/<group>/
```

For non-default groups, create the regular group first so URSA has a `group.yaml` endpoint whitelist to copy/use.
