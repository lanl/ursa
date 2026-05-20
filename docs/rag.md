# Persistent RAG Agents

URSA supports persistent Retrieval-Augmented Generation (RAG) collections. A persistent RAG collection lets you ingest documents once, store the resulting vectorstore on disk, and query that collection later from the CLI or through another URSA agent as a tool.

Persistent RAG collections are separate from regular persisted URSA agents. They are stored under:

```text
~/.cache/ursa_rag/<group>/<rag_agent_name>/
```

For example, a RAG collection named `papers` in the default group is stored under:

```text
~/.cache/ursa_rag/default/papers/
```

## What gets stored

A persistent RAG collection stores URSA's RAG artifacts, such as the vectorstore and summaries, in its cache directory.

Raw source documents are **not copied** into the RAG cache during ingestion. Instead, `rag-ingest` reads from the file or directory you provide and stores the indexed information in the collection's vectorstore.

This avoids duplicating large document trees and avoids copying unrelated large files into URSA's cache.

## Ingest documents

Use `rag-ingest` to create or update a named RAG collection:

```bash
ursa rag-ingest <file-or-directory> --name <rag_agent_name>
```

Example:

```bash
ursa rag-ingest ./papers --name papers
```

This creates or updates:

```text
~/.cache/ursa_rag/default/papers/
```

and indexes ingestible files from `./papers`.

You can also ingest a single file:

```bash
ursa rag-ingest ./notes/report.pdf --name reports
```

### Ingestion options

`rag-ingest` supports:

```bash
ursa rag-ingest <source> \
  --name <rag_agent_name> \
  --group default \
  --return-k 10 \
  --chunk-size 1000 \
  --chunk-overlap 200
```

Options:

- `--name`: required name for the persistent RAG collection.
- `--group`: RAG group name. Defaults to `default`.
- `--return-k`: number of chunks to retrieve when the ingest command invokes the RAG graph. Defaults to `10`.
- `--chunk-size`: text chunk size used during ingestion. Defaults to `1000`.
- `--chunk-overlap`: overlap between text chunks. Defaults to `200`.

During ingestion, URSA prints the source path and confirms that raw documents were not copied:

```text
RAG agent: papers
Group: default
Path: /home/user/.cache/ursa_rag/default/papers
Source: /home/user/project/papers
```

## Query a RAG collection

Use `rag-query` to query a named persistent RAG collection:

```bash
ursa rag-query --name <rag_agent_name> <query>
```

Example:

```bash
ursa rag-query --name papers What are the main conclusions about catalyst stability?
```

You can specify a group:

```bash
ursa rag-query --name papers --group chemistry What are the main conclusions?
```

You can also omit the query to enter a simple RAG query loop:

```bash
ursa rag-query --name papers
```

Then type questions at the prompt:

```text
rag> What datasets are mentioned?
rag> Which papers discuss uncertainty?
```

Press `Ctrl-D` or enter a blank line to exit.

## Manage RAG collections

List RAG collections in a group:

```bash
ursa list-rag-agents --group default
```

Show details for a RAG collection:

```bash
ursa show-rag-agent reports --group default
```

Save a timestamped checkpoint copy of a RAG collection:

```bash
ursa save-rag-agent reports --group default
```

Delete a RAG collection:

```bash
ursa delete-rag-agent reports --group default
```

## Use RAG collections as tools

Persisted RAG collections can be bound as tools to tool-capable URSA agents.

From the CLI, pass one or more RAG collection names with `--rag-tools`:

```bash
ursa --name assistant --rag-tools reports
```

Multiple collections can be comma-separated:

```bash
ursa --name assistant --rag-tools reports,lab-notes
```

When the agent calls a RAG collection as a tool, URSA prints the request so it is clear that the RAG tool was used:

```text
[Request to papers]: What did the documents say about catalyst stability?
```

The RAG tool then returns the RAG summary to the calling agent.

### Python usage

When constructing an agent in Python, pass `rag_tools` as a string or list:

```python
from ursa.agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    rag_tools=["papers", "lab-notes"],
)
```

or:

```python
agent = ChatAgent(
    llm=llm,
    rag_tools="papers,lab-notes",
)
```

For agents that support tools, URSA builds one RAG query tool per named collection.

## Groups and model whitelist policy

RAG groups are aligned with regular URSA agent groups.

Regular agent groups are stored under:

```text
~/.cache/ursa_agents/<group>/
```

RAG groups are stored under:

```text
~/.cache/ursa_rag/<group>/
```

For the `default` group, URSA creates the RAG group as needed.

For a non-default group, the corresponding regular URSA agent group must already exist. For example, before creating a RAG collection in group `chemistry`, this directory must exist:

```text
~/.cache/ursa_agents/chemistry/
```

and it must contain:

```text
group.yaml
```

When a RAG group is first created, URSA copies:

```text
~/.cache/ursa_agents/<group>/group.yaml
```

to:

```text
~/.cache/ursa_rag/<group>/group.yaml
```

This keeps the RAG group associated with the same whitelist configuration as the corresponding regular agent group.

If the regular agent group does not exist, URSA raises an error and asks you to create the group first:

```bash
ursa create-group <group> <group_config_file>
```

See the CLI guide for more information about creating and managing groups.

## Typical workflow

1. Create a group if needed:

   ```bash
   ursa create-group chemistry allowed_urls.yaml
   ```

2. Ingest a document directory into a named RAG collection:

   ```bash
   ursa rag-ingest ./papers --name catalyst-papers --group chemistry
   ```

3. Query the RAG collection directly:

   ```bash
   ursa rag-query --name catalyst-papers --group chemistry What are the major findings?
   ```

4. Bind the RAG collection as a tool to an URSA agent:

   ```bash
   ursa --name catalyst-assistant --group chemistry --rag-tools catalyst-papers
   ```

5. Ask questions in the URSA CLI. If the agent calls the RAG tool, you will see output like:

   ```text
   [Request to catalyst-papers]: Summarize the evidence for improved stability.
   ```

## Notes and limitations

- `rag-ingest` does not copy raw documents into the RAG cache. If the original documents are moved or deleted, already-ingested vectorstore content remains in the persistent RAG collection, but future ingestion from that original source path will require the files to still exist.
- Re-running `rag-ingest` on the same source updates the persistent RAG collection by indexing documents not already present in the vectorstore.
- RAG collection names use the same naming policy as persisted URSA agents.
- RAG tools are available to URSA agents that support tools.
- The RAG CLI uses URSA's configured language model and embedding model settings.
