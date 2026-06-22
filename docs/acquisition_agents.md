# Acquisition Agents

URSA's acquisition agents are related agents that acquire external documents or pages, materialize them into local cached artifacts, and then either summarize them directly or run a RAG-style summarization path.

The current concrete acquisition agents are:

- [`ArxivAgent`](arxiv_agent.md) — searches arXiv, downloads paper PDFs, extracts text, and summarizes or indexes them.
- [`OSTIAgent`](osti_agent.md) — searches OSTI records, resolves available full text or landing-page content, and summarizes or indexes it.
- [`WebSearchAgent`](web_search_agent.md) — searches the open web with DDGS, retrieves HTML/PDF content, and summarizes or indexes it.

All three are built on the same `BaseAcquisitionAgent` workflow and are integrated as optional tools for other agents. 

In general these agents are best used as bound tools to other agents, rather than direct querying, however they can be used and integrated into workflows by users.

## Shared workflow

`BaseAcquisitionAgent` implements an acquire-then-summarize/RAG graph:

1. `_search_query` — if `state["query"]` is not provided, asks the LLM to derive a short search query from `state["context"]`.
2. `_fetch_node` — calls the concrete agent's `_search()` method and materializes hits with `_materialize()`.
3. If `summarize=True` and no `rag_embedding` is configured:
   - `_summarize_node` summarizes each acquired item in context.
   - `_aggregate_node` combines item summaries into `state["final_summary"]`.
4. If `summarize=True` and `rag_embedding` is configured:
   - `_rag_node` builds/uses a RAG workflow over the acquired database path and writes `state["final_summary"]`.
5. If `summarize=False`, the graph finishes after acquisition and leaves retrieved items in `state["items"]`.

## Shared state

Acquisition agents use `AcquisitionState`, with these fields:

- `query` — search query. Optional if `context` is provided.
- `context` — the user task or question used for query generation and summarization.
- `items` — acquired `ItemMetadata` records.
- `summaries` — per-item summaries when direct summarization is used.
- `final_summary` — aggregate summary or RAG answer.

Each acquired `ItemMetadata` may contain:

- `id`
- `title`
- `url`
- `local_path`
- `full_text`
- `extra`

## Shared parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for query generation and summarization. |
| `summarize` | `bool` | `True` | Whether to summarize/RAG over acquired items. If `False`, acquisition stops after `items` are populated. |
| `rag_embedding` | optional embedding object | `None` | If provided, the agent uses the RAG summarization path instead of per-item direct summarization. |
| `process_images` | `bool` | `True` | For PDF-backed items, optionally append image interpretations when image-description support is available. |
| `max_results` | `int` | base default `5` | Maximum search hits to acquire. `ArxivAgent` overrides this default to `3`. |
| `database_path` | `str` | agent-specific | Directory under the agent den where acquired PDFs/HTML/text are cached. |
| `summaries_path` | `str` | agent-specific | Directory under the agent den where per-item and final summaries are written. |
| `vectorstore_path` | `str` | agent-specific | Stored vector-store path configuration. The current shared RAG node constructs a `RAGAgent` for the acquired database when `rag_embedding` is provided. |
| `num_threads` | `int` | `4` | Maximum number of concurrent materialization/summarization workers. |
| `download` | `bool` | `True` | If `True`, search and acquire new items. If `False`, read cached `.pdf`, `.txt`, or `.html` files from `database_path`. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent`, including workspace/den and persistence options. |

## Input formats

You can pass a string. The string becomes `context`, and the LLM derives a short search query:

```python
state = agent.invoke("Find recent literature relevant to alloy phase stability.")
```

For more control, pass both `query` and `context`:

```python
state = agent.invoke({
    "query": "alloy phase stability machine learning potentials",
    "context": "Summarize the evidence most relevant to validating a new alloy potential.",
})
```

`agent.format_result(state)` returns `state["final_summary"]` when a final summary is available.

## Cached outputs

Acquired documents are stored under the configured `database_path` inside the agent den. Direct summarization writes:

- one summary file per acquired item
- `summaries_combined.txt`
- `final_summary.txt`

Exact filenames and citations are determined by each concrete acquisition agent's `_id()` and `_citation()` methods.

## CLI availability

The interactive CLI currently registers `arxiv` and `web` acquisition agents. `OSTIAgent` is exported from `ursa.agents` for Python/API use; it is not currently registered as a CLI short name in the inspected source.
