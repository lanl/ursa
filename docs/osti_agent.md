# OSTIAgent Documentation

`OSTIAgent` is an acquisition agent for OSTI records. It subclasses `BaseAcquisitionAgent`, so it uses the same acquire-then-summarize/RAG graph as `ArxivAgent` and `WebSearchAgent`.

`OSTIAgent` searches an OSTI records API, resolves available full-text or landing-page content, caches acquired PDFs/HTML, optionally augments PDF text with image descriptions, and then summarizes the acquired content or runs the RAG path when an embedding model is configured.

See also: [Acquisition Agents](acquisition_agents.md).

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import OSTIAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = OSTIAgent(llm=llm)

state = agent.invoke({
    "query": "molten salt reactor materials corrosion",
    "context": "Summarize OSTI records relevant to corrosion of materials in molten salt reactors.",
})

print(agent.format_result(state))
```

You can also pass a plain string. In that case, the string becomes `context`, and the agent asks the LLM to generate a short OSTI search query:

```python
state = agent.invoke("Find OSTI records about materials corrosion in molten salt reactors.")
print(agent.format_result(state))
```

## Parameters

`OSTIAgent` uses the shared `BaseAcquisitionAgent` parameters and adds `api_base`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for query generation and summarization. |
| `api_base` | `str` | `"https://www.osti.gov/api/v1/records"` | OSTI records API endpoint used by `_search()`. |
| `summarize` | `bool` | `True` | Whether to summarize/RAG over acquired OSTI items. If `False`, acquisition stops after `items` are populated. |
| `rag_embedding` | optional embedding object | `None` | If provided, use the RAG path instead of direct per-record summarization. |
| `process_images` | `bool` | `True` | For PDF-backed records, optionally append image interpretations when image-description support is available. |
| `max_results` | `int` | `5` | Maximum number of OSTI records to acquire. |
| `database_path` | `str` | `"acq_db"` | Directory under the agent den for cached PDF/HTML files. |
| `summaries_path` | `str` | `"acq_summaries"` | Directory under the agent den for per-item and final summaries. |
| `vectorstore_path` | `str` | `"acq_vectorstores"` | Stored vector-store path configuration inherited from the acquisition base. The current shared RAG node constructs a `RAGAgent` over the acquired database when `rag_embedding` is provided. |
| `num_threads` | `int` | `4` | Maximum number of concurrent materialization/summarization workers. |
| `download` | `bool` | `True` | If `True`, search and retrieve OSTI records. If `False`, read cached `.pdf`, `.txt`, or `.html` files from `database_path`. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `BaseAcquisitionAgent`, including workspace/den and persistence options. |

The implementation also consults the `UNPAYWALL_EMAIL` environment variable when resolving PDFs from OSTI records.

## How it works

`OSTIAgent` implements the acquisition hooks required by `BaseAcquisitionAgent`:

- `_search(query)` — calls `api_base` with query parameter `q` and `size=max_results`, then normalizes either a `records` list or a top-level list response.
- `_id(hit_or_item)` — prefers `osti_id`, then `id`, then a hash of `landing_page` or the full hit.
- `_materialize(hit)` — resolves an available PDF or landing page from the OSTI record:
  - If a PDF is resolved and the response validates as PDF, it is downloaded, cached, and parsed.
  - If no PDF text is available, the agent tries a DOE PAGES or citation/landing page and extracts bounded page text.
  - If neither source is available, the item text records that no PDF or landing-page text was available.
- `_citation(item)` — formats citations as `OSTI <id>: <title>` when a title is available.

After acquisition, the shared graph either:

- summarizes each record and aggregates the summaries into `state["final_summary"]`, or
- invokes the RAG path when `rag_embedding` is configured, or
- stops after `state["items"]` when `summarize=False`.

## Output state

Important fields in the returned state include:

- `query` — final OSTI search query.
- `context` — task/question used for summarization.
- `items` — acquired OSTI metadata, including local cache paths and extracted text.
- `summaries` — per-item summaries when direct summarization is used.
- `final_summary` — aggregate response to the requested context.

Each OSTI item may also include `extra["raw_hit"]` with the source API record.

`agent.format_result(state)` returns `state["final_summary"]` when present.

## Cached files

By default, the agent writes artifacts under the agent den:

- cached PDF/HTML/text: `acq_db/`
- per-item summaries: `acq_summaries/`
- combined direct-summarization file: `acq_summaries/summaries_combined.txt`
- final direct-summarization file: `acq_summaries/final_summary.txt`
- RAG workflow artifacts are managed by the shared RAG path when `rag_embedding` is provided.

## CLI

`OSTIAgent` is exported from `ursa.agents` for Python/API use:

```python
from ursa.agents import OSTIAgent
```

In the inspected source, it is not currently registered as an interactive CLI short name.
