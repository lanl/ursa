# WebSearchAgent Documentation

`WebSearchAgent` is an acquisition agent for open-web research. It subclasses `BaseAcquisitionAgent`, so it uses the same acquire-then-summarize/RAG graph as `ArxivAgent` and `OSTIAgent`.

`WebSearchAgent` uses DDGS search, retrieves HTML or PDF content from result URLs, extracts readable text, optionally augments PDF text with image descriptions, and then summarizes the acquired content or runs the RAG path when an embedding model is configured.

See also: [Acquisition Agents](acquisition_agents.md).

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import WebSearchAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = WebSearchAgent(llm=llm)

state = agent.invoke({
    "query": "2025 Detroit Tigers top prospects birth years",
    "context": "Who are the 2025 Detroit Tigers top 10 prospects and what year was each born?",
})

print(agent.format_result(state))
```

You can also pass a plain string. In that case, the string becomes `context`, and the agent asks the LLM to generate a short web search query:

```python
state = agent.invoke("Find current information about recent quantum computing developments.")
print(agent.format_result(state))
```

## Parameters

`WebSearchAgent` uses the shared `BaseAcquisitionAgent` parameters and adds `user_agent`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for query generation and summarization. |
| `user_agent` | `str` | `"Mozilla/5.0"` | User-Agent header used when retrieving web pages. |
| `summarize` | `bool` | `True` | Whether to summarize/RAG over acquired web items. If `False`, acquisition stops after `items` are populated. |
| `rag_embedding` | optional embedding object | `None` | If provided, use the RAG path instead of direct per-result summarization. |
| `process_images` | `bool` | `True` | For PDF results, optionally append image interpretations when image-description support is available. |
| `max_results` | `int` | `5` | Maximum number of web search results to acquire. |
| `database_path` | `str` | `"acq_db"` | Directory under the agent den for cached HTML/PDF files. |
| `summaries_path` | `str` | `"acq_summaries"` | Directory under the agent den for per-item and final summaries. |
| `vectorstore_path` | `str` | `"acq_vectorstores"` | Stored vector-store path configuration inherited from the acquisition base. The current shared RAG node constructs a `RAGAgent` over the acquired database when `rag_embedding` is provided. |
| `num_threads` | `int` | `4` | Maximum number of concurrent materialization/summarization workers. |
| `download` | `bool` | `True` | If `True`, search and retrieve web results. If `False`, read cached `.pdf`, `.txt`, or `.html` files from `database_path`. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `BaseAcquisitionAgent`, including workspace/den and persistence options. |

`WebSearchAgent` requires the DDGS dependency imported as `ddgs.DDGS`. If it is unavailable, initialization raises `ImportError`.

## How it works

`WebSearchAgent` implements the acquisition hooks required by `BaseAcquisitionAgent`:

- `_search(query)` — uses DDGS text search with `max_results` and `backend="auto"`.
- `_id(hit_or_item)` — hashes the result URL to create a stable cache ID.
- `_materialize(hit)` — retrieves each result URL:
  - PDF-looking URLs are downloaded and parsed as PDFs.
  - Other URLs are fetched as HTML, cached, and passed through main-text extraction.
- `_citation(item)` — formats citations as `title (url)` when a title is available.

After acquisition, the shared graph either:

- summarizes each result and aggregates the summaries into `state["final_summary"]`, or
- invokes the RAG path when `rag_embedding` is configured, or
- stops after `state["items"]` when `summarize=False`.

## Output state

Important fields in the returned state include:

- `query` — final web search query.
- `context` — task/question used for summarization.
- `items` — acquired web item metadata, including local cache paths and extracted text.
- `summaries` — per-item summaries when direct summarization is used.
- `final_summary` — aggregate response to the requested context.

Each web item may also include `extra["snippet"]` from the search result body.

`agent.format_result(state)` returns `state["final_summary"]` when present.

## Cached files

By default, the agent writes artifacts under the agent den:

- cached HTML/PDF/text: `acq_db/`
- per-item summaries: `acq_summaries/`
- combined direct-summarization file: `acq_summaries/summaries_combined.txt`
- final direct-summarization file: `acq_summaries/final_summary.txt`
- RAG workflow artifacts are managed by the shared RAG path when `rag_embedding` is provided.

## CLI

The interactive CLI registers this agent as:

```text
web
```
