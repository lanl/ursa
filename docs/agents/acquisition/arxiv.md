# ArxivAgent Documentation

[`ArxivAgent`][ursa.agents.acquisition_agents.ArxivAgent] is an acquisition
agent for arXiv papers. It subclasses
[`BaseAcquisitionAgent`][ursa.agents.acquisition_agents.BaseAcquisitionAgent],
so it uses the same acquire-then-summarize/RAG graph as WebSearchAgent and
OSTIAgent.

`ArxivAgent` searches the arXiv API, downloads matching paper PDFs, extracts text, optionally augments PDF text with image descriptions, and then summarizes the acquired papers or runs the RAG path when an embedding model is configured.

See also: [Acquisition Agents][acquisition-agents].

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ArxivAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = ArxivAgent(llm=llm)

state = agent.invoke({
    "query": "neutron star radius constraints",
    "context": "What are the constraints on neutron star radius, and what uncertainties affect those constraints?",
})

print(agent.format_result(state))
```

You can also pass a plain string. In that case, the string becomes `context`, and the agent asks the LLM to generate a short arXiv search query:

```python
state = agent.invoke("Summarize recent arXiv papers about neutron star radius constraints.")
print(agent.format_result(state))
```

## Parameters

`ArxivAgent` uses the shared `BaseAcquisitionAgent` parameters with arXiv-specific defaults:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for query generation and summarization. |
| `process_images` | `bool` | `True` | For downloaded PDFs, optionally append image interpretations when image-description support is available. |
| `max_results` | `int` | `3` | Maximum number of arXiv results to acquire. |
| `download` | `bool` | `True` | If `True`, search arXiv and download PDFs. If `False`, read cached files from `database_path`. |
| `rag_embedding` | optional embedding object | `None` | If provided, use the RAG path instead of direct per-paper summarization. |
| `database_path` | `str` | `"arxiv_papers"` | Directory under the agent den for downloaded/cached papers. |
| `summaries_path` | `str` | `"arxiv_generated_summaries"` | Directory under the agent den for per-paper and final summaries. |
| `vectorstore_path` | `str` | `"arxiv_vectorstores"` | Stored vector-store path configuration inherited from the acquisition base. The current shared RAG node constructs a `RAGAgent` over the acquired database when `rag_embedding` is provided. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `BaseAcquisitionAgent`, including workspace/den and persistence options. |

Inherited acquisition parameters such as `summarize` and `num_threads` can also be supplied through the shared base class path.

## How it works

`ArxivAgent` implements the acquisition hooks required by `BaseAcquisitionAgent`:

- `_search(query)` — queries `http://export.arxiv.org/api/query` and normalizes entries into lightweight hits containing arXiv IDs and titles.
- `_id(hit_or_item)` — uses the arXiv ID as the stable item ID.
- `_materialize(hit)` — downloads `https://arxiv.org/pdf/<arxiv_id>.pdf`, stores it in `database_path`, and extracts PDF text.
- `_citation(item)` — formats citations as `ArXiv ID: <id>`.

After acquisition, the shared graph either:

- summarizes each paper and aggregates the summaries into `state["final_summary"]`, or
- invokes the RAG path when `rag_embedding` is configured, or
- stops after `state["items"]` when `summarize=False`.

## Output state

Important fields in the returned state include:

- `query` — final arXiv search query.
- `context` — task/question used for summarization.
- `items` — acquired paper metadata, including local PDF paths and extracted text.
- `summaries` — per-paper summaries when direct summarization is used.
- `final_summary` — aggregate response to the requested context.

`agent.format_result(state)` returns `state["final_summary"]` when present.

## Cached files

By default, the agent writes artifacts under the agent den:

- PDFs: `arxiv_papers/`
- per-paper summaries: `arxiv_generated_summaries/`
- combined direct-summarization file: `arxiv_generated_summaries/summaries_combined.txt`
- final direct-summarization file: `arxiv_generated_summaries/final_summary.txt`
- RAG workflow artifacts are managed by the shared RAG path when `rag_embedding` is provided.

## CLI

The interactive CLI registers this agent as:

```text
arxiv
```
