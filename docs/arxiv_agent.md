# ArxivAgent Documentation

`ArxivAgent` (current implementation) is an acquisition agent that fetches ArXiv papers, extracts content, and returns context-aware summaries.

## Basic Usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ArxivAgent

llm = init_chat_model("openai:gpt-5.2")
agent = ArxivAgent(llm=llm, max_results=3)

result = agent.invoke(
    query="Experimental Constraints on neutron star radius",
    context="What are the constraints on neutron star radius and what uncertainties are reported?",
)

print(result["final_summary"])
```

## Parameters

- `llm`: required chat model
- `max_results`: number of papers to fetch
- `summarize`: summarize fetched items (`True` default)
- `process_images`: attempt image extraction + vision description
- `download`: if `False`, use local cache in `database_path`
- `database_path`, `summaries_path`, `vectorstore_path`: workspace-relative folders

## Advanced Usage

### Customizing the Agent

```python
agent = ArxivAgent(
    llm=llm,
    max_results=5,
    process_images=False,
    download=False,
)
```

## Notes

- Returned state includes `items`, optional per-item `summaries`, and `final_summary`.
- Legacy `ArxivAgentLegacy` has been retired. Use `ursa.agents.ArxivAgent`.
