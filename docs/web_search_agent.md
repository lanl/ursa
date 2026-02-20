# WebSearchAgent Documentation

`WebSearchAgent` (current implementation) is an acquisition agent that uses `ddgs` search, downloads/web-scrapes sources, and returns a synthesized summary in context.

## Basic Usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import WebSearchAgent

llm = init_chat_model("openai:gpt-5.2")
websearcher = WebSearchAgent(llm=llm, max_results=3)

result = websearcher.invoke({
    "query": "Detroit Tigers top prospects 2025 birth year",
    "context": "Who are the top prospects and what year were they born?",
})
print(result["final_summary"])
```

## Parameters

- `llm`: required chat model
- `max_results`: max search hits to materialize
- `summarize`: summarize fetched content (`True` by default)
- `download`: if `False`, uses cached files from `database_path`
- `database_path`, `summaries_path`, `vectorstore_path`: storage folders under workspace

## Features

- DuckDuckGo discovery via `ddgs`
- HTML/PDF materialization to local cache
- boilerplate-stripped text extraction
- per-source summaries + final aggregate summary

## Output

- `final_summary`: synthesized answer
- `items`: fetched source items (metadata/content)
- `summaries`: per-item summaries

## Notes

- This documentation covers the current exported `ursa.agents.WebSearchAgent`.
- Legacy `WebSearchAgentLegacy` has been retired. Use `ursa.agents.WebSearchAgent`.
