# WebSearchAgent Documentation

`WebSearchAgent` is a powerful tool for conducting internet-based research on any topic. It leverages language models and web search capabilities to gather, process, and summarize information from online sources.

## Basic Usage

```python
from ursa.agents import WebSearchAgent
from langchain_openai import ChatOpenAI

# Initialize with default model (gpt-5-mini)
websearcher = WebSearchAgent()

# Or initialize with a custom model
model = ChatOpenAI(model="gpt-5-mini", max_completion_tokens=10000)
websearcher = WebSearchAgent(llm=model)

# Run a web search query
result = websearcher.invoke("Who are the 2025 Detroit Tigers top 10 prospects and what year were they born?")

# Access the web search results
final_summary = result["messages"][-1].content
sources = result["urls_visited"]

print("Web Search Summary:")
print(final_summary)
print("Sources:", sources)
```

## Parameters

### Initialization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | init_chat_model("openai:gpt-5-mini") | The language model to use for web search |
| `**kwargs` | `dict` | `{}` | Additional parameters passed to the base agent |

### Run Method

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | Required | The web search question or topic |
| `recursion_limit` | int | 100 | Maximum recursion depth for the web search process |

## Features

- **Automated Web Search**: Uses DuckDuckGo to find relevant information
- **Content Processing**: Extracts and summarizes content from web pages
- **Iterative Web Search**: Continues researching until sufficient information is gathered
- **Source Tracking**: Records all URLs visited during research
- **Internet Connectivity Check**: Verifies internet access before attempting research

## Output

The agent returns a dictionary containing:

- `messages`: A list of message objects, with the final message containing the comprehensive web search summary
- `urls_visited`: A list of all sources consulted during the web search process

## Advanced Usage

```python
from langchain.chat_model import init_chat_model
from ursa.agents import WebSearchAgent

# Initialize with custom parameters
websearcher = WebSearchAgent(
    llm=init_chat_model("openai:gpt-5-mini"),
    url="https://www.example.com"  # Custom URL for internet connectivity check
)

# Run with higher recursion limit for complex topics
result = websearcher.invoke(
    "What are the latest developments in quantum computing? Summarize in markdown format.",
    recursion_limit=200
)
```

## Notes

- The agent requires internet connectivity to function properly
- Rate limiting is implemented to avoid overwhelming search services
- For networks with SSL inspection, you may need to set the `CERT_FILE` environment variable
- The websearch process includes multiple steps: search, content processing, review, and final summarization
