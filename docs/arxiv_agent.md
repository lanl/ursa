# ArxivAgent Documentation

`ArxivAgent` is a class that helps fetch, process, and summarize scientific papers from arXiv. It uses LLMs to generate summaries of papers relevant to a given query and context.

## Basic Usage

```python
from ursa.agents import ArxivAgent

# Initialize the agent
agent = ArxivAgent()

# Run a query
result = agent.invoke(
    arxiv_search_query="Experimental Constraints on neutron star radius", 
    context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?"
)

# Print the summary
print(result)
```

## Parameters

When initializing `ArxivAgent`, you can customize its behavior with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | `init_chat_model("openai:gpt-5-mini")` | The LLM model to use for summarization |
| `summarize` | bool | True | Whether to summarize the papers or just fetch them |
| `process_images` | bool | True | Whether to extract and describe images from papers |
| `max_results` | int | 3 | Maximum number of papers to fetch from arXiv |
| `database_path` | str | 'arxiv_papers' | Directory to store downloaded PDFs |
| `summaries_path` | str | 'arxiv_generated_summaries' | Directory to store paper summaries |
| `vectorstore_path` | str | 'arxiv_vectorstores' | Directory to store vector embeddings |
| `download_papers` | bool | True | Whether to download papers or use existing ones |

## Advanced Usage

### Customizing the Agent

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ArxivAgent

agent = ArxivAgent(
    llm=init_chat_model("openai:gpt-5-mini"),  # Use a more powerful model
    max_results=5,       # Fetch more papers
    process_images=False,  # Skip image processing to save time
    download_papers=False  # Use only papers already in database_path
)
```

### Running Multiple Queries

```python
# First query
result1 = agent.invoke(
    arxiv_search_query="quantum computing error correction", 
    context="Summarize recent advances in quantum error correction techniques"
)

# Second query (will reuse downloaded papers if applicable)
result2 = agent.invoke(
    arxiv_search_query="quantum computing algorithms", 
    context="What are the most promising quantum algorithms for near-term devices?"
)
```

## How It Works

1. **Fetching Papers**: The agent searches arXiv for papers matching your query and downloads them as PDFs.

2. **Processing**: If `summarize=True`, each paper is:
   - Converted to text
   - Split into chunks
   - Embedded into a vector database
   - If `process_images=True`, images are extracted and described using GPT-4 Vision

3. **Summarization**: The agent:
   - Retrieves the most relevant chunks based on your context
   - Generates a summary for each paper
   - Creates a final summary addressing your specific context

4. **Output**: Returns a comprehensive summary that synthesizes information from all relevant papers.

## Notes

- Summaries and vector stores are cached, making subsequent queries faster.
- The agent uses a ThreadPoolExecutor to process papers in parallel.
- You can find the combined summaries in 'summaries_combined.txt' and the final summary in 'final_summary.txt'.
