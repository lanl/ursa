# Using ArxivAgent and ExecutionAgent Together

This guide demonstrates how to combine the `ArxivAgent` and `ExecutionAgent` for comprehensive research and analysis workflows.

## Overview

This workflow enables you to:
1. Search and analyze papers from arXiv
2. Process the research findings
3. Generate executable code to analyze and visualize the data

## Basic Usage

```python
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from ursa.agents import ArxivAgent, ExecutionAgent

# Initialize the language model
model = init_chat_model(
    model="openai:gpt-5-mini",
    max_tokens=50000,
)

# Initialize the ArxivAgent
arxiv_agent = ArxivAgent(
    llm=model,
    summarize=True,
    process_images=False,
    max_results=20,
    database_path="arxiv_papers_materials1",
    summaries_path="arxiv_summaries_materials1",
    vectorstore_path="arxiv_vectorstores_materials1",
    download_papers=True,
)

# Run a search and analysis
research_results = arxiv_agent.invoke(
    arxiv_search_query="high entropy alloy hardness",
    context="What data and uncertainties are reported for hardness of the high entropy alloy and how that that compare to other alloys?",
)

# Initialize the ExecutionAgent
executor = ExecutionAgent(llm=model)

# Create a task for the ExecutionAgent
execution_plan = f"""
The following is the summaries of research papers on the high entropy alloy hardness: 
{research_results}

Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
will be reviewed by experts in the field so technical accuracy and clarity is critical.
"""

# Prepare input for the ExecutionAgent
init = {"messages": [HumanMessage(content=execution_plan)]}

# Execute the plan
final_results = executor.invoke(init)

# Display results
for message in final_results["messages"]:
    print(message.content)
```

## Parameters

### ArxivAgent

| Parameter | Description |
|-----------|-------------|
| `llm` | Language model to use for analysis |
| `summarize` | Whether to summarize papers (boolean) |
| `process_images` | Whether to process images in papers (boolean) |
| `max_results` | Maximum number of papers to retrieve |
| `database_path` | Path to store downloaded papers |
| `summaries_path` | Path to store paper summaries |
| `vectorstore_path` | Path to store vector embeddings |
| `download_papers` | Whether to download full papers (boolean) |

### ExecutionAgent

| Parameter | Description |
|-----------|-------------|
| `llm` | Language model to use for code generation and execution |

## Workflow Steps

1. **Research Phase**: ArxivAgent searches arXiv for relevant papers based on your query, downloads them, and analyzes their content according to your context.

2. **Analysis Phase**: ExecutionAgent takes the research results and generates code to analyze and visualize the data.

3. **Output**: The ExecutionAgent produces a markdown document with analysis, visualizations, and insights from the research.

## Use Cases

- Literature reviews on scientific topics
- Data extraction and visualization from research papers
- Comparative analysis across multiple publications
- Technical report generation

## Notes

- Ensure you have sufficient disk space for paper storage
- Processing a large number of papers may take significant time
- The quality of analysis depends on the capabilities of the chosen language model
