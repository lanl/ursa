# Using ArxivAgent and ExecutionAgent for Astrophysics Research

This guide demonstrates how to use the `ArxivAgent` and `ExecutionAgent` together to research neutron star properties and generate a comprehensive analysis.

## Overview

This workflow allows you to:
1. Search arXiv for papers on neutron star radius constraints
2. Process and summarize the research findings
3. Generate a markdown document with data visualization

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
    max_results=5,
    database_path="arxiv_papers_neutron_star",
    summaries_path="arxiv_summaries_neutron_star",
    vectorstore_path="arxiv_vectorstores_neutron_star",
    download_papers=True,
)

# Run a search on neutron star radius constraints
research_results = arxiv_agent.invoke(
    arxiv_search_query="Experimental Constraints on neutron star radius",
    context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
)

# Initialize the ExecutionAgent
executor = ExecutionAgent(llm=model)

# Create a task for the ExecutionAgent
execution_plan = f"""
The following is the summaries of research papers on the contraints on neutron
star radius: 
{research_results}

Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
will be reviewed by experts in the field so technical accuracy and clarity is 
critical.
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
| `max_results` | Maximum number of papers to retrieve (5 in this example) |
| `database_path` | Path to store downloaded papers |
| `summaries_path` | Path to store paper summaries |
| `vectorstore_path` | Path to store vector embeddings |
| `download_papers` | Whether to download full papers (boolean) |

### ExecutionAgent

| Parameter | Description |
|-----------|-------------|
| `llm` | Language model to use for code generation and execution |

## Workflow Steps

1. **Research Phase**: ArxivAgent searches arXiv for papers on neutron star radius constraints, downloads them, and analyzes their content.

2. **Analysis Phase**: ExecutionAgent processes the research summaries and generates code to visualize the constraints and uncertainties.

3. **Output**: The ExecutionAgent produces a markdown document with analysis, visualizations, and insights about neutron star radius constraints.

## Use Cases

- Astrophysics literature reviews
- Compilation of experimental constraints on astronomical objects
- Visualization of scientific data from multiple sources
- Creation of technical reports for expert audiences

## Notes

- The quality of analysis depends on the available papers and the capabilities of the language model
- Consider adjusting `max_results` based on the breadth of research needed
- Ensure proper storage paths are set to avoid conflicts with other research projects
