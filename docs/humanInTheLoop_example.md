# URSA Human-in-the-Loop Agent Interface Documentation

## Overview

This module implements a human-in-the-loop (HITL) interface for the URSA agent framework, allowing users to directly interact with different specialized AI agents through a command-line interface. The system maintains context between agent interactions and provides persistent storage for agent states.

The file can be run with:

`python /path/to/hitl_basic.py`

### Setup and Initialization

1. Creates a workspace directory for storing agent data and checkpoints
2. Initializes SQLite databases and checkpointers for various agents:
   - Executor agent
   - Planner agent
   - WebSearcher agent
3. Configures the language model and embedding model
4. Instantiates the following agents:
   - ArxivAgent
   - ExecutionAgent
   - PlanningAgent
   - WebSearchAgent
   - RecallAgent

### User Interaction Loop

The function runs a continuous interaction loop until the user enters "[USER DONE]":

1. Displays a header explaining how to use the interface
2. Prompts the user for input
3. Parses the input to determine which agent to invoke:
   - `[Arxiver]`: Searches ArXiv for academic papers
   - `[Executor]`: Executes code or commands
   - `[Planner]`: Creates plans or strategies
   - `[WebSearcher]`: Performs web searches
   - `[Rememberer]`: Retrieves information from memory
   - `[Chatter]`: Has a general conversation using the language model
4. **Importantly, the output from the previous agent interaction is automatically included in the prompt to the next agent.** This creates a continuous context flow where each agent has access to what the previous agent produced.
5. Invokes the appropriate agent with the user's query and context from previous interactions
6. Displays the agent's response
7. Continues the loop until the user indicates they're done

### Agent-Specific Behavior

- **ArxivAgent**: Converts user query into a search query, retrieves relevant papers, and summarizes results
- **ExecutionAgent**: Processes user instructions in the context of previous outputs, can execute code
- **PlanningAgent**: Creates plans based on user requirements and previous context
- **WebSearchAgent**: Performs web searches based on user queries
- **RecallAgent**: Retrieves relevant information from persistent memory
- **Chatter**: Provides direct access to the language model for general conversation

## Usage

Run the script and interact with agents by prefixing your queries with the appropriate agent tag:

```
[Arxiver] Find recent papers on transformer architecture improvements
[Executor] Write a Python function to calculate Fibonacci numbers
[Planner] Create a research plan for analyzing climate data
[WebSearcher] What are the latest developments in quantum computing?
[Rememberer] What did we discuss about neural networks earlier?
[Chatter] Explain the concept of attention mechanisms in simple terms
```

### Context Continuity

A key feature of this interface is that each agent receives both your current query AND the output from the previous agent interaction. This allows for natural follow-up queries and building on previous results. For example:

1. `[WebSearcher] Find information about large language models`
2. `[Planner] Create a research plan based on this information`

In this sequence, the Planner would receive both your planning request AND the search results about large language models from the WebSearcher, enabling it to create a more informed and contextually relevant plan.

Use "[USER DONE]" to exit the interface.

