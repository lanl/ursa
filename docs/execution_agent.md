# ExecutionAgent Documentation

`ExecutionAgent` is a class that enables AI-powered code execution, writing, and editing. It uses a state machine architecture to safely execute commands, write code files, and search for information.

## Basic Usage

```python
from ursa.agents import ExecutionAgent

# Initialize the agent
agent = ExecutionAgent()

# Run a prompt
result = agent.invoke("Write and execute a python script to print the first 10 integers.")

# Access the final response
print(result["messages"][-1].content)
```

## Parameters

When initializing `ExecutionAgent`, you can customize its behavior with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | str or BaseChatModel | "openai/gpt-4o-mini" | The LLM model to use |
| `**kwargs` | dict | {} | Additional parameters passed to the base agent |

## Features

### Code Execution

The agent can safely execute shell commands in a controlled environment:

```python
result = agent.invoke("Install numpy and create a script that uses it to calculate the mean of [1, 2, 3, 4, 5]")
```

### Code Writing

The agent can write code files to a workspace directory:

```python
result = agent.invoke("Create a Flask web application that displays 'Hello World'")
```

## Advanced Usage

### Customizing the Workspace

The agent creates a workspace folder with a randomly generated name for each run. You can access this workspace path from the result:

```python
result = agent.invoke("Create a Python script")
workspace_path = result["workspace"]
print(f"Files were created in: {workspace_path}")
```

### Setting a Recursion Limit

For complex tasks, you may need to adjust the recursion limit:

```python
result = agent.invoke(
    "Create a complex project with multiple files and tests", 
    recursion_limit=2000
)
```

### Safety Features

The agent includes built-in safety checks for shell commands:

1. Commands are evaluated for safety before execution
2. Unsafe commands are blocked with explanations
3. The agent suggests safer alternatives when appropriate

## How It Works

1. **State Machine**: The agent uses a directed graph to manage its workflow:
   - `agent` node: Processes user requests and generates responses
   - `safety_check` node: Evaluates command safety
   - `action` node: Executes tools (run_cmd, write_code, edit_code, search)
   - `summarize` node: Creates a final summary when complete

2. **Tools**:
   - `run_cmd`: Executes shell commands in the workspace directory
   - `write_code`: Creates new code files with syntax highlighting
   - `edit_code`: Modifies existing code files with diff preview
   - `search_tool`: Performs web searches via DuckDuckGo

3. **Visualization**:
   - Code changes are displayed with syntax highlighting
   - File edits show detailed diffs
   - Command execution shows stdout and stderr

## Notes

- The agent creates a new workspace directory for each run
- Files are written to and executed from this workspace
- Shell commands have a 60000-second timeout by default
- The agent can handle keyboard interrupts during command execution