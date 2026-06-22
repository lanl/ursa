# ExecutionAgent Documentation

`ExecutionAgent` is URSA's execution-focused agent for carrying out concrete tasks in a workspace. It can write files, edit code, read artifacts, download data, execute safe shell commands, and use optional search/RAG/MCP tools. Current implementations also include a structured **review-until-complete** loop before the final recap.

Use `ExecutionAgent` when the request is action-oriented: create or modify files, run code, inspect command output, perform analyses, or assemble artifacts. Use `ChatAgent` for lighter conversational assistance with tools.

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ExecutionAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = ExecutionAgent(llm=llm, workspace="analysis_workspace")

state = agent.invoke("Write and execute a Python script to print the first 10 integers.")
print(agent.format_result(state))
```

The returned state contains the conversation/tool transcript in `state["messages"]` and the final recap as the last message.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for execution, review, and recap. |
| `log_state` | `bool` | `False` | Write debug state to `execution_agent.json`. |
| `extra_tools` | `list[BaseTool] \| None` | `None` | Additional LangChain tools to bind. |
| `tokens_before_summarize` | `int` | `50000` | Token budget before message context is summarized/compacted. |
| `messages_to_keep` | `int` | `20` | Number of recent messages to preserve during context compaction. |
| `use_web` | `bool` | `False` | Add web, OSTI, and arXiv search tools. |
| `safe_codes` | `list[str] \| None` | `['python', 'julia']` | Interpreters/languages treated as safe command contexts by the command-safety machinery. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `AgentWithTools`, including workspace, persistence, groups, RAG tools, MCP, and checkpoint options. |

## Default tools

The default execution tool set includes:

- `run_command` — execute shell commands in the workspace with safety review.
- `write_code` — write new files.
- `edit_code` — edit existing files.
- `read_file` — read files, including PDF text extraction support.
- `download_file_tool` — download files from URLs.
- `read_image_tool` — load images.
- `list_experiences` — list durable Markdown experience files.
- `read_experience` — read an experience file.
- `write_experience` — write or append an experience file.
- `edit_experience` — edit an experience file.

When `use_web=True`, the agent also receives:

- `run_web_search`
- `run_osti_search`
- `run_arxiv_search`

You can add custom tools through `extra_tools`, persistent RAG tools through `rag_tools`, or MCP tools through the CLI/MCP configuration.

## Review-until-complete workflow

`ExecutionAgent` compiles a LangGraph state machine with four main nodes:

- `agent` — the executor LLM decides what to do next and may request tool calls.
- `action` — executes requested tools.
- `review` — asks the LLM for a structured completeness assessment.
- `recap` — produces the final concise summary returned to the user.

The current loop is:

1. The executor receives the user request and current message context.
2. If it requests tool calls, the `action` node executes them.
3. Tool results return to the executor for the next step.
4. When the executor emits an ordinary assistant response with no tool calls, the graph routes to `review` rather than finishing immediately.
5. The review node evaluates whether the work adequately addresses the original user request.
6. If the review says the work is incomplete, the review rationale is appended as a new human feedback message and the graph loops back to `agent`.
7. If the review says the work is complete, the graph proceeds to `recap` and finishes.

The structured review object is represented as:

```python
class ReviewAssessment(BaseModel):
    is_complete: bool
    reason: str
```

The graph stores this value in `state["review"]`. Incomplete reviews intentionally drive further work, so the agent can recover from premature “done” responses.

## State model

`ExecutionState` includes these key fields:

- `messages` — ordered system, human, AI, and tool messages.
- `symlinkdir` — optional symlink metadata for exposing external directories inside the workspace.
- `review` — optional `ReviewAssessment` from the structured review node.

The last message after a successful invocation is the recap. `agent.format_result(state)` returns that message text.

## Custom tools

```python
from math import sqrt
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from ursa.agents import ExecutionAgent

@tool
def hypotenuse(a: int, b: int) -> float:
    """Compute sqrt(a**2 + b**2)."""
    return sqrt(a**2 + b**2)

llm = init_chat_model("openai:gpt-5.4-mini")
agent = ExecutionAgent(llm=llm, extra_tools=[hypotenuse])
state = agent.invoke("Use the hypotenuse tool for a=3 and b=4.")
print(agent.format_result(state))
```

## Workspace and safety notes

- Files are written to and commands run from the configured workspace.
- Use a dedicated workspace for agent-driven execution.
- Shell command execution is safety-gated and unsafe command/tool results can be surfaced back to the graph.
- Web/search tools are opt-in with `use_web=True` or `ursa --use-web` in the CLI.
- Long transcripts are compacted using the shared message-context preparation helpers according to the configured token and message limits.
