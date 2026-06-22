# Agent Behaviors and Tools

URSA now works best if you separate persistent identity from behavior.

A named agent can be reused with different behaviors:

- Chat
- Execution
- Planning + Execution
- Prompting
- other registered dashboard agents

The behavior determines the graph, prompts, and tools available for a run. The name and group determine durable storage and security scope.

## ChatAgent

`ChatAgent` is a tool-capable conversational agent.

Default tools include file/code/workspace and experience tools:

- `run_command`
- `write_code`
- `edit_code`
- `read_file`
- `read_image_tool`
- `list_experiences`
- `write_experience`
- `read_experience`
- `edit_experience`

If web access is enabled with `use_web=True`, additional search tools are included:

- web search
- OSTI search
- arXiv search

Python example:

```python
from ursa.agents import ChatAgent

agent = ChatAgent(
    llm=llm,
    agent_name="lab-assistant",
    use_web=False,
)
```

CLI example:

```bash
ursa --name lab-assistant
```

## ExecutionAgent

`ExecutionAgent` is intended for model-driven code execution, shell interaction, file editing, and iterative execution workflows.

It uses a dedicated execution graph with:

- executor prompting
- tool calls
- recap/summarization behavior
- optional extra tools

Default tools overlap with ChatAgent but the behavior is different. ExecutionAgent is more focused on carrying out computational or workspace-changing tasks.

Python example:

```python
from ursa.agents import ExecutionAgent

agent = ExecutionAgent(
    llm=llm,
    agent_name="lab-assistant",
    safe_codes=["python", "julia"],
    use_web=False,
)
```

## ChatAgent vs ExecutionAgent

Use ChatAgent when you want a conversational assistant that can use tools as needed.

Use ExecutionAgent when you want a run loop designed for execution-heavy tasks, code generation/editing, shell commands, and recap of progress.

The same persistent named agent can be used with either behavior:

```python
chat = ChatAgent(llm=llm, agent_name="lab-assistant")
executor = ExecutionAgent(llm=llm, agent_name="lab-assistant")
```

This does not mean the two classes are identical. It means they can operate against the same named persistent storage.

## Shared message-history utilities

Shared context maintenance was moved into `BaseAgent` and used by current message/tool graph paths.

This includes utilities for:

- preparing message context
- repairing dangling tool-call history in supported paths
- summarizing long context
- truncating oversized tool messages
- updating message state safely when history was modified

The goal is to make long-running sessions more robust across ChatAgent, ExecutionAgent, and other message-based agents using the shared helpers.

## RAG tools

Persisted RAG agents can be attached as tools to tool-capable agents.

Python:

```python
agent = ChatAgent(
    llm=llm,
    agent_name="lab-assistant",
    rag_tools=["papers", "lab-notes"],
)
```

CLI:

```bash
ursa --name lab-assistant --rag-tools papers,lab-notes
```

Dashboard:

- open Settings
- choose Agent tools
- add persisted RAG agents from the active group
- click Save

Selected RAG tools are attached to new Chat, Execution, and Planning + Execution runs.

## Behavior switching in the dashboard

The dashboard lets a session keep a persistent agent name while choosing an agent behavior per message.

For example, the same named agent can:

1. discuss a plan with Chat
2. refine instructions with Prompting
3. run code with Execution
4. use Planning + Execution for a larger task

Prompt/context history records agent-type labels so switching behavior remains visible in the transcript.
