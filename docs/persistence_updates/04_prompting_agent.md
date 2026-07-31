# PromptingAgent

`PromptingAgent` helps turn a rough request into a clearer, self-contained prompt for a downstream URSA agent or workflow.

It is designed for prompt refinement, not direct tool execution.

## What it does

PromptingAgent:

- accepts an initial rough prompt
- proposes a refined downstream-agent prompt
- accepts feedback in later turns
- revises the prompt
- marks the prompt approved when the user confirms it

Approval phrases include common confirmations such as:

```text
approve
approved
looks good
yes
ok
use it
done
final
confirm
```

## What it does not do

PromptingAgent does not bind or call tools directly.

Instead, it can include descriptions of tools that may be available to downstream agents such as ChatAgent or ExecutionAgent. This helps it write prompts that are realistic for the next agent to run.

## Python usage

```python
from ursa.agents import PromptingAgent

agent = PromptingAgent(
    llm=llm,
    agent_name="prompt-helper",
    use_web=False,
)

state = agent.invoke("Help me make a better prompt for analyzing my simulation output.")
print(agent.format_result(state))
```

On a later turn, provide feedback:

```python
state = agent.invoke("Make it emphasize uncertainty and reproducibility.", state)
```

Approve the current prompt:

```python
state = agent.invoke("approved", state)
print(agent.format_result(state))
```

## Dashboard usage

In the dashboard, choose the Prompting behavior in the composer when you want URSA to improve a prompt before running another behavior.

A common workflow:

1. Start or select a named agent session.
2. Choose Prompting.
3. Enter a rough task request.
4. Iterate on the proposed prompt.
5. Approve it.
6. Copy or use the approved prompt with Chat, Execution, or Planning + Execution.

## Tool context

PromptingAgent can be initialized with:

```python
PromptingAgent(llm=llm, use_web=True)
```

When `use_web=True`, the prompt-writing context describes downstream web, OSTI, and arXiv search tools as potentially available to downstream agents.

You can also pass extra execution tools for prompt context:

```python
PromptingAgent(
    llm=llm,
    extra_execution_tools=[my_tool],
)
```

This describes the tools to the PromptingAgent; it does not make PromptingAgent call them.

## Best use cases

Use PromptingAgent when:

- the task is complex or underspecified
- you want a clearer instruction set before running code
- you want to convert a conversation into a standalone task prompt
- you want to prepare a prompt for Planning + Execution
