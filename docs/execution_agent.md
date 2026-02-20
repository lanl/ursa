# ExecutionAgent Documentation

`ExecutionAgent` runs iterative plan/act loops with tools for reading files, writing/editing code, running commands, and doing acquisition-style web/literature lookups.

## Basic Usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ExecutionAgent

llm = init_chat_model("openai:gpt-5.2")
agent = ExecutionAgent(llm=llm, workspace="ursa_workspace")

result = agent.invoke("Write and run a Python script that prints the first 10 integers.")
print(result["messages"][-1].text)
```

## Key Parameters

- `llm`: required `BaseChatModel`
- `workspace`: directory for files and command execution (default `ursa_workspace`)
- `extra_tools`: optional additional tools
- `safe_codes`: trusted language/tool hints for command safety checks
- `tokens_before_summarize`, `messages_to_keep`: context compaction controls

## Graph Shape

- `agent`: LLM decides next action
- `action`: executes tool calls
- `recap`: summarizes final result

The graph loops `agent -> action -> agent` until no tool calls remain, then goes to `recap`.

## Built-in Tools

- `run_command`
- `write_code`
- `edit_code`
- `read_file`
- `run_web_search`
- `run_osti_search`
- `run_arxiv_search`

## Safety Notes

- `run_command` performs an LLM safety check before execution.
- Unsafe commands are blocked and returned with a reason.
