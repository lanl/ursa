# ChatAgent Documentation

[`ChatAgent`][ursa.agents.chat_agent.ChatAgent] is URSA's general conversational
agent. Unlike [`BasicChatAgent`][ursa.agents.chat_agent.BasicChatAgent], the
current public ChatAgent is tool-capable: it can answer conversationally, call
workspace/file tools when useful, and optionally use web/literature search
tools.

Use `ChatAgent` when you want an interactive assistant that can inspect files, record experience notes, download/read artifacts, or use configured retrieval tools while staying conversational. Use `ExecutionAgent` when you want a more execution-focused workflow with explicit completeness review and final recap.

## Basic usage

```python
from langchain.chat_models import init_chat_model
from ursa.agents import ChatAgent

llm = init_chat_model("openai:gpt-5.4-mini")
agent = ChatAgent(llm=llm)

state = agent.invoke("Summarize the files in this workspace.")
print(agent.format_result(state))
```

For conversational continuation, reuse the returned state through `format_query` or let the TUI maintain state for you.

```python
state = agent.invoke("Remember that this project studies alloy phase stability.")
query = agent.format_query("What should I look at next?", state=state)
state = agent.invoke(query)
print(agent.format_result(state))
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | required | Language model used for chat and tool selection. |
| `use_web` | `bool` | `False` | When true, adds web, OSTI, and arXiv search tools. |
| `**kwargs` | `dict` | `{}` | Passed to `BaseAgent` / `AgentWithTools`, including workspace, persistence, group, RAG tool, MCP, and checkpoint-related options. |

## Context summarization

Chat and execution agents summarize conversation history when its approximate
token count reaches **70% of the selected model's context window**. The count
includes the system prompt and agent tool definitions. Recent messages are
preserved according to `messages_to_keep`.

URSA resolves the window when initializing the CLI's selected model:

- An explicit `profile.max_input_tokens` or Ollama `num_ctx` takes precedence.
- Custom OpenAI-compatible endpoints are queried for LiteLLM `/model/info`,
  `/model_group/info`, or enriched `/models` metadata. Limits are matched by the
  selected public alias; if several deployments advertise limits, the smallest
  is used. Requests retain the configured client's authentication and TLS settings,
  use a three-second timeout, and do not retry connection failures.
- Ollama's `/api/show` can supply a model's configured `num_ctx`; `/api/ps` reports
  the allocated context of a running model. The TUI checks again after each model
  response. An unloaded model without `num_ctx` remains unknown until then:
  its architectural maximum does not establish the server's allocated window.
- Other models use their LangChain integration's local profile catalog. For an
  official OpenAI model missing from that catalog, URSA reads the current
  [models.dev catalog](https://models.dev), the upstream source of LangChain's
  model profiles. The public catalog is fetched anonymously once per process,
  with a three-second timeout. URSA keeps no model-specific OpenAI table.
  OpenAI's `/models` API does not expose context limits; a model can still be
  unknown if the catalog is unavailable or has not added it yet.

The resolved limit is shared by summarization and the TUI. `/status` shows its
source. Token checks and status rendering make no requests. Output limits such
as `max_tokens` are never treated as context limits.

Metadata routes can be absent, restricted by the API key, or missing a limit.
When the window is unknown, `tokens_before_summarize` is the fallback threshold
(default: `50000`). Set it globally with top-level `tokens_before_summarize` in
YAML or `URSA_TOKENS_BEFORE_SUMMARIZE`, or override it for an individual agent
under `agent_config`. This is a token threshold, not a context window to which
70% is applied. A public model name alone cannot establish a custom endpoint's
deployment limit. You can supply the actual limit explicitly in configuration:

```yaml
llm_model:
  model: my-model
  model_provider: openai
  base_url: https://inference.example/v1
  profile:
    max_input_tokens: 32768
```

The same `profile` can be passed when constructing a LangChain chat model.
This is a summarization trigger, not a guarantee that every request fits: token
counts are approximate, and a large new message or preserved tail can exceed
the window.

## Default tools

The current `ChatAgent` always binds the following tools:

- `run_command` — execute shell commands in the workspace with safety checks.
- `write_code` — write source/text files.
- `edit_code` — edit existing files.
- `read_file` — read files, including PDF text extraction support.
- `download_file_tool` — download a URL to the workspace.
- `read_image_tool` — load an image for model use.
- `list_experiences` — list persistent Markdown experience files.
- `write_experience` — create or append durable Markdown notes.
- `read_experience` — read a saved experience file back into context.
- `edit_experience` — edit a saved experience file.

When `use_web=True`, it also binds:

- `run_web_search`
- `run_osti_search`
- `run_arxiv_search`

If persistent RAG tools are configured through `rag_tools`, `AgentWithTools` can expose those as additional tools. MCP tools can also be attached in the TUI when MCP servers are configured.

## Graph behavior

`ChatAgent` compiles a small LangGraph loop:

1. The `respond` node calls the LLM with the current chat history and bound tools.
2. If the LLM response contains tool calls, the graph routes to `tool_node`.
3. `tool_node` executes the requested tools.
4. Control returns to `respond` so the model can use tool outputs.
5. When a model response has no tool calls, the graph finishes.

This means `ChatAgent` can use multiple tools over multiple turns, but it does not perform the explicit review-until-complete loop used by `ExecutionAgent`.

## BasicChatAgent

The module also contains `BasicChatAgent`, a simple chat-only implementation with no tool loop. It is useful for minimal conversational behavior, but `ChatAgent` is the public, tool-capable chat agent exported by `ursa.agents` and used by the TUI `chat` behavior.

## TUI usage

In the URSA TUI, use the `chat` agent. Web/search tools are opt-in through a CLI
flag:

```bash
ursa --use-web
```

Without `--use-web`, the chat agent still has local workspace and experience tools, but it does not have web, OSTI, or arXiv search tools.
