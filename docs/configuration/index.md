# Configuration

YAML configuration files are the preferred way to configure URSA. They make model settings, workspaces, groups, agent options, RAG tools, and MCP servers easy to reuse and version with a project.

URSA can also be configured with CLI flags and environment variables, but the recommended order is:

1. **YAML configuration files** for reusable project settings.
2. **CLI arguments** for one-off overrides.
3. **Environment variables** mainly for secrets and automation.

## Minimal config file

Create `config.yaml`:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY

workspace: ./ursa-workspace
```

Run URSA with:

```bash
ursa --config config.yaml
```

## Common top-level settings

```yaml
workspace: ./ursa-workspace
group: default
thread_id: null
use_web: false

llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
  max_completion_tokens: 10000

emb_model: null
rag_tools: []
agent_config: null
mcp_servers: {}
```

Use:

```bash
ursa --print-config
```

to inspect the active configuration and defaults.

## Model configuration

URSA uses LangChain's unified model initialization. Model names usually use this form:

```text
<provider>:<model-name>
```

Examples:

```yaml
llm_model:
  model: openai:gpt-5.4
```

```yaml
llm_model:
  model: anthropic:claude-sonnet-4-5
```

```yaml
llm_model:
  model: google_genai:gemini-2.5-pro
```

```yaml
llm_model:
  model: ollama:gpt-oss-2b
  base_url: http://localhost:11434
```

## Prefer `api_key_env` for secrets

Avoid hard-coding API keys in YAML files. Prefer:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
```

Then set the key in your shell or secret manager.

## More configuration topics

- [OpenAI-compatible endpoints][openai-compatible-endpoints]
- [Ollama and local endpoints][ollama-and-local-endpoints]
- [LangChain providers][langchain-providers]
- [Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
- [MCP server configuration][mcp-server-configuration]
