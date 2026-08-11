# Configuration

YAML files make model, workspace, agent, RAG, and MCP settings easy to reuse.
URSA configs can also be
[layered with environment variables and CLI flags][configuration-files-cli-flags-and-environment-variables],
with commands to inspect the resulting configuration.

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
agent_config: {}
mcp_servers: {}
```

Use:

```bash
ursa --print-config
```

to inspect the full active configuration, including defaults and null values.

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

## Inference providers

Use `inference_providers` to share endpoint and credential settings between
models:

```yaml
inference_providers:
  openai_public:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
llm_model:
  model: openai:gpt-5.4
  inference_provider: openai_public
emb_model:
  model: openai:text-embedding-3-large
  inference_provider: openai_public
```

Models inherit unspecified provider settings; model-specific values override
them. The selected provider must exist. Set a nullable model value to `null` to
clear an inherited value.

## `use_web` and `agent_config`

Use `agent_config` for per-agent options. Top-level `use_web: true` enables web
tools for supported agents unless an agent overrides it:

```yaml
use_web: true
agent_config:
  prompt:
    use_web: false
```

## `workspace: tmp`

Set `workspace: tmp` to create a temporary workspace for the run. If a local
directory named `tmp` already exists, URSA uses that directory.

## More configuration topics

- [OpenAI-compatible endpoints][openai-compatible-endpoints]
- [Ollama and local endpoints][ollama-and-local-endpoints]
- [LangChain providers][langchain-providers]
- [Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
- [MCP server configuration][mcp-server-configuration]
