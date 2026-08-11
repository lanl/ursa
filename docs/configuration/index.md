# Configuration

YAML configuration files are the preferred way to configure URSA. They make model settings, workspaces, groups, agent options, RAG tools, and MCP servers easy to reuse and version with a project.

URSA configuration has two useful views:
- **merged config**: the result of applying configuration layers in precedence order
- **resolved config**: the merged config plus derived behavior.

URSA also accepts CLI flags and environment variables. See
[Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
for the authoritative precedence order.

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

to inspect the full resolved configuration, including defaults and null values.
Use `ursa --print-config=merged` to inspect the full configuration before
resolution.

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

Use `inference_providers` to define reusable provider-level settings once, then reference them from `llm_model` or `emb_model` with `inference_provider`.

```yaml
inference_providers:
  openai_public:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    ssl_verify: true
llm_model:
  model: openai:gpt-5.4
  inference_provider: openai_public
emb_model:
  model: openai:text-embedding-3-large
  inference_provider: openai_public
```

This is useful when multiple model configs share the same endpoint, credentials, or SSL settings. A model inherits provider settings that it does not specify. Model-specific settings override inherited values. For example, this keeps the shared endpoint but uses a different credential source for just the LLM:

```yaml
inference_providers:
  openai_public:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
llm_model:
  model: openai:gpt-5.4
  inference_provider: openai_public
  api_key_env: PROJECT_OPENAI_API_KEY
```

The selected provider must exist in `inference_providers`. Omit a model setting
to inherit it from the provider. For nullable settings, use `null` to clear an
inherited value.

### Sharing inference providers across configuration files

`inference_providers` works especially well with URSA's layered configuration.
A user-level config can define the available providers once, while a file passed
with `--config` can choose a provider or model for a particular project. See
[Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
for how those sources are merged.

For example, a user config might define reusable providers:

```yaml
inference_providers:
  openai_personal:
    base_url: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  openai_project:
    base_url: https://api.openai.com/v1
    api_key_env: PROJECT_OPENAI_API_KEY
llm_model:
  model: openai:gpt-5.4
  inference_provider: openai_personal
```

Then an explicitly selected config file can switch billing or model selection
without redefining the provider details:

```yaml
llm_model:
  model: openai:gpt-5.4-mini
  inference_provider: openai_project
```

Pass that file with `ursa --config ./.ursa/config.yaml`. This keeps a stable
provider catalog in user config while allowing each invocation to select the
right provider, API key, or model.

## `use_web` and `agent_config`

`use_web` is a top-level convenience setting. During config resolution, `use_web: true` fills in missing `use_web` values for these agents:
- `chat`
- `execute`
- `deep_review`
- `prompt`

Explicit agent settings win. For example:

```yaml
use_web: true
agent_config:
  prompt:
    use_web: false
```

Keeps web tools enabled for the other affected agents while leaving `prompt` with `use_web: false`.

`agent_config` is the main way to set per-agent options. It is a mapping from HITL agent name to that agent's configuration dictionary.

Example:

```yaml
agent_config:
  chat:
    use_web: true
  execute:
    safe_codes:
      - python
      - julia
  prompt:
    use_web: false
```

## `workspace: tmp`

If you set:

```yaml
workspace: tmp
```

URSA will use a temporary directory for the run instead of literal folder named `tmp`.
If a literal `tmp` folder exists in the current directory, URSA *will* use that instead.

## More configuration topics

- [OpenAI-compatible endpoints][openai-compatible-endpoints]
- [Ollama and local endpoints][ollama-and-local-endpoints]
- [LangChain providers][langchain-providers]
- [Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
- [MCP server configuration][mcp-server-configuration]
