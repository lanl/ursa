# Configuration

OpenAI works with the built-in defaults after you set `OPENAI_API_KEY`; you do
not need a config file for the standard first run. YAML files make changed
model, workspace, agent, RAG, and MCP settings easy to reuse. URSA configs can be
[layered with environment variables and CLI flags][configuration-files-cli-flags-and-environment-variables],
with commands to inspect the resulting configuration.

## Prefer a user config

Put settings that should apply across projects in your platform's persistent
user config. For example, this changes only the embedding model:

```yaml
emb_model:
  model: openai:text-embedding-3-large
```

See [Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
for user config paths and precedence. Use `--config` for a project-specific or
one-off override, not as a requirement for ordinary OpenAI use.

## Define reusable inference providers

Put endpoint and credential settings under `inference_providers`, then select a
provider from each model. This keeps connection details in one place when chat
and embedding models share an endpoint:

```yaml
inference_providers:
  research_gateway:
    base_url: https://models.example.edu/v1
    api_key:
      env: RESEARCH_LLM_API_KEY
llm_model:
  model: openai:chat-model-name
  inference_provider: research_gateway
  max_completion_tokens: 10000
emb_model:
  model: openai:embedding-model-name
  inference_provider: research_gateway
```

The provider name is local to your configuration. A selected provider must
exist, and model-specific values override values inherited from it. Keep secrets
in environment references; put non-secret values such as `base_url` directly in
the YAML file.

For ordinary OpenAI use, the built-in `openai` provider already supplies the
endpoint. Set `OPENAI_API_KEY` and skip this configuration entirely.

Use this command to inspect the complete resolved configuration:

```bash
ursa --print-config
```

to inspect the full active configuration, including defaults and null values.

## Common top-level settings

```yaml
workspace: ./ursa-workspace
group: default
thread_id: null
use_web: false
llm_model:
  model: openai:gpt-5.4
emb_model: null
rag_tools: []
agent_config: {}
mcp_servers: {}
```

Model names normally use `<provider>:<model-name>`. See
[Models and inference providers][models-and-inference-providers] for complete,
tabbed examples covering OpenAI, OpenAI-compatible services, Anthropic, Google,
Ollama, and Azure OpenAI.

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

- [Models and inference providers][models-and-inference-providers]
- [Secrets][secrets]
- [Configuration files, CLI flags, and environment variables][configuration-files-cli-flags-and-environment-variables]
- [MCP server configuration][mcp-server-configuration]
