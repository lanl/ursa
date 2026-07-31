# Configuration files, CLI flags, and environment variables

URSA supports three configuration mechanisms. Prefer YAML files for project settings, CLI flags for temporary overrides, and environment variables for secrets or automation.

## YAML files: preferred

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY

workspace: ./ursa-workspace
group: default
use_web: false
```

Run:

```bash
ursa --config config.yaml
```

YAML files are best when you want to reuse the same model, workspace, group, MCP server, or agent settings across multiple runs.

## CLI flags: useful overrides

```bash
ursa --config config.yaml --llm_model.model openai:gpt-5.4
```

Common flags include:

```text
--workspace
--group
--thread_id
--use_web
--name
--llm_model.model
--llm_model.base_url
--llm_model.api_key_env
--llm_model.ssl_verify
--llm_model.max_completion_tokens
--emb_model
--mcp_servers
--rag-tools
```

Use `ursa --help` for the authoritative list.

## Environment variables: secrets and automation

URSA exposes environment-variable equivalents for many CLI settings, but for most users environment variables are best for API keys and automated deployment.

Example:

```bash
export OPENAI_API_KEY="..."
```

Then in YAML:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
```

You can also set URSA configuration options directly:

```bash
URSA_LLM_MODEL__MODEL=openai:gpt-5.4 ursa
```

Use `ursa --help` to view supported `URSA_...` variables.

## Environment interpolation in config files

URSA config loading supports environment interpolation in YAML values. For MCP server environment blocks, this is useful for passing secrets to subprocesses:

```yaml
mcp_servers:
  example:
    transport: stdio
    command: example-server
    env:
      API_KEY: ${EXAMPLE_API_KEY}
      OPTIONAL_SETTING: ${OPTIONAL_SETTING:default-value}
```

## Inspect the active configuration

```bash
ursa --print-config
```
