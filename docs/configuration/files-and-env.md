# Configuration files, CLI flags, and environment variables

URSA configs can be layered. Use YAML files for reusable settings, environment
variables for secrets or automation, and CLI flags for temporary overrides.

## Configuration precedence

URSA loads configuration in this order, with later sources overriding earlier ones:

1. Built-in defaults
2. The native platform system config
3. User configs: native platform location, `~/.config/ursa/config.yaml`, then
   `$XDG_CONFIG_HOME/ursa/config.yaml`
4. Environment variables
5. The file passed to `--config`
6. CLI flags

Higher-precedence sources only override settings they specify. Set a nullable
setting to `null` to clear it.

`XDG_CONFIG_HOME` is supported on every platform and only affects the user
configuration layer.

## YAML files: preferred

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key:
    env: OPENAI_API_KEY
workspace: ./ursa-workspace
group: default
use_web: false
agent_config:
  execute:
    safe_codes:
      - python
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
--llm_model.api_key.env
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
  api_key:
    env: OPENAI_API_KEY
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

`--print-config` prints the full active configuration, including defaults and
null values:

```bash
ursa --print-config
ursa --print-config=resolved
```

To inspect values before provider settings and other derived values are applied:

```bash
ursa --print-config=merged
```

You can also inspect a particular file layer:

```bash
ursa --config ./.ursa/config.yaml --print-config=file,resolved
```

The complete form is `--print-config=LEVEL[+],STAGE`. Levels are `system`,
`user`, `file`, and `final`; stages are `merged` and `resolved`. Add `+` to include
lower-precedence sources.
