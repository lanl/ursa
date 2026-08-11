# Configuration files, CLI flags, and environment variables

URSA supports layered configuration. Higher-precedence sources are merged on
top of lower-precedence sources to produce a single merged configuration, which
is then resolved into the effective runtime configuration. Prefer YAML files
for reusable settings, CLI flags for temporary overrides, and environment
variables for secrets or automation.

## Configuration precedence

URSA loads configuration in this order, with later sources overriding earlier ones:

1. XDG-compliant user config, typically `~/.config/ursa/config.yaml`
2. The file passed to `--config`
3. Environment variables
4. CLI flags

Higher-precedence sources only override settings they specify. Set a nullable
setting to `null` to clear it.

See the [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir-spec/latest/)
for background on XDG-compliant config locations.

## YAML files: preferred

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
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

By default, `--print-config` prints the full resolved final configuration,
including default and null values:

```bash
ursa --print-config
ursa --print-config=resolved
```

To see the full configuration before final resolution:

```bash
ursa --print-config=merged
```

Without a suffix, a level selects only that source:

```bash
ursa --print-config=user,merged
ursa --config ./.ursa/config.yaml --print-config=file,resolved
```

Levels are:

- `user`: XDG configuration files
- `file`: only the file passed with `--config`
- `final`: all files plus environment and CLI overrides

Add `+` to include lower-precedence sources. For example, use
`ursa --config ./.ursa/config.yaml --print-config=file+,resolved` when an
explicit file selects a provider defined in the user config. Stages are
`merged` and `resolved`.

Resolved output shows the complete effective configuration. Merged output shows
the complete configuration before provider inheritance and other resolution
steps.
