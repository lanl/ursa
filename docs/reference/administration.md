# Administration

This guide covers organization-wide URSA configuration and deployment for
administrators and IT teams.

## Customize installation and setup

An organization can provide a site-owned installer that preloads non-secret
defaults into the guided setup wizard. Users still enter credentials, validate
connections and models, and approve the final configuration. The site-owned
script wraps the [official URSA installer](https://lanl.github.io/ursa/latest/install.sh).

### Installer defaults

The site installer writes a JSON object and passes its path to the official
installer with `--config`. The object supports these fields:

| Field | Value | Default | Purpose |
| --- | --- | --- | --- |
| `python_version` | String | `"3.13"` | Python version for URSA. |
| `extras` | Array of strings | `["dashboard"]` | URSA extras to install. |
| `extra_packages` | Array of requirement strings | `[]` | Additional packages to install. |
| `inference_providers` | Array of provider objects with `name` | `[]` | Provider defaults shown in setup. |
| `llm_model` | Model configuration object | None | Default chat model shown in setup. |
| `emb_model` | Model configuration object | None | Default embedding model shown in setup. |
| `mcp_servers` | Object keyed by server name | `{}` | MCP server defaults shown in setup. |

Provider defaults use the normal inference-provider configuration with an added
`name` field. MCP defaults use the normal
[MCP server configuration][mcp-server-configuration] shape. To prompt for a
bearer token using a keyring entry named after the provider or MCP server,
provide its formatting template without a value:

```json
{
  "headers": {
    "Authorization": {"template": "Bearer %s"}
  }
}
```

Named keyring and environment-variable references are safe to include when a
site needs to share one credential between entries. Do not put API keys,
tokens, or passwords in the payload. Setup collects credential values and
stores them in the operating-system keyring. Model defaults use the normal
model configuration shape and must name an `inference_provider`. A model
default is ignored when that provider is not configured. Existing provider,
model, and MCP entries in `~/.config/ursa/config.yaml` take precedence over
site defaults.

See [Models and inference providers][models-and-inference-providers] and
[Secrets][secrets] for the corresponding configuration formats.

### Site-owned installer

Customize the URLs, packages, providers, and MCP servers in this example, then
publish it as your organization's `install.sh`:

```sh
#!/bin/sh

set -eu

URSA_INSTALLER_URL=https://lanl.github.io/ursa/latest/install.sh
installer=$(mktemp "${TMPDIR:-/tmp}/ursa-install.XXXXXX")
config=$(mktemp "${TMPDIR:-/tmp}/ursa-config.XXXXXX")

cleanup() {
    rm -f "$installer" "$config"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

curl -LsSf "$URSA_INSTALLER_URL" -o "$installer"

cat <<'JSON' >"$config"
{
  "python_version": "3.13",
  "extras": ["dashboard"],
  "extra_packages": [
    "company-ursa-plugin>=1"
  ],
  "inference_providers": [
    {
      "name": "company",
      "base_url": "https://models.example.com/v1"
    },
    {
      "name": "anthropic",
      "base_url": "https://api.anthropic.com"
    }
  ],
  "llm_model": {
    "model": "gpt-5.4",
    "model_provider": "openai",
    "inference_provider": "company"
  },
  "emb_model": {
    "model": "text-embedding-3-large",
    "model_provider": "openai",
    "inference_provider": "company"
  },
  "mcp_servers": {
    "company_tools": {
      "transport": "streamable_http",
      "url": "https://tools.example.com/mcp",
      "headers": {
        "Authorization": {"template": "Bearer %s"}
      },
      "timeout": 30
    }
  }
}
JSON

sh "$installer" --config "$config"
```

Users can then install URSA from the organization's URL:

```bash
curl -LsSf https://software.example.com/ursa/install.sh | sh
```

Serve the script over trusted HTTPS, review it like any other deployment
artifact, and keep secrets out of scripts, URLs, package requirements, and
configuration-management logs.
