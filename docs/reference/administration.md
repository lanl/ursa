# Administration

This guide covers organization-wide URSA configuration and deployment for
administrators and IT teams.

## Customize installation and setup

An organization can provide a site-owned installer that preloads non-secret
defaults into the guided setup wizard. Users still enter credentials, validate
connections and models, and approve the final configuration. The site-owned
script wraps the [official URSA installer](https://lanl.github.io/ursa/install.sh).

### Installer defaults

The site installer passes a JSON object to the official installer with
`--config-stdin`. The object supports these fields:

| Field | Value | Default | Purpose |
| --- | --- | --- | --- |
| `python_version` | String | `"3.13"` | Python version for URSA. |
| `extras` | Array of strings | `["dashboard"]` | URSA extras to install. |
| `extra_packages` | Array of requirement strings | `[]` | Additional packages to install. |
| `inference_providers` | Array of `{name, base_url}` objects | `[]` | Provider defaults shown in setup. |
| `mcp_servers` | Object keyed by server name | `{}` | MCP server defaults shown in setup. |

Provider defaults contain only `name` and `base_url`. MCP defaults use the
normal [MCP server configuration][mcp-server-configuration] shape, excluding
credentials. To prompt for an MCP bearer token, provide its formatting template
without a value:

```json
{
  "headers": {
    "Authorization": {"template": "Bearer %s"}
  }
}
```

Do not put API keys, tokens, passwords, keyring references, or environment
secret references in the payload. Setup collects credentials and stores them in
the operating-system keyring. Existing provider and MCP entries in
`~/.config/ursa/config.yaml` take precedence over site defaults.

See [Models and inference providers][models-and-inference-providers] and
[Secrets][secrets] for the corresponding configuration formats.

### Site-owned installer

Customize the URLs, packages, providers, and MCP servers in this example, then
publish it as your organization's `install.sh`:

```sh
#!/bin/sh

set -eu

URSA_INSTALLER_URL=https://lanl.github.io/ursa/install.sh
installer=$(mktemp "${TMPDIR:-/tmp}/ursa-install.XXXXXX")

cleanup() {
    rm -f "$installer"
}
trap cleanup EXIT
trap 'exit 1' HUP INT TERM

curl -LsSf "$URSA_INSTALLER_URL" -o "$installer"

cat <<'JSON' | sh "$installer" --config-stdin
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
```

Users can then install URSA from the organization's URL:

```bash
curl -LsSf https://software.example.com/ursa/install.sh | sh
```

Serve the script over trusted HTTPS, review it like any other deployment
artifact, and keep secrets out of scripts, URLs, package requirements, and
configuration-management logs.
