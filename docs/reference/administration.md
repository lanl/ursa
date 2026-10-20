# Managed setup defaults

Organizations can preload non-secret provider, model, and MCP choices into the
guided setup. Existing entries in the user's URSA configuration take
precedence, and the user still reviews and validates every connection before
saving.

The JSON object accepts these setup fields:

| Field | Value |
| --- | --- |
| `inference_providers` | Provider configuration objects with an additional `name` field. |
| `llm_model` | A default chat model configuration. |
| `emb_model` | A default embedding model configuration. |
| `mcp_servers` | MCP configurations keyed by server name. |

Do not put API keys, tokens, or passwords in the payload. Named keyring and
environment references are safe because they identify where setup should find
a secret without containing the secret itself. A template without a source
prompts the user for a new keyring value:

```json
{
  "inference_providers": [
    {
      "name": "company",
      "base_url": "https://models.example.com/v1"
    }
  ],
  "llm_model": {
    "model": "gpt-5.4",
    "model_provider": "openai",
    "inference_provider": "company"
  },
  "mcp_servers": {
    "company_tools": {
      "transport": "streamable_http",
      "url": "https://tools.example.com/mcp",
      "headers": {
        "Authorization": {"template": "Bearer %s"}
      }
    }
  }
}
```

Download the installer before passing a payload; standard input is already
occupied when the installer itself is piped into a shell.

=== "Linux/MacOS"

    ```bash
    curl -LsSf https://lanl.github.io/ursa/latest/install.sh -o install.sh
    sh install.sh --config defaults.json
    ```

    Use `--config-json` for an inline object or `--config-stdin` to read the
    object from standard input.

=== "Windows"

    ```powershell
    irm https://lanl.github.io/ursa/latest/install.ps1 -OutFile install.ps1
    .\install.ps1 -Config defaults.json
    ```

    Use `-ConfigJson` for an inline object.

The installer still installs the published `ursa-ai` package; setup defaults
do not change its Python, dependencies, source, or version. Legacy
`python_version`, `extras`, and `extra_packages` fields are accepted so existing
payloads keep working, but they no longer alter the installation.
