# Managed setup defaults

Organizations can preload non-secret provider, model, and MCP choices into guided setup. Existing user configuration takes precedence, and users review and validate every connection before saving.

!!! warning "Never include secrets"
    Do not put API keys, tokens, or passwords in this payload. Environment and named keyring references are safe because they only identify where setup should find a secret; a template without a source prompts for a new keyring value.

| Field | Value |
| --- | --- |
| `inference_providers` | Provider configuration objects with an additional `name` field. |
| `llm_model` | A default chat model configuration. |
| `emb_model` | A default embedding model configuration. |
| `mcp_servers` | MCP configurations keyed by server name. |

```json
{"inference_providers":[{"name":"company","base_url":"https://models.example.com/v1"}],"llm_model":{"model":"gpt-5.4","model_provider":"openai","inference_provider":"company"},"mcp_servers":{"company_tools":{"transport":"streamable_http","url":"https://tools.example.com/mcp","headers":{"Authorization":{"template":"Bearer %s"}}}}}
```

Download before passing `--config`/`-Config`: Linux/macOS: `curl -LsSf https://lanl.github.io/ursa/latest/install.sh -o install.sh && sh install.sh --config defaults.json --version 1.2.3 --python 3.12`; Windows: `irm https://lanl.github.io/ursa/latest/install.ps1 -OutFile install.ps1; .\install.ps1 -Config defaults.json -Version 1.2.3 -Python 3.12`.

Omit version to install the latest `ursa-ai`; use `--version`/`-Version` for a PyPI release or `--branch`/`-Branch` for a LANL URSA branch. Targets are passed to uv without validation. `--python`/`-Python` defaults to 3.13. Setup defaults never select a package source or version; legacy `python_version`, `extras`, and `extra_packages` fields remain accepted but do not affect installation.
