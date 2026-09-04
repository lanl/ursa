# Secrets

Avoid storing API keys and other credentials directly in configuration files.
URSA can read secrets from environment variables or from the operating system's
credential store through the Python [keyring package](https://pypi.org/project/keyring/).

## Store a secret in the keyring

Reference a keyring entry from an inference provider:

```yaml
inference_providers:
  openai:
    api_key:
      keyring: true
```

Then store its value with URSA:

```bash
ursa auth login openai
```

When `keyring: true` is used for an inference provider, its name is also used
as the keyring username. A string can select another username instead:

```yaml
api_key:
  keyring: shared-openai-key
```

Environment-backed secrets use the same reference form:

```yaml
api_key:
  env: OPENAI_API_KEY
```

URSA resolves the reference only when the credential is needed. Resolved
values remain masked in Pydantic models and configuration output.

## Format a secret with `SecretTemplate`

MCP headers often need a scheme or another prefix around the credential. Add a
`template` containing `%s`, which URSA replaces with the resolved secret:

```yaml
mcp_servers:
  remote-tools:
    transport: streamable-http
    url: https://tools.example.com/mcp
    headers:
      Authorization:
        keyring: true
        template: "Bearer %s"
```

Here, `keyring: true` uses `remote-tools`, the MCP server name, as the keyring
username.

The `ursa auth login` and `ursa auth list` commands are useful for populating
and checking the secrets referenced by a configuration.
