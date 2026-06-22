# Groups and endpoint security

URSA groups organize persistent agents and provide an endpoint-security boundary. A group can restrict which model endpoint base URLs are allowed for agents in that group.

This is useful when you need to control information flow, separate projects, or prevent accidental use of an unapproved endpoint.

## Create an allowed-URL file

Create `allowed_urls.yaml`:

```yaml
allowed_base_urls:
  - https://api.openai.com
  - https://api.anthropic.com
  - http://localhost:11434
```

For a custom endpoint:

```yaml
allowed_base_urls:
  - https://my-approved-endpoint.example.com
```

## Create a group

```bash
ursa create-group research allowed_urls.yaml
```

## Use the group

```bash
ursa --config config.yaml --group research --name literature-agent
```

## Manage groups

```bash
ursa list-groups
ursa show-group research
ursa update-group research updated_allowed_urls.yaml
ursa delete-group research
```

## How this relates to model configuration

If your model config uses a custom `base_url`:

```yaml
llm_model:
  model: openai:my-model
  base_url: https://my-approved-endpoint.example.com/v1
  api_key_env: MY_ENDPOINT_API_KEY
```

then the group allowlist should include the approved base URL domain.

## Best-practice guidance

- Use separate groups for separate projects or sensitivity levels.
- Keep allowlists narrow.
- Prefer local endpoints such as Ollama for data that should not leave a machine.
- Review group policy before sharing/importing persistent agents.

See also [Sandboxing and information control](../best-practices/sandboxing.md).
