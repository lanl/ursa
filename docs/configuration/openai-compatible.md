# OpenAI-compatible endpoints

Many hosted and self-hosted model services expose an OpenAI-compatible API. Configure these with the `openai` provider plus a custom `base_url`.

## YAML configuration

```yaml
llm_model:
  model: openai:my-model-name
  base_url: https://my-endpoint.example.com/v1
  api_key_env: MY_ENDPOINT_API_KEY
```

Run:

```bash
ursa --config config.yaml
```

## CLI override

```bash
ursa \
  --llm_model.model openai:my-model-name \
  --llm_model.base_url https://my-endpoint.example.com/v1 \
  --llm_model.api_key_env MY_ENDPOINT_API_KEY
```

## SSL verification

By default URSA verifies TLS certificates. If you are using a test endpoint with a custom certificate, you can configure:

```yaml
llm_model:
  model: openai:my-model-name
  base_url: https://my-endpoint.example.com/v1
  api_key_env: MY_ENDPOINT_API_KEY
  ssl_verify: false
```

Only disable SSL verification when you understand the risk.

## Endpoint allowlists and groups

For controlled environments, combine custom endpoints with URSA groups and allowed base URLs. See [Groups and endpoint security](../persistence/groups-and-security.md).
