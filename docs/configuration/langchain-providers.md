# LangChain providers

URSA initializes chat and embedding models through LangChain. The model string usually uses:

```text
<provider>:<model-name>
```

The following provider integrations are installed with URSA's core dependencies and are common choices for URSA workflows.

## OpenAI

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
```

## Anthropic

```yaml
llm_model:
  model: anthropic:claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY
```

## Google GenAI

```yaml
llm_model:
  model: google_genai:gemini-2.5-pro
  api_key_env: GOOGLE_API_KEY
```

## Ollama

```yaml
llm_model:
  model: ollama:llama3.1
  base_url: http://localhost:11434
```

## Azure OpenAI

```yaml
llm_model:
  model: azure_openai:deployment-name
  base_url: https://your-resource.openai.azure.com/
  api_key_env: AZURE_OPENAI_API_KEY
```

Azure deployments often need provider-specific settings. LangChain accepts additional provider keyword arguments, and URSA allows extra model fields in the YAML model configuration.

## Additional provider options

URSA passes model settings to LangChain. For provider-specific options, consult the LangChain provider documentation:

- `langchain-openai`
- `langchain-anthropic`
- `langchain-google-genai`
- `langchain-ollama`

If you use a provider integration that is not installed by URSA, install the relevant LangChain package in the same environment.
