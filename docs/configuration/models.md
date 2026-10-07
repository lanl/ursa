# Models and inference providers

URSA initializes chat and embedding models through LangChain. Model names
normally use the form `<provider>:<model-name>`. Connection settings belong in
an `inference_providers` entry, and `llm_model` and `emb_model` select that
entry with `inference_provider`.

For the standard OpenAI service, no YAML is required. Set `OPENAI_API_KEY` and
run `ursa`; URSA's built-in `openai` provider supplies the endpoint and
defaults.

## Hosted and local model examples

=== "OpenAI (built in)"

    === "macOS/Linux"

        ```bash
        export OPENAI_API_KEY="..."
        ursa
        ```

    === "Windows PowerShell"

        ```powershell
        $env:OPENAI_API_KEY = "..."
        ursa
        ```

    To change only the model in your user config:

    ```yaml
    llm_model:
      model: openai:gpt-5.4
    ```

=== "OpenAI-compatible"

    ```yaml
    inference_providers:
      research_gateway:
        base_url: https://models.example.edu/v1
        api_key:
          env: RESEARCH_LLM_API_KEY
    llm_model:
      model: openai:my-model-name
      inference_provider: research_gateway
    ```

    Use the provider's actual model name, and put the non-secret URL directly
    in the file.

=== "Anthropic"

    ```yaml
    inference_providers:
      anthropic:
        api_key:
          env: ANTHROPIC_API_KEY
    llm_model:
      model: anthropic:claude-sonnet-4-5
      inference_provider: anthropic
    ```

=== "Google GenAI"

    ```yaml
    inference_providers:
      google:
        api_key:
          env: GOOGLE_API_KEY
    llm_model:
      model: google_genai:gemini-2.5-pro
      inference_provider: google
    ```

=== "Ollama"

    Install [Ollama](https://ollama.com/), run `ollama pull gpt-oss-20b`, and
    use:

    ```yaml
    inference_providers:
      local_ollama:
        base_url: http://localhost:11434
    llm_model:
      model: ollama:gpt-oss-20b
      inference_provider: local_ollama
    emb_model:
      model: ollama:nomic-embed-text:latest
      inference_provider: local_ollama
    ```

=== "Azure OpenAI"

    ```yaml
    inference_providers:
      azure:
        base_url: https://your-resource.openai.azure.com/
        api_key:
          env: AZURE_OPENAI_API_KEY
    llm_model:
      model: azure_openai:deployment-name
      inference_provider: azure
    ```

    Azure deployments can require additional provider-specific model fields.
    URSA passes extra model settings through to the LangChain integration.

## Use one provider for chat and embeddings

Models inherit endpoint and credential values from the selected provider. They
can share one provider while using different model names:

```yaml
inference_providers:
  lab:
    base_url: https://models.example.edu/v1
    api_key:
      env: LAB_LLM_API_KEY
llm_model:
  model: openai:chat-model
  inference_provider: lab
emb_model:
  model: openai:embedding-model
  inference_provider: lab
```

A value set directly on a model overrides the provider value. Set a nullable
model field to `null` to clear an inherited value.

## Temporary CLI overrides

Configuration files are preferable for reusable endpoint settings, but you can
also override any model field for a single run. For example:

=== "macOS/Linux"

    ```bash
    ursa \
      --llm_model.model openai:my-model-name \
      --llm_model.inference_provider research_gateway
    ```

=== "Windows PowerShell"

    ```powershell
    ursa `
      --llm_model.model openai:my-model-name `
      --llm_model.inference_provider research_gateway
    ```

The referenced `research_gateway` still comes from a loaded system, user, or
explicit config file. See [Files, CLI flags, and environment
variables][configuration-files-cli-flags-and-environment-variables] for the
complete precedence order.

## TLS verification

URSA verifies TLS certificates by default and loads the operating system trust
store. For a temporary test endpoint only, you can disable verification on the
provider:

```yaml
inference_providers:
  test_endpoint:
    base_url: https://test-model.example/v1
    ssl_verify: false
```

Disabling verification exposes credentials and traffic to interception. Install
the correct certificate authority instead whenever possible.

## Install additional integrations

URSA includes `langchain-openai`, `langchain-anthropic`,
`langchain-google-genai`, and `langchain-ollama`. Other model integrations use
the corresponding `langchain-*` package. LangGraph extensions, such as durable
checkpoint backends, use `langgraph-*` packages.

=== "ursa self"

    When URSA is installed with `uv tool install ursa-ai`, you can use
    `ursa self modify` to add or remove additional packages from your URSA
    installation:

    ```bash
    ursa self modify --with langgraph-checkpoint-postgres
    ```

    For an additional model provider, replace the `langgraph-*` package with
    its integration package, for example `--with langchain-groq`.

    You can also enable URSA extras with the `--extra` flag. For example, to
    enable the LAMMPS agent:

    ```bash
    ursa self modify --extra lammps
    ```

=== "uv tool installation"

    Recreate URSA's isolated tool environment and add the required package with
    `--with`. This example adds PostgreSQL checkpoint support:

    === "macOS/Linux"

        ```bash
        uv tool install --force \
          --python 3.13 \
          --with langgraph-checkpoint-postgres \
          'ursa[dashboard]'
        ```

    === "Windows PowerShell"

        ```powershell
        uv tool install --force `
          --python 3.13 `
          --with langgraph-checkpoint-postgres `
          'ursa[dashboard]'
        ```

    For an additional model provider, replace the `langgraph-*` package with
    its integration package, for example `--with langchain-groq`.

=== "uv virtual environment — macOS/Linux"

    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install 'ursa-ai[dashboard]' langgraph-checkpoint-postgres
    ```

=== "uv virtual environment — Windows PowerShell"

    ```powershell
    uv venv
    .\.venv\Scripts\Activate.ps1
    uv pip install 'ursa-ai[dashboard]' langgraph-checkpoint-postgres
    ```

Packages must be installed in the same environment as URSA. Installing them in
an unrelated project environment will not make them available to a `uv tool`
installation.

## Local-model caveats

Local models vary in tool-calling support, context length, instruction
following, and their ability to recover from execution errors. For
execution-heavy workflows, choose a model with reliable tool calling and test
it first in a disposable workspace.

## Endpoint controls

For controlled environments, combine custom endpoints with URSA groups and
allowed base URLs. See
[Groups and endpoint security](../persistence/groups-and-security.md).
