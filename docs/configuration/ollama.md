# Ollama and local endpoints

URSA can use local models through Ollama via LangChain's `ollama` provider.

## Start Ollama

Install Ollama from the [Ollama website](https://ollama.com/) and start the service. Then pull a model:

```bash
ollama pull gpt-oss-20b 
```

## Configure URSA

Create `config.yaml`:

```yaml
llm_model:
  model: ollama:gpt-oss-20b
  base_url: http://localhost:11434

workspace: ./ursa-ollama-workspace
```

Run:

```bash
ursa --config config.yaml
```

## CLI equivalent

```bash
ursa \
  --llm_model.model ollama:gpt-oss-20b \
  --llm_model.base_url http://localhost:11434
```

## Embeddings with Ollama

If you need an embedding model, configure `emb_model`:

```yaml
emb_model:
  model: ollama:nomic-embed-text:latest
  base_url: http://localhost:11434
```

## Caveats for local models

Local models vary widely in:

- tool-calling support,
- context length,
- instruction following,
- code-generation quality,
- ability to recover from execution errors.

For execution-heavy workflows, use a model with reliable tool-calling behavior and test on a small workspace first.
