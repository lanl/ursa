# Persistence quick reference

## Named agents

```bash
ursa --config config.yaml --name my-agent --group default
ursa list-agents --group default
ursa show-agent --name my-agent --group default
ursa save-agent --name my-agent --group default
ursa copy-agent --name my-agent-copy --from my-agent --group default --from-group default
ursa share-agent --name my-agent --group default
ursa import-agent path/to/agent.tar.gz --group default
ursa delete-agent --name my-agent --group default
```

## Groups

```bash
ursa list-groups
ursa create-group research allowed_urls.yaml
ursa show-group research
ursa update-group research updated_allowed_urls.yaml
ursa delete-group research
```

## RAG collections

```bash
ursa rag-ingest ./papers --name papers --group default
ursa rag-query --name papers --group default "What are the main findings?"
ursa list-rag-agents --group default
ursa show-rag-agent papers --group default
ursa save-rag-agent papers --group default
ursa delete-rag-agent papers --group default
```

## Attach RAG collections as tools

```bash
ursa --config config.yaml --name assistant --rag-tools papers
```

Multiple collections can be comma-separated:

```bash
ursa --config config.yaml --name assistant --rag-tools papers,lab-notes
```
