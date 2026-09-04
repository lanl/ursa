# Quick Reference

## Persistent agents

```bash
ursa --name my-agent
ursa --name my-agent --group my-group

ursa list-agents --group default
ursa show-agent --name my-agent --group default
ursa save-agent --name my-agent --group default
ursa copy-agent --name new-agent --from my-agent --group default
ursa delete-agent --name my-agent --group default

ursa share-agent --name my-agent --group default
ursa share-agent --name my-agent --group default --no-checkpoint
ursa import-agent ./archive.tar.gz --group default
```

## Groups

```bash
ursa list-groups
ursa create-group my-group group.yaml
ursa show-group my-group
ursa update-group my-group group.yaml
ursa delete-group my-group
```

Example `group.yaml`:

```yaml
allowed_base_urls:
  - https://models.example.org/v1
  - http://127.0.0.1:8000/v1
```

## RAG agents

```bash
ursa rag-ingest ./docs --name docs-rag
ursa rag-ingest ./paper.pdf --name paper-rag --group my-group

ursa rag-query --name docs-rag What is in the documents?
ursa rag-query --name docs-rag

ursa list-rag-agents --group default
ursa show-rag-agent docs-rag --group default
ursa save-rag-agent docs-rag --group default
ursa delete-rag-agent docs-rag --group default
```

## RAG tools

CLI:

```bash
ursa --name my-agent --rag-tools docs-rag,paper-rag
```

Python:

```python
agent = ChatAgent(
    llm=llm,
    agent_name="my-agent",
    rag_tools=["docs-rag", "paper-rag"],
)
```

Dashboard:

```text
Settings -> Agent tools -> Add RAG agents -> Save
```

## Dashboard

```bash
ursa-dashboard
ursa-dashboard --group my-group
```

Dashboard concepts:

- persistent named agents come from `~/.cache/ursa_agents/<group>/`
- sessions are dashboard conversation records
- behavior can be selected per message
- RAG tools are selected in Settings -> Agent tools
- agent management is in Settings -> Agent management

## Python examples

Chat behavior:

```python
from ursa.agents import ChatAgent

agent = ChatAgent(llm=llm, agent_name="my-agent", group="default")
```

Execution behavior:

```python
from ursa.agents import ExecutionAgent

agent = ExecutionAgent(llm=llm, agent_name="my-agent", group="default")
```

Prompt refinement:

```python
from ursa.agents import PromptingAgent

agent = PromptingAgent(llm=llm, agent_name="prompt-helper")
```
