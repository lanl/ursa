# URSA: Universal Research and Scientific Agent

URSA is a flexible agentic workflow for accelerating scientific tasks. It helps connect language models, research tools, code execution, planning, persistent agent state, retrieval-augmented generation (RAG), and Model Context Protocol (MCP) servers into reusable scientific workflows.

Use URSA when you want to:

- chat with a research assistant in the terminal or dashboard,
- ask an agent to plan a technical task,
- ask an execution agent to write, edit, and run code in a workspace,
- connect to OpenAI, Anthropic, Google GenAI, Ollama, or an OpenAI-compatible endpoint,
- persist named agents and reuse them across sessions,
- ingest documents into persistent RAG collections,
- expose URSA as an MCP server for another client or agent framework,
- compose multiple agents into teams or symposium-style peer review environments.

!!! warning "Execution and network access"
    Some URSA agents can write files, run shell commands, and use web or MCP tools. Use a dedicated workspace, review actions carefully, and read the [Sandboxing and information-control guidance](best-practices/sandboxing.md) before running high-trust or data-sensitive workflows.

## Quick install

We recommend installing with [`uv`](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install ursa-ai
```

If you prefer `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ursa-ai
```

For the web dashboard, install the dashboard extra:

```bash
uv pip install "ursa-ai[dashboard]"
# or
python -m pip install "ursa-ai[dashboard]"
```

See the [Installation](installation/index.md) section for platform-specific details.

## Quick first run

Create a reusable configuration file, for example `config.yaml`:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY
```

Then run:

```bash
ursa --config config.yaml
```

Inside the URSA prompt, type `help` or try:

```text
ursa> Summarize what URSA can help me do.
ursa> execute Write and run a Python script that prints the first 10 prime numbers.
```

## Where to go next

- [Install URSA](installation/index.md)
- [Get started with the CLI](getting-started/cli.md)
- [Get started with Python scripts](getting-started/python-scripts.md)
- [Configure model endpoints](configuration/index.md)
- [Use named agents and persistence](persistence/index.md)
- [Run URSA as an MCP server](getting-started/mcp-server.md)
- [Compose agents with environments](environments/index.md)
- [Review sandboxing and information-control guidance](best-practices/sandboxing.md)
