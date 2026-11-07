# URSA: Universal Research and Scientific Agent

URSA is a flexible agentic workflow for accelerating scientific tasks. It helps connect language models, research tools, code execution, planning, persistent agent state, retrieval-augmented generation (RAG), and Model Context Protocol (MCP) servers into reusable scientific workflows.

Use URSA when you want to:

- chat with a research assistant in the terminal or dashboard,
- ask an agent to plan a technical task,
- ask an execution agent to write, edit, and run code in a workspace,
- connect to OpenAI, Anthropic, Google GenAI, Ollama, or an OpenAI-compatible endpoint,
- persist named agents and reuse them across sessions,
- ingest documents into persistent RAG collections,
- compose multiple agents into teams or symposium-style peer review environments.
- expose URSA as an MCP server for another client or agent framework,

!!! warning "Execution and network access"
    Some URSA agents can write files, run shell commands, and use web or MCP
    tools. Use a dedicated workspace, review actions carefully, and read the
    [Sandboxing and information-control guidance][sandboxing-and-information-control]
    before running high-trust or data-sensitive workflows.

## Quick install

We recommend installing URSA by running the following command in a terminal:

=== "Linux/MacOS"

    ```bash
    curl -LsSf https://lanl.github.io/ursa/latest/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy Bypass -c "irm https://lanl.github.io/ursa/latest/install.ps1 | iex"
    ```

The installer will install URSA using [`uv`](https://docs.astral.sh/uv/), then launch a setup wizard for URSA.

## Where to go next

- [Follow the getting started guide][getting-started]
- [Try a worked example][examples]
- [Get started with Python scripts][getting-started-python-scripts]
- [Configure model endpoints](configuration/index.md)
- [Use named agents and persistence](persistence/index.md)
- [Run URSA as an MCP server][getting-started-mcp-server]
- [Compose agents with environments][environments-agents-working-together]
- [Review sandboxing and information-control guidance][sandboxing-and-information-control]
