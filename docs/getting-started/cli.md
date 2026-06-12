# Getting Started - CLI

This guide walks through starting URSA from the terminal, configuring a model, chatting with the default assistant, and running the planning and execution agents.

## Prerequisites

- URSA is installed. See [Installation](../installation/index.md).
- You have access to an LLM endpoint.
- You have a dedicated workspace directory for files URSA may create or modify.

!!! warning "Use a dedicated workspace"
    The execution agent can write files and run shell commands. Start in a directory created for URSA work, not in a source tree or data directory you cannot risk modifying.

## 1. Create a configuration file

YAML configuration files are the recommended way to configure URSA because they are reusable and easy to edit.

Create `config.yaml`:

```yaml
llm_model:
  model: openai:gpt-5.4
  api_key_env: OPENAI_API_KEY

workspace: ./ursa-workspace
```

Then set your API key in the shell:

=== "macOS/Linux"

    ```bash
    export OPENAI_API_KEY="..."
    ```

=== "Windows PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "..."
    ```

See [Configuration](../configuration/index.md) for Ollama, Anthropic, Google GenAI, and custom OpenAI-compatible endpoints.

## 2. Start URSA

```bash
ursa --config config.yaml
```

You should see the URSA prompt:

```text
ursa>
```

Type `help` or `?` inside the prompt to see available interactive commands.

## 3. Chat with the assistant

```text
ursa> Summarize what URSA can help me do.
```

Plain text input is handled by the default chat behavior.

## 4. Use the planning agent

Run the planning agent with the `plan` command:

```text
ursa> plan Write a plan for analyzing a CSV file with pandas and plotting summary statistics.
```

You can also type the agent name first and provide the prompt interactively:

```text
ursa> plan
plan: Write a plan for analyzing a CSV file with pandas and plotting summary statistics.
```

## 5. Use the execution agent

The execution agent can write files and run commands in the configured workspace.

```text
ursa> execute Write and run a Python script that prints the first 10 prime numbers.
```

Review the actions and outputs carefully. For more safety guidance, see [Sandboxing and information control](../best-practices/sandboxing.md).

## 6. Optional: use a named agent

A named agent stores state so you can return to it later:

```bash
ursa --config config.yaml --name my-first-agent
```

For detailed commands to list, save, copy, share, import, and delete agents, see [Persistence](../persistence/index.md).

## Useful CLI commands

```bash
ursa --help
ursa --print-config
ursa --config config.yaml
ursa --config config.yaml --name my-agent
ursa --config config.yaml --use-web
```

Web tools are opt in. Use `--use-web` or `use_web: true` only when you want URSA to make network requests through its web-search tools.

## Where next?

- [Configure model endpoints](../configuration/index.md)
- [Learn about persistence](../persistence/index.md)
- [Use URSA from Python scripts](python-scripts.md)
- [Run URSA as an MCP server](mcp-server.md)
