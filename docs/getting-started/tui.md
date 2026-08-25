# Getting Started - TUI

This guide walks through URSA's terminal user interface (TUI): starting it with
the `ursa` CLI command, chatting with the default assistant, and running the
planning and execution agents.

## Prerequisites

- URSA is installed. See [Getting started][getting-started].
- You have access to an LLM endpoint.
- You have a dedicated workspace directory for files URSA may create or modify.

!!! warning "Be aware of your workspace"
    The execution agent can write files and run shell commands. Be careful using workspaces with source tree or data directory you cannot risk modifying. Good practice is to make backups or copies of directories before working.

## 1. Start with the default configuration

OpenAI works without a config file. Set the API key in your shell:

=== "macOS/Linux"

    ```bash
    export OPENAI_API_KEY="..."
    ```

=== "Windows PowerShell"

    ```powershell
    $env:OPENAI_API_KEY = "..."
    ```

The built-in `openai` provider supplies the model and base URL. Use a persistent
[user configuration][configuration-files-cli-flags-and-environment-variables]
only when you need to change a default or select another provider.

## 2. Start URSA

```bash
ursa
```

You should see the full-screen URSA interface. Type `/` to browse app
commands, `#` to route a message to an agent, or `@` to insert a workspace
path.

### Full-screen interface controls

| Input | Action |
|---|---|
| `/` | Browse application commands. Use `/keymap` for every keyboard shortcut. |
| `#` | Choose an agent and route the message to it. |
| `@` | Insert a workspace file or directory into the message. |
| **Enter** | Submit the message. |
| **Shift+Enter** or **Ctrl+J** | Insert a newline. Some terminals cannot distinguish Shift+Enter, so Ctrl+J is the portable option. |
| **Ctrl+Q** or `/exit` | Exit gracefully, waiting for an active turn to finish. |
| **Ctrl+D** | Exit immediately without cleanup; reserve this for a stuck turn. |

## 3. Chat with the assistant

```text
Summarize what URSA can help me do.
```

Plain text input is handled by the default chat behavior.

## 4. Use the planning agent

Run the planning agent with the `#plan` macro. Typing `#` opens the agent
picker and inserts the selected behavior at the front of the prompt:

```text
#plan Write a plan for building a suite of surrogate models on data.csv and performing assessment of predictive capability and uncertainty quantification.
```

The leading `#` is required; `plan ...` without it is ordinary chat input.

## 5. Use the execution agent

The execution agent can write files and run commands in the configured workspace.

```text
#execute Write and run a Python script that prints the first 10 prime numbers.
```

Review the actions and outputs carefully. For more safety guidance, see
[Sandboxing and information control][sandboxing-and-information-control].

## 6. Optional: use a named agent

A named agent stores state so you can return to it later:

```bash
ursa --name my-first-agent
```

For detailed commands to list, save, copy, share, import, and delete agents, see [Persistence](../persistence/index.md).

## Useful `ursa` CLI commands

```bash
ursa
ursa --help
ursa --print-config
ursa --name my-agent
ursa --use-web
```

Web tools are opt in. Use `--use-web` or `use_web: true` only when you want URSA
to make network requests through its web-search tools.

## Where next?

- [Configure model endpoints](../configuration/index.md)
- [Learn about persistence](../persistence/index.md)
- [Use URSA from Python scripts][getting-started-python-scripts]
- [Run URSA as an MCP server][getting-started-mcp-server]
