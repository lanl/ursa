# Getting Started - CLI

This guide walks through starting URSA from the terminal, chatting with the
default assistant, and routing messages to the planning and execution agents.

## Prerequisites

- URSA is installed. See [Installation](../installation/index.md).
- `OPENAI_API_KEY` is set for the default OpenAI endpoint.
- You have a dedicated workspace directory for files URSA may create or modify.

!!! warning "Be aware of your workspace"
    The execution agent can write files and run shell commands. Be careful using workspaces with source tree or data directory you cannot risk modifying. Good practice is to make backups or copies of directories before working.

For Ollama, Anthropic, Google GenAI, custom OpenAI-compatible endpoints, and
configuration files, see [Configuration](../configuration/index.md).

## 1. Start URSA

Set your OpenAI API key and launch URSA:

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

## 2. Chat with the assistant

```text
Summarize what URSA can help me do.
```

Plain text input is handled by the default chat behavior.

## 3. Route a message to the planning agent

Route a message to the planning agent with the `#plan` macro. Typing `#` opens
the agent picker and inserts the selected agent at the front of the message:

```text
#plan Write a plan for building a suite of surrogate models on data.csv and performing assessment of predictive capability and uncertainty quantification.
```

The leading `#` is required; `plan ...` without it is ordinary chat input.

## 4. Route a message to the execution agent

The execution agent can write files and run commands in the configured workspace.

```text
#execute Write and run a Python script that prints the first 10 prime numbers.
```

Review the actions and outputs carefully. For more safety guidance, see
[Sandboxing and information control][sandboxing-and-information-control].

To direct the agent to a particular workspace file, type `@` and choose it
from the path picker:

```text
#execute Read @data/measurements.csv and create a histogram of the pressure column.
```

The picker inserts the path into the message; the receiving agent decides how
to use it and must have an appropriate file tool.

## 5. Optional: use a named agent

A named agent stores state so you can return to it later:

```bash
ursa --name my-first-agent
```

For detailed commands to list, save, copy, share, import, and delete agents, see [Persistence](../persistence/index.md).

## Useful CLI commands

```bash
ursa
ursa --help
ursa --print-config
ursa --name my-agent
ursa --use-web
```

Web tools are opt in. Use `--use-web` only when you want URSA to make network
requests through its web-search tools. Configuration files and their
`use_web` setting are covered in [Configuration](../configuration/index.md).

## Where next?

- [Configure model endpoints](../configuration/index.md)
- [Learn about persistence](../persistence/index.md)
- [Use URSA from Python scripts][getting-started-python-scripts]
- [Run URSA as an MCP server][getting-started-mcp-server]
