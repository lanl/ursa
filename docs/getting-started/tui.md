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
    setx OPENAI_API_KEY "..."

    # Then start a new shell and confirm with `echo $env:OPENAI_API_KEY`
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

Use `/status` to see the context window and its 70% summarization threshold
(or the default fallback, which individual agents can override), plus the source
of the context limit. The **Config** tab shows the effective
`tokens_before_summarize`; comments preserve the configured fallback values.
Set `tokens_before_summarize` in your config or `URSA_TOKENS_BEFORE_SUMMARIZE`
to change the fallback, which defaults to 50,000 tokens.
These values refresh when you switch models. For Ollama,
the allocated window may only become available after the first response.

### Full-screen interface controls

| Input | Action |
|---|---|
| `/` | Browse application commands. Use `/keymap` for every keyboard shortcut. |
| `/subagents` | Switch the message target within the active agent hierarchy. |
| `/coordination` | Open live messages and the subagent hierarchy. |
| `#` | Choose an agent and route the message to it. |
| `@` | Insert a workspace file or directory into the message. |
| **Enter** in the editor | Submit the message. During active work, queue it for the selected agent. |
| **Enter** or **Space** on a focused card | Expand or collapse its details. Use **Shift+Tab** from the editor to reach conversation cards. |
| **Esc** | Close an open picker or dialog. Does not submit messages or cancel agent work. |
| **Shift+Enter** or **Ctrl+J** | Insert a newline. Some terminals cannot distinguish Shift+Enter, so Ctrl+J is the portable option. |
| **Ctrl+Q** or `/exit` | Exit gracefully, waiting for an active turn to finish. |
| **Ctrl+D** | Exit immediately without cleanup; reserve this for a stuck turn. |

### View managed terminals

Run `/terms` for a live, view-only browser of managed terminal sessions. It
opens the newest tab first, refreshes while open, and returns to the prompt with
**Escape** or **Q**. Ghostty tabs preserve their emulated grid and styles;
ProcessTerm tabs soft-wrap captured output. Agent tools remain responsible for
terminal input; an empty browser explains when no sessions exist.
Messages waiting for an agent appear dimmed above the prompt. They move into
the conversation only when that agent begins processing them.

Each turn groups agent messaging, spawning, waiting, and stopping into one
**Coordination** card with a summary and its two latest actions. Expand the
card to inspect individual calls. `/coordination` opens on Messages, which
displays 50 previews at a time; its Subagents tab shows the live hierarchy.
Use **Older** and **Newer** to browse, then **Latest** to follow live messages.
Scrolling back or expanding a message pauses live following so incoming traffic
does not move what you are reading. Press **End** or select **Latest** to resume.
Use **Tab** and **Shift+Tab** to move between message cards and controls; the
focused card has a visible outline. `/keymap` lists the card shortcuts.
Message bodies and coordination details are stored in SQLite and loaded when
expanded.

See [Async messages and subagents](../agents/async-messages.md) for the Python
API and hierarchy addressing rules.

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

Reopening the same named agent restores its saved TUI cards, token totals,
expanded cards, and reading position. Scrollback loads as you browse, so long
conversations do not require keeping every card on screen. **Ctrl+L** clears the
view without deleting the agent's history. Older sessions created before card
persistence have no saved TUI transcript to restore.

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

Useful commands inside the full-screen interface include:

| Command | Purpose |
| --- | --- |
| `/status` | Show runtime, model, endpoint, group, terminal backend, and MCP status. |
| `/terms` | Browse managed terminal sessions in a live, view-only modal. |
| `/keymap` | Show the complete keyboard map. |
| `/models` | Switch chat or embedding providers. |
| `/theme` | Change the interface theme. |
| `/exit` | Exit gracefully. |

Terminal activity is grouped by session. Expand its compact latest-line card
for the live styled view; repeated calls update the same card.

## Where next?

- [Configure model endpoints](../configuration/index.md)
- [Learn about persistence](../persistence/index.md)
- [Use URSA from Python scripts][getting-started-python-scripts]
- [Run URSA as an MCP server][getting-started-mcp-server]
