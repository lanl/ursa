# URSA CLI Guide

This guide introduces the `ursa` command-line interface for new users. It covers the main CLI entry points and the newer commands for managing agent groups and agent directories.

## Overview

URSA currently exposes two primary CLI entry points:

- `ursa`  
  Launches the main CLI interface.
- `ursa mcp-server`  
  Launches URSA agents as tools in an MCP server.

The CLI also supports configuration options such as `--config`, `--log-level`, `--print-config`, and `--name`, along with newer commands for organizing agents into groups and managing saved agent directories.

## Basic usage

### Start the main URSA CLI

```bash
ursa
```

This launches the default URSA interactive CLI interface.

### Start with a config file

```bash
ursa --config my_config.yaml
```

This loads configuration from a YAML or JSON file.

### Start with a workspace and agent name

```bash
ursa --workspace docs --name testbot
```

Use `--name` to set the user-facing name of the agent. Internally, URSA still uses its existing `agent_name` configuration field, but the CLI now consistently exposes `--name`.

### Set log level

```bash
ursa --log-level info
```

Supported log levels are:

- `debug`
- `info`
- `notice`
- `warning`
- `error`
- `critical`

### Print the resolved config

```bash
ursa --print-config
```

This prints the effective URSA configuration and exits.

## Run URSA as an MCP server

```bash
ursa mcp-server
```

This runs URSA as an MCP server.

Example:

```bash
ursa mcp-server --transport stdio
```

Additional MCP server options include host, port, transport, and log level.

## Naming convention

The CLI now consistently uses `--name` for user-facing agent naming.

Examples:

```bash
ursa --name testbot
ursa show-agent --name testbot --group default
ursa save-agent --name testbot --group default
ursa copy-agent --name testbot_new --from testbot
```

## Agent groups

URSA stores agent groups under:

```text
~/.cache/ursa_agents/
```

There should typically be a default group:

```text
~/.cache/ursa_agents/default
```

Groups provide a simple way to separate sets of agents. This is useful for organizing work by project, security boundary, or other workflow needs.

### List groups

```bash
ursa list-groups
```

Lists directory names under `~/.cache/ursa_agents/`.

### Create a group

```bash
ursa create-group research allowed_urls.yaml
```

This creates:

```text
~/.cache/ursa_agents/research
```

and copies the config file into that directory.

### Group config format

The group config file must be YAML and must contain `allowed_base_urls` as a non-empty list of strings.

Example:

```yaml
allowed_base_urls:
  - https://base_url1.com
  - https://base_url2.com
```

### Show a group

```bash
ursa show-group research
```

Displays the group path and the current contents of the group directory.

### Update a group

```bash
ursa update-group research updated_urls.yaml
```

Validates the YAML file and copies it into the existing group directory.

### Delete a group

```bash
ursa delete-group research
```

Deletes the group directory.

Notes:

- The `default` group cannot be deleted.
- Deleting a group removes its contents.

## Agent management

URSA also provides CLI commands for working with agent directories inside groups.

At present, this is a filesystem-based model:

- each group is a directory
- each agent is a directory inside a group
- saved checkpoints are timestamped copies of agent directories

This provides a simple starting point for organizing and preserving agent state.

## List agents

```bash
ursa list-agents --group default
```

Lists the agent directories inside the specified group.

## Show an agent

```bash
ursa show-agent --name testbot --group default
```

Displays:

- agent name
- group name
- agent path
- directory contents

## Save an agent checkpoint

```bash
ursa save-agent --name testbot --group default
```

Creates a timestamped copy of the agent directory.

For example, if the agent is named `testbot`, saving it may create a copy like:

```text
testbot_20260429_101530
```

This allows you to continue working with `testbot` while preserving a checkpoint from an earlier state.

## Copy an agent

Copy an agent within the same group:

```bash
ursa copy-agent --name testbot_new --from testbot
```

Copy an agent to a different group:

```bash
ursa copy-agent --name test_bot_work --group work --from testbot --from-group default
```

This is useful for:

- branching from an earlier agent state
- moving work into a different group
- creating a new working copy without modifying the original

## Example workflow

### 1. List existing groups

```bash
ursa list-groups
```

### 2. Create a new group

```bash
ursa create-group work allowed_urls.yaml
```

### 3. List agents in that group

```bash
ursa list-agents --group work
```

### 4. Start URSA with a named agent

```bash
ursa --workspace . --name testbot --group work
```

then do work with the agent through prompting.

### 5. Save an existing agent

```bash
ursa save-agent --name testbot --group work
```

### 6. Copy the saved state into a new working agent

```bash
ursa copy-agent --name testbot_experiment --from testbot_20260429_101530 --from-group work --group work
```

## Tips

- Use groups to separate different categories of work.
- Use `save-agent` before major changes if you want a timestamped checkpoint.
- Use `copy-agent` to branch from a previous agent directory without overwriting it.
- Use `show-group` and `show-agent` to inspect what is currently stored on disk.

## Summary of commands

### Core commands

```bash
ursa
ursa --config my_config.yaml
ursa --workspace . --name testbot
ursa --print-config
ursa mcp-server
```

### Group commands

```bash
ursa list-groups
ursa create-group <group-name> <config_file.yaml>
ursa show-group <group-name>
ursa update-group <group-name> <config_file.yaml>
ursa delete-group <group-name>
```

### Agent commands

```bash
ursa list-agents --group default
ursa show-agent --name <agent-name> --group default
ursa save-agent --name <agent-name> --group default
ursa copy-agent --name <new-agent-name> --from <source-agent-name> --group default --from-group default
```

## Notes for future evolution

The current agent-management commands operate on the present directory-based layout. This is intended as a practical starting point for management and checkpointing. As the CLI evolves, these capabilities may grow into a more structured system for versioning, checkpointing, and reusing agent states.
