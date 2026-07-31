# Named agents

A named agent stores durable state under a group. This lets you return to work later, branch from saved state, or share/import agent state.

## Start a named agent

```bash
ursa --config config.yaml --name catalyst-assistant --group chemistry
```

If you omit `--group`, URSA uses the default group.

## List agents

```bash
ursa list-agents --group default
```

## Show agent details

```bash
ursa show-agent --name catalyst-assistant --group chemistry
```

## Continue work with an agent

```bash
ursa --config config.yaml --name catalyst-assistant --group chemistry
```

## Naming guidance

Use names that describe the project or purpose:

```text
literature-review
catalyst-assistant
plan-execute-demo
proposal-helper
```

Avoid putting secrets, credentials, or sensitive details in agent names.
