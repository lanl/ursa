# Sandboxing and information control

URSA is designed for powerful scientific workflows. Some agents can write files, run shell commands, read local files, call web tools, and connect to external MCP servers. This power requires careful information-control practices.

## Use a dedicated workspace

Run URSA in a workspace created for the task:

```bash
mkdir ursa-workspace
cd ursa-workspace
ursa --config ../config.yaml
```

Avoid running execution workflows directly inside repositories, home directories, credential directories, or folders containing sensitive data unless that is intentional.

## Understand execution risk

The Execution Agent can:

- create and edit files,
- run shell commands,
- install or import packages if asked,
- inspect outputs and continue iterating.

Potential risks include:

- modifying or deleting files,
- leaking data through prompts, network calls, or logs,
- running expensive or long-lived processes,
- executing unsafe generated code.

Use a virtual machine, container, dedicated user account, or isolated environment for high-risk tasks.

## Web tools are opt in

URSA web tools are opt in for information-control reasons. They are disabled by default in core CLI configuration.

Enable them only when you want URSA to make network requests through web-search tools:

```yaml
use_web: true
```

or:

```bash
ursa --config config.yaml --use-web
```

When web tools are enabled, URSA may query search services or retrieve web pages as part of an agent workflow. Do not enable web tools for data that should remain fully local.

## Use groups to restrict endpoints

Security groups can restrict which model endpoint base URLs are allowed:

```yaml
allowed_base_urls:
  - https://api.openai.com
  - http://localhost:11434
```

Create and use a group:

```bash
ursa create-group research allowed_urls.yaml
ursa --config config.yaml --group research --name research-agent
```

Use narrow allowlists for sensitive projects. See [Groups and endpoint security](../persistence/groups-and-security.md).

## Be careful with MCP servers

MCP servers can expose powerful external tools. Only connect MCP servers that you trust, and do not run the URSA MCP server as a shared multi-user service without adding isolation.

## Protect secrets

- Prefer `api_key_env` rather than literal API keys in YAML files.
- In the desktop dashboard, prefer **Secure system storage** so keys are kept
  in the operating system credential manager instead of configuration files.
- Do not store API keys in persistent agent names or prompts.
- Review shared agent archives before distributing them.
- Avoid committing project-specific config files that contain endpoint secrets.

Dashboard-stored model credentials are bound to their provider or endpoint
origin. Changing the endpoint host requires saving the key again. The dashboard
passes a key to its worker through a one-time private pipe and removes
credential-like variables from the worker environment so agent-launched
commands do not inherit model API keys.

## Containerization options

Containers can provide partial isolation. The exact command depends on your environment, but the general pattern is:

1. mount only the workspace URSA needs,
2. pass only the environment variables URSA needs,
3. avoid mounting your full home directory,
4. run as a non-root user where practical.

The repository README includes Docker and Charliecloud examples for containerized execution.

## Responsible-use checklist

Before running an execution or web-enabled workflow, ask:

- Am I in a dedicated workspace?
- Does this task need web access?
- Does this task need access to sensitive files?
- Is the model endpoint approved for this data?
- Should I use a local endpoint or isolated container?
- Am I comfortable sharing any generated persistent state?
