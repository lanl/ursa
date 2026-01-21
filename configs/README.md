# Configuring URSA

> Note: The contents of this folder will be moved to the URSA documentation in the near future

The Ursa HITL can be configured via CLI flags, environmental variables or by passing a configuration file:

```shell
> ursa --llm_model.model openai:gpt-5
[...]

> URSA_LLM_MODEL__MODEL=openai:gpt-5 ursa
[...]

> ursa --config config.yaml
[...]
```

Supported environmental variables for the CLI can be viewed by running: `ursa --help`.

## Default Configuration Options

The default configuration is defined by [src/ursa/cli/config.py](../src/ursa/cli/config.py).
However, for many options (Agent config, MCP servers) the available options are documented by
their respective subsystems. An [example configuration file](./example.yaml) is provided.

To see the default configuration run: `ursa --print-config`

Configuration files can be provided in [JSON](https://www.json.org/) or [YAML](https://yaml.org/) formats.

## Configuring Agents

URSA Agents can be configured using the `agent_config` field in a configuration file.
Exact options are dependent on the agent (see their documentation) and is limited to
options that can be expressed in a YAML file.

Agent configurations are keyed by their subcommand name in the HITL. For example,
the ExecutionAgent's config is under `execute`.

The Memory agent (used by the ExecutionAgent and RecallAgent) is keyed under `memory`.

The current agent configuration (excluding defaults) can be viewed using the `agents` command
from URSA's HITL (run `ursa` then type `agents`).

## Configuring MCP Servers

MCP servers are configured using a nested dictionary mapping server names to their configurations.
The [stdio and streamable-http](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
transports are supported. The available options for both are documented at:

- [stdio](https://modelcontextprotocol.github.io/python-sdk/api/?h=stdioserverparameters#mcp.StdioServerParameters)
- [streamable-http](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/client/session_group.py#L49)
