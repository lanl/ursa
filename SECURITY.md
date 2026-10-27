# Security and execution safety

URSA agents can read and write files, execute generated code and shell commands,
and install packages with the permissions of the process running URSA. Package
installation may occur while an agent resolves missing dependencies. The
configured workspace is not a security sandbox, and LLM-based command screening
does not guarantee safe execution.

We recommend using an isolated environment, such as a suitably configured container or virtual
machine, when running workflows you do not fully trust. Limit access to sensitive
files and credentials.
See the [sandboxing documentation](https://lanl.github.io/ursa/latest/best-practices/sandboxing/)
for guidance on workspaces, containers, model endpoints, and MCP servers.
