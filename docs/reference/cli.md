# CLI reference

Use the built-in help for the authoritative command list:

```bash
ursa --help
```

For subcommand-specific help:

```bash
ursa mcp-server --help
ursa exec --help
ursa rag-ingest --help
ursa rag-query --help
ursa self --help
```

## Common top-level commands

URSA configs can be
[layered with environment variables and CLI flags][configuration-files-cli-flags-and-environment-variables].

```bash
ursa --config config.yaml
ursa --print-config
ursa --config config.yaml --name my-agent --group default
ursa --config config.yaml --use-web
```

## Main subcommands

Current URSA installations include subcommands for:

- running the MCP server,
- managing groups,
- managing persistent agents,
- sharing/importing agents,
- managing persistent RAG collections,
- inspecting and managing the URSA installation,
- non-interactive execution.

Use `ursa --help` to confirm the exact set in your installed version.

## Self management

Installation status is available for every installation method:

```bash
ursa self status
```

It reports the URSA version and the running Python version and executable.
For uv tool installations, it also reports the selected URSA extras and any
additional packages installed in the tool environment.

Run the complete health check with:

```bash
ursa self doctor
ursa self doctor --json
```

Doctor reports installation status, every URSA executable on `PATH`, inference
and dependency integrity, workspace/cache writeability, provider
authentication/connectivity, chat model invocation, forced tool use, structured
output, embedding invocation, and MCP server connectivity. These are live
probes and can make billable model requests. The configuration heading
parses every discovered system, user, and explicit configuration file
independently in precedence order, then reports the final merged and resolved
configuration separately. Individual file checks are parse-only. A failed
check produces exit status 1; warnings do not. Checks are isolated, so a broken
provider, plugin, configuration, or rendering path does not prevent later
checks from running. Independent headings run concurrently and report their
wall-clock duration while retaining a stable display order. Interactive
terminals show a live spinner with completed-check progress; JSON and redirected
output remain non-interactive. Every heading has a 60-second deadline. The PATH
report preserves PATH order, warns about additional URSA copies, and reports
copies above the running URSA as shadowing errors.

Packages can add checks through the `ursa-healthcheck` entry-point group:

```toml
[project.entry-points."ursa-healthcheck"]
database = "my_package.health:check_database"
```

The callable receives a `DoctorContext` and returns a `CheckResult`. Sync and
async callables are supported. Import both public types from `ursa.cli.doctor`.
Health-check plugins must not write to stdout, which is reserved for the doctor
report and JSON document; return diagnostic text in the result message instead.
This is a plugin contract rather than an enforced sandbox restriction.

When URSA was installed with `uv tool install`, it can update itself while
preserving the installation recipe:

```bash
ursa self update
```

Use `modify` to change the recipe. Options can add extras or additional
packages, select an exact release or Git ref, and clear the existing extras
and additional packages:

```bash
ursa self modify --extra dashboard
ursa self modify --with some-package
ursa self modify --version 1.2.3
ursa self modify --ref main
ursa self modify --clean --extra fm
```

Run `ursa self modify --help` for the complete option list. For pip, Conda,
and other installations, use the same package manager that installed URSA;
`self update` and `self modify` will reject installations not managed by uv.

## Python callback API

::: ursa.cli.callbacks.HITLLogEventHandler
    options:
      show_root_heading: true
      show_source: true
