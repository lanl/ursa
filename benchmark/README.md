# Startup benchmarks with ASV

```sh
uv sync --locked --group dev
uv run --no-sync asv machine --yes  # Once per machine
just benchmark
# Or: uv run --no-sync asv run --environment existing --show-stderr
```

ASV runs `startup.Startup.track_max_ready` for nine scenarios: default, named,
provider-prefixed model, stdio MCP, SSE MCP, streamable HTTP MCP, all three MCP
transports together, help, and version. Each result is the **maximum of five
fresh-process startup measurements**, in seconds. The target is **less than
1.0 second for every scenario**. Raw samples are included in ASV's captured
stderr (`--show-stderr`). No warmup samples are discarded.

```sh
# Run just the mixed MCP configuration:
just benchmark --bench 'startup.Startup.track_max_ready.*mcp-mixed'

# Check benchmark discovery and syntax:
uv run --no-sync asv check --environment existing

# Record the currently installed checkout under its commit, then build reports:
# Do this on a clean checkout so the commit label describes the measured code.
just benchmark --set-commit-hash HEAD
uv run --no-sync asv publish
uv run --no-sync asv preview
```

`asv.conf.json` uses the existing uv environment so development runs include
uncommitted source and the dependencies from `uv.lock`. Ordinary existing-env
runs print results but do not save historical measurements. ASV manages saved
results and reports under `.asv/`, which is ignored by Git. Its exit status
reports benchmark execution failures; inspect the recorded maximum against the
one-second target. ASV does not automatically enforce an absolute time limit.

## Measurement boundary

ASV's `track_` benchmark records the probe's measured duration rather than timing
its entire subprocess lifetime. A normal `time_` benchmark would also include
shutdown and cleanup, while `timeraw_` starts its timer after interpreter startup.
The small standard-library helper in `_startup.py` exists to measure exactly the
boundary we care about; ASV owns benchmark discovery, execution, and reporting.

Timing starts before spawning the fresh child and stops after Textual's first
screen refresh. The probe checks that the real `PromptArea` is focused, enabled,
and accepts text. POSIX runs use a 100-column, 35-row pseudo-terminal; Windows
uses Textual's headless driver at the same dimensions. Both paths exercise the
real parser, configuration loading, runtime, widgets, and graceful cleanup.
The runtime is never mocked. Help/version timing runs through successful exit
with nonempty output.

Each child uses a temporary home, cache, configuration, and workspace. Named
sessions start without saved history. The provider scenario parses an Ollama
model prefix without making a request. MCP scenarios load real configuration
files, including an environment-backed authorization header with a test token.
Their commands and endpoints are deliberately unusable: startup must reach the
prompt without starting a server or contacting an endpoint. Regression tests
also verify that the MCP SDK and agent implementations stay unloaded.

These are fresh-process measurements with warm filesystem caches. Installation,
uv environment checks, terminal emulator paint latency, and first model-response
or MCP connection time are excluded. Python startup and probe imports are
included. Run on an otherwise idle machine: the maximum deliberately exposes
scheduler and filesystem delays on shared hosts instead of hiding outliers.

## Example results

On Linux aarch64 with Python 3.11.16, the locked development environment, and
the PTY driver, the startup changes based on `origin/main` at `8270b21f`
measured the following maxima across five fresh processes per scenario:

| Scenario | Maximum |
| --- | ---: |
| Default prompt | 0.359 s |
| Named prompt | 0.378 s |
| Provider-prefixed model | 0.464 s |
| Stdio MCP | 0.361 s |
| SSE MCP | 0.368 s |
| Streamable HTTP MCP | 0.365 s |
| All three MCP transports | 0.361 s |
| Help | 0.222 s |
| Version | 0.235 s |

All 45 samples were below one second. These are measurements of the modified
checkout on one machine, not the unmodified base commit or a performance
guarantee for other machines.

## Where loading is delayed

The runtime registers agent class names and loads implementations when metadata
or execution is requested. `has_extra(extra, package="ursa")` checks declared
optional dependencies through distribution metadata and `find_spec`, without
importing them. Model factories, MCP clients, checkpoint backends, RAG tools,
and callback rendering load at their use sites. Lightweight MCP configuration
models preserve validation without loading the SDK. Configuration and endpoint
policy validation still happen at startup. Model construction, provider
connectivity checks, and MCP credential resolution happen on first use, before
an agent executes. Concurrent requests share that initialization, and shutdown
waits for it even if its callers are cancelled.

Dependency provenance is recorded in [DEPENDENCIES.md](DEPENDENCIES.md).
