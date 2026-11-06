# CLI latency benchmarks

This is an [ASV benchmark project](https://asv.readthedocs.io/en/stable/using.html)
with `asv.conf.json` beside the `benchmarks/` Python package. Run it from the
repository root through the just recipes:

```sh
uv sync --locked --group dev
just asv machine --yes                      # Once per machine
just benchmark                             # All latency measurements
just benchmark --bench 'startup.Startup'    # Time to first prompt
just benchmark --bench 'startup.Exec'       # Time to exec response
just asv check --environment existing
```

The existing uv environment measures the current checkout, including uncommitted
changes. Results are the maximum of five fresh processes; individual samples
appear in the output. ASV reports execution failures, but does not fail a run
when a measurement exceeds the one-second target.

To save results for a **clean** checkout and view them:

```sh
just benchmark --set-commit-hash HEAD
just asv publish
just asv preview
```

Results and reports live in `benchmark/.asv/` (ignored by Git). Ordinary runs
without `--set-commit-hash` print results without saving history.

## What is measured

- **Startup:** Python process creation through the first rendered, focused,
  editable prompt. Scenarios cover default/named sessions, a model-provider
  prefix, saved provider credentials, each MCP transport, and mixed MCP
  settings. Help/version measure through exit. POSIX uses a pseudo-terminal;
  Windows uses Textual's headless driver. These measurements exclude uv.
- **Exec:** `uv run ursa exec 'say hello'` through the visible response, using a
  local HTTP model fixture. This includes uv, the real SDK, model validation,
  agent construction, tool schemas, and rendering. Samples also report time to
  the model request and process exit. Server setup is outside the measurement.

`track_` benchmarks report these precise boundaries instead of timing teardown.
Helpers use temporary configuration, credentials, and workspaces. MCP startup
fixtures cannot connect; exec uses only the local model endpoint. Neither path
mocks the runtime. External inference/MCP latency must be measured separately
with a real configuration. Run on an idle machine; filesystem caches are warm.

On Linux aarch64/Python 3.11.16, the latest five-sample batches measured a maximum
of 0.462 s to the prompt and 0.880 s to the local exec response. Exec exited in
1.018–1.119 s, versus 2.216 s in the initial diagnostic before its loading fixes.
These are observations on one machine, not portable performance guarantees.

Dependency provenance: [DEPENDENCIES.md](DEPENDENCIES.md).
