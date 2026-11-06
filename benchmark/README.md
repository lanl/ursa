# CLI latency

After `uv sync --locked --group dev`, run from the repository root:

```sh
just asv machine --yes                       # Once per machine
just asv check
just benchmark                              # Prompt, MCP prompt, exec response
just benchmark --bench 'startup.*exec'       # Filter scenarios
just benchmark --set-commit-hash HEAD        # Save results for a clean checkout
just asv publish                            # Generate .asv/html
```

Each result is the maximum of five fresh processes, including `uv run` overhead.
Startup ends after Textual's first headless refresh with an editable prompt;
exec ends when `Hello!` is flushed, before cleanup. Both use isolated configs and
a lazy `FakeListChatModel`; startup should never instantiate it. Exec retains
the real agent and tool schemas but excludes provider SDK/network latency.
The MCP case configures all three transports without connecting. Results measure
the current checkout; the one-second target is not an automatic pass/fail gate.
