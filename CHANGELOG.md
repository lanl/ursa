# Change Log

## v0.17.0

Changes since `v0.16.4`:

### Terminal UI (TUI)

- Replaced the Rich REPL with a new Textual terminal application, split into
  focused modules with per-event "cards" for source code, command output,
  files, diffs, plans, tools, search, and agent/artifact activity. Added an
  expandable exception card, an explicit `exit` command, theming, a growing
  multi-line prompt, runtime hardening, and randomized startup tips with
  platform-aware keymaps (#315).
- Surfaced persistent named-agent sessions and agent routing directly in the
  TUI, and marked each user turn in the scrollback so prompts stay visible
  between agent output blocks (#311, #315).
- Hardened TUI teardown so a Mount dispatch racing app shutdown can no longer
  crash the app, and stabilized the associated timing tests (#324).

### Configuration and inference providers

- Added layered CLI configuration with XDG-based overrides and reusable,
  provider-aware inference-provider definitions, so models can be declared once
  and shared across agents and environments (#314).
- Added external secret references resolved from the OS credential store,
  including support for injecting secrets into MCP request headers (#314).
- Expanded the `print_config` command and hardened configuration resolution and
  source merging, with correct XDG config paths on Windows (#314).

### CLI

- Added an `ursa self` command to inspect and manage a `uv tool` installation,
  with `status`, `update` (preserving the install recipe), and `modify`
  (extras, extra packages, exact version, or Git ref) subcommands (#325).
- Added the running URSA version to the startup banner (#307).
- Reported missing/invalid API keys and model-initialization errors cleanly,
  without a traceback, for OpenAI and non-OpenAI endpoints (#302).

### Agents

- Made the acquisition agents concurrent (async search, materialization, and
  cached-item loading) and registered their typed graph state; added a
  `build_config` helper and `SourceTask`/`ProcessedSource` structures (#314,
  #315).
- Added an `UnregisteredAgentStateWarning` that flags `BaseAgent` subclasses
  declaring a typed state they never register (#297).
- Landed context summaries as framed human-role messages so a summarized
  history never ends on an assistant summary, keeping message sequences
  provider-valid (#309).
- Kept deep-review role/phase prompts out of the persisted message channel,
  ensured each debate phase leads with its own role prompt, and propagated
  phase model failures immediately instead of swallowing them (#308).
- Fixed `print_visited_sites` to return only the visited-sites update, so
  deep-review outputs report exactly one entry per real iteration (#313).
- Added cross-agent regression coverage pinning provider-valid message
  sequences across agents (#297).

### Observability

- Repaired the OpenTelemetry OTLP/HTTP export path (Tier 1 of #259): each export
  builds a private, per-call tracer provider (shut down in-call) that never
  touches the global provider, emits a real-time root span plus one GenAI
  semantic-convention child span per LLM call, and returns a structured,
  truthfully reported result. Endpoint resolution follows a documented
  precedence (parameter, field, then `OTEL_EXPORTER_OTLP_*` env vars); headers
  accept a mapping or an env-style string; a missing `otel` extra now warns with
  an install hint; and the inert `otel_metrics` constructor argument is
  deprecated (#305).
- Captured reasoning and cached-token counts from all usage carriers
  (langchain `usage_metadata` details, raw OpenAI Responses/Anthropic shapes,
  service-tier-prefixed keys) without double-counting (#316).
- Recorded error samples when `on_llm_error` has no matching start, so failed
  LLM calls are no longer silently dropped from the per-LLM metrics (#293).
- Emitted arXiv and web search progress events asynchronously via `aemit`
  (#320).

### Dashboard

- Gave the dashboard chat visible failure feedback so pre-run send errors name
  the real problem instead of silently orphaning the user's message (#301,
  #306).

### Documentation and maintenance

- Reorganized and expanded the getting-started, configuration (files/env,
  models, secrets, MCP), CLI/TUI, and reference (OTel, CLI) documentation; added
  a filterable example catalog with per-example READMEs and runnable project
  scaffolding; and added versioned-docs build hooks (#314, #315, #318, #320).
- Migrated packaged TUI stylesheets into the distribution, bumped `rich`,
  `langchain-mcp-adapters`, `textual`, and `keyring` dependencies, narrowed the
  `otel` extra to the OTLP/HTTP exporter, and added `hypothesis` to the dev
  group (#314, #315, #305).

## v0.16.4

Changes since `v0.16.3`:

### Agents and CLI

- Expanded the LAMMPS agent with user-supplied potentials, input templates and
  structure data, GPU launch support, potential-selection-only mode, automatic
  result summarization, improved repair history, and structured progress events
  (#277).
- Added richer default CLI logging with MIME-aware rendering for source code,
  command output, diffs, plans, and tool/agent progress. Fixed artifact
  rendering for named agents (#278, #288).
- Fixed PlanningAgent state and message accumulation, and simplified its
  review/revision flow to avoid invalid model message sequences (#285, #286).
- Added `--config` support to `ursa mcp-server` (#275).

### Dashboard

- Added secure API-key management using the operating-system credential store,
  while retaining environment-variable and keyless configuration options
  (#275).
- Added dashboard launching, monitoring, continuation, and cancellation of agent
  teams and symposia (#280).
- New sessions now require an explicit persistent or temporary workspace instead
  of receiving a hidden default workspace; legacy session workspaces remain
  recoverable (#282).
- Refined dashboard layout, controls, settings, and model-provider handling
  (#275, #276).

### Persistence

- Unnamed CLI sessions are now ephemeral and no longer create checkpoint
  databases. Named sessions remain persistent, with a migration warning for
  legacy workspace checkpoints (#279, #287).
- Persistent-agent SQLite databases now prune obsolete checkpoint history after
  successful runs, substantially reducing long-running storage growth (#286).

### Documentation and maintenance

- Reorganized and expanded the agent, logging, API, environment, persistence,
  installation, and tutorial documentation (#278).
- Added versioned documentation publishing and updated MCP server guidance
  (#281).
