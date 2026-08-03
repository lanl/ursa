# Change Log

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
