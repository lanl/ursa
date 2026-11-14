# Runtime workers

The CLI's `AgentRuntime` and the reusable `AgentManager` share three execution
modes. Message routing, callbacks, hierarchy tools, state commits, and the TUI
remain host-managed in every mode. Secrets are resolved by the host before an
agent is constructed.

| `runtime.backend` | Execution |
|---|---|
| `process` (default) | One isolated local process per agent. |
| `thread` | One event-loop thread per agent, inside the host process. |
| `pool` | A bounded set of worker processes, each hosting multiple agent threads. |

```yaml
runtime:
  backend: pool
  min_workers: 1
  max_workers: 4
  agents_per_worker: 8
  rebalance_interval: 10
  idle_timeout: 60
```

Workers start lazily. Local pools periodically grow toward one worker per
`agents_per_worker` agents, within the configured bounds. This setting is a
placement target, not a limit on the agent population; `subagents.max_agents`
continues to control that limit. New agents go to the least busy available
worker. Each agent still serializes its own invocations.

Rebalancing moves idle agents using their last committed state. It never
replays or moves an in-flight operation. Custom agents must expose `state`,
`snapshot_state`, or `export_state` (and restore that state through `state` or
`restore_state`) to participate in migration. An agent without exported state
stays on its original worker. Empty local workers retire after `idle_timeout`;
`min_workers` are retained. Set equal minimum and maximum counts for a fixed
local pool.

## MPI and Slurm

Set `launch_command` to start a fixed group of `max_workers` processes. The host
accepts their connections and assigns agents to them dynamically. The launcher
does not need MPI collectives or `mpi4py`; MPI/Slurm provide process placement.
The same Python environment, URSA source, workspaces/checkpoints, rendezvous
directory, and TLS certificate must be accessible on every node.

```yaml
runtime:
  backend: pool
  max_workers: 4
  rebalance_interval: 10
  bind_host: 0.0.0.0
  advertise_host: host.example.org
  rendezvous_dir: /shared/my-job
  tls_certificate: /shared/my-job/host-cert.pem
  tls_key: /shared/my-job/host-key.pem
  launch_command:
    - srun
    - --ntasks={workers}
    - --mpi=none
    - "{python}"
    - -m
    - ursa.runtime.worker
    - --rendezvous
    - "{rendezvous}"
```

For MPI, replace the launcher prefix with `mpiexec`, `-n`, `{workers}`. Run the
host inside an appropriate scheduler allocation and supply the site's node,
CPU binding, account, and partition options. An argument list is recommended;
a string is split into arguments with shell quoting rules. Commands are never
evaluated by a shell. Supported placeholders are `{workers}`, `{python}`, and
`{rendezvous}`. Use an explicit Python path if nodes use a different location.

Externally launched pools retain their fixed process count while agents are
created and stopped. Periodic balancing still runs. `min_workers`,
`agents_per_worker`, and `idle_timeout` control only local process elasticity.
Worker loss fails affected in-flight requests rather than replaying them.
Registered agents retain their last committed state and can restart on a
surviving worker; `AgentSpec.restart_on_failure` enables automatic restart.
If an MPI launcher terminates the entire job, restart the runtime/job.

Non-loopback listeners require TLS. The certificate must identify the
advertised host; workers trust that certificate and mutually authenticate with
a per-runtime key before accepting Python objects. The key lives in an
owner-only rendezvous file, not command arguments. The host removes that file
and terminates its launcher when closed. Workspaces and checkpoints must be on
a filesystem supporting their existing SQLite and file-locking requirements.

## Threaded execution and cancellation

Threaded agents have separate event loops and configurations but share process
globals. Per-agent `AgentSpec.environment` and `working_directory` overrides
require the `process` backend; use explicit workspace paths in threaded code.
Agent output is routed through logging while the host TUI retains its output.

Stopping a threaded agent closes admission immediately and cancels its async
work without waiting for other agents. Python cannot forcibly kill a thread
inside an uncooperative synchronous/native tool; that tool may continue until
it returns. Use `process` for tools requiring hard termination or process-level
isolation. Killing a worker process affects every agent assigned to it.

For library use, pass `runtime=RuntimeConfig(...)` to `AgentManager` (or
`SyncAgentManager`). `manager.worker_snapshots()` reports worker IDs, hostnames,
PIDs, assigned agents, and pending operations. See the
[scaling example](../../examples/runtime_workers/README.md) for a repeatable
fixed-worker, variable-agent workload using Nemotron 3.5 Super.
