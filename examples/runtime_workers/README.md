# A changing team on a fixed worker pool

This example starts two worker processes and varies the population through 4,
16, 32, and 4 agents. Each agent asks Nemotron 3.5 Super for a science joke. After
each wave, the host sends every agent a message and verifies its sender and
content. Agents retain independent state as the population changes.

The supplied config sets only the workspace and worker pool. From the
repository root, make a local copy:

```bash
cp examples/runtime_workers/pool.yaml runtime-workers.local.yaml
```

Add your own `llm_model` settings and any referenced `inference_providers` to
that copy, using your usual endpoint and credential mechanism. Select Nemotron
3.5 Super (`NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4` or
`nemotron-3.5-super`) and allow at least 1,024 completion tokens. The standalone
script reads only the config file passed below; it does not merge CLI user
settings automatically. Then run:

```bash
uv run python -m examples.runtime_workers.scaling \
  --config runtime-workers.local.yaml \
  --agents 4 16 32 4 --output scaling-results.json
```

`scaling-results.json` records wave time, worker placement, each reply, model
request time, process ID, thread ID, and per-agent call count. Compare different
worker counts with identical populations and model settings. Wave time includes
startup, model work, and message verification; it is not pure model latency.

Use `--transport-only --config examples/runtime_workers/pool.yaml` to verify
placement and messaging without model settings or model calls. Every real
model call in this example is restricted to Nemotron 3.5 Super.

To test multiple nodes, replace the `runtime` section with the MPI/Slurm setup
in [Runtime workers](../../docs/agents/runtime.md). Start the host within your
allocation, use a shared workspace and rendezvous directory, and provide a TLS
certificate identifying the host. The worker process count remains fixed while
the example creates and removes agents.

## Validation

This workload was exercised with Nemotron 3.5 Super locally and on two Chicoma
nodes. Every wave verified nonempty model replies, message sender/content, and
independent per-agent state. Worker counts remained fixed throughout.

| Placement | Workers | Agent populations | Result |
|---|---:|---|---|
| Local processes | 2 | 4 → 16 → 32 → 4 | Passed |
| Slurm, two nodes, job 24048490 | 4 | 4 → 16 → 32 → 4 | Passed |
| Slurm, two nodes, job 24048494 | 2 | 4 → 16 → 32 → 4 | Passed |

The automated runtime tests also cover MPI launching, elastic growth and
retirement, idle-state migration, worker failure, cancellation, TLS, and
executor saturation. These are correctness checks; model-service caching,
startup, and network conditions make single-run timings unsuitable for ranking
worker counts.
