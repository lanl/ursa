# Harbor validation waves

Run these configurations from the repository root. Each wave uses four concurrent
trials, a 2x agent-timeout multiplier, and `openai/gpt-5.4-nano`.
On this validation host, run the whole process with Docker-group access; the
Apptainer builder can also use Docker as its Dockerfile fallback.

```bash
sg docker -c 'jobs/wave-validation-793a452a34138ea8/run-wave.sh easy ursa-docker'
sg docker -c 'jobs/wave-validation-793a452a34138ea8/run-wave.sh easy codex-docker'
sg docker -c 'jobs/wave-validation-793a452a34138ea8/run-wave.sh easy ursa-apptainer'
sg docker -c 'jobs/wave-validation-793a452a34138ea8/run-wave.sh easy codex-apptainer'
```

After all four configurations finish, repeat them with `examples`, then `medium`.
The launcher sources `.env` when present and assigns a distinct job name to every
wave/configuration pair. `waves.json` records the complete seeded task selection.
