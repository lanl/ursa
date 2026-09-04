# Harbor validation waves

> **Invalid attempt:** the shared host `.venv` lost a Rich Unicode resource,
> causing Harbor's final report to crash. No scores from this attempt count.

This is a fresh, independent validation attempt with seed
`6f5d75c9149c42f0`. Run from the repository root. Each wave uses four
concurrent trials, a 2x agent-timeout multiplier, and
`openai/gpt-5.4-nano`.

```bash
jobs/wave-validation-6f5d75c9149c42f0/run-wave.sh easy ursa-docker
jobs/wave-validation-6f5d75c9149c42f0/run-wave.sh easy codex-docker
jobs/wave-validation-6f5d75c9149c42f0/run-wave.sh easy ursa-apptainer
jobs/wave-validation-6f5d75c9149c42f0/run-wave.sh easy codex-apptainer
```

After all four configurations finish, repeat them with `examples`, then
`medium`. The launcher sources `.env` when present and safely obtains Docker
group access without changing the account's normal primary group. `waves.json`
records the complete seeded task selection.
