# Harbor validation waves

This attempt supersedes invalid seed `6f5d75c9149c42f0` after isolating the
host uv environment. Its fresh seed is `82fa0ed3eb2e4ef4`. Run from the
repository root. Each wave uses four concurrent trials, a 2x agent-timeout
multiplier, and `openai/gpt-5.4-nano`.

```bash
jobs/wave-validation-82fa0ed3eb2e4ef4/run-wave.sh easy ursa-docker
jobs/wave-validation-82fa0ed3eb2e4ef4/run-wave.sh easy codex-docker
jobs/wave-validation-82fa0ed3eb2e4ef4/run-wave.sh easy ursa-apptainer
jobs/wave-validation-82fa0ed3eb2e4ef4/run-wave.sh easy codex-apptainer
```

After all four configurations finish, repeat them with `examples`, then
`medium`. The launcher sources `.env` when present and safely obtains Docker
group access without changing the account's normal primary group. It runs
Harbor in an isolated, locked uv environment and checks Rich rendering before
starting trials. `waves.json` records the complete seeded task selection.
