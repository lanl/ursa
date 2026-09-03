# Harbor validation waves

Run these configurations from the repository root. Each wave uses four concurrent
trials, a 2x agent-timeout multiplier, and `openai/gpt-5.4-nano`.
The launcher safely obtains Docker-group access while preserving the account's
normal primary group, which Apptainer fakeroot requires.

```bash
jobs/wave-validation-26027050c9f34b52/run-wave.sh easy ursa-docker
jobs/wave-validation-26027050c9f34b52/run-wave.sh easy codex-docker
jobs/wave-validation-26027050c9f34b52/run-wave.sh easy ursa-apptainer
jobs/wave-validation-26027050c9f34b52/run-wave.sh easy codex-apptainer
```

After all four configurations finish, repeat them with `examples`, then `medium`.
The launcher sources `.env` when present and assigns a distinct job name to every
wave/configuration pair. `waves.json` records the complete seeded task selection.
