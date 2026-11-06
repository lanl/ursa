# URSA environment examples

URSA environments compose persistent agents and other environments behind a single
`invoke(...)` interface. This directory contains YAML configuration examples for:

- `AgentTeamEnvironment`: a hierarchical team led by a PI agent. Team members are
  exposed to the PI as delegation tools.
- `AgentSymposiumEnvironment`: multiple agents or teams independently solve a
  problem, review all submissions, revise their own work, and then an organizer
  synthesizes a final answer.

Load a team from YAML with:

```python
from ursa.cli.config import UrsaConfig
from ursa.environments import AgentTeamEnvironment

config = UrsaConfig.user_config()
llm = config.llm_model.init_chat_model()
team = AgentTeamEnvironment.from_yaml("examples/environments/agent_team.yaml", llm=llm)
result = team.invoke("Analyze this problem...")
```

Load a symposium from YAML with:

```python
from ursa.cli.config import UrsaConfig
from ursa.environments import AgentSymposiumEnvironment

config = UrsaConfig.user_config()
llm = config.llm_model.init_chat_model()
symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml", llm=llm
)
result = symposium.invoke("Solve this complex problem...")
print(result["final"])
```

Named configs can also be saved with `save_team_config(...)` and
`save_symposium_config(...)`; by default they are written under:

- `~/.cache/ursa/<group>/environments/agent_teams/<name>/team.yaml`
- `~/.cache/ursa/<group>/environments/agent_symposia/<name>/symposium.yaml`

Agent checkpoint persistence uses the same group hierarchy under
`~/.cache/ursa/<group>/agents/<agent_name>`. If no workspace is provided for an
environment, its workspace defaults to
`~/.cache/ursa/<group>/environments/workspaces/<name>`.
