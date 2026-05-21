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
from langchain.chat_models import init_chat_model
from ursa.environments import AgentTeamEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")
team = AgentTeamEnvironment.from_yaml("examples/environments/agent_team.yaml", llm=llm)
result = team.invoke("Analyze this problem...")
```

Load a symposium from YAML with:

```python
from langchain.chat_models import init_chat_model
from ursa.environments import AgentSymposiumEnvironment

llm = init_chat_model(model="openai:gpt-4o-mini")
symposium = AgentSymposiumEnvironment.from_yaml(
    "examples/environments/agent_symposium.yaml", llm=llm
)
result = symposium.invoke("Solve this complex problem...")
print(result["final"])
```

Named configs can also be saved with `save_team_config(...)` and
`save_symposium_config(...)`; by default they are written under:

- `~/.cache/agent_teams/<group>/<name>/team.yaml`
- `~/.cache/agent_symposiums/<group>/<name>/symposium.yaml`

Agent checkpoint persistence still uses the existing URSA agent cache under
`~/.cache/ursa_agents/<group>/<agent_name>`.
