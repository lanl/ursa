from ursa.cli.config import UrsaConfig
from ursa.environments import AgentTeamEnvironment

config = UrsaConfig.user_config()
llm = config.llm_model.init_chat_model()
team = AgentTeamEnvironment.from_yaml("agent_team.yaml", llm=llm)
result = team.invoke(
    "Compare two defensible approaches to a small data-analysis task."
)
print(result)
