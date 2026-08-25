from langchain.chat_models import init_chat_model

from ursa.environments import AgentTeamEnvironment


llm = init_chat_model("openai:gpt-5.4-mini")
team = AgentTeamEnvironment.from_yaml("agent_team.yaml", llm=llm)
result = team.invoke(
    "Compare two defensible approaches to a small data-analysis task."
)
print(result)
