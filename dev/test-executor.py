from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent

models = {
    "ollama-nemotron-nano": init_chat_model(
        "ollama:nemotron-3-nano:4b", reasoning=True
    ),
    "ollama-nemotron-3-super": init_chat_model(
        "ollama:nemotron-3-super:120b", reasoning=True
    ),
    "openai-gpt-5-nano": init_chat_model(
        "openai:gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
    ),
    # NOTE: This model does not reason
    "ollama-nemotron-mini": init_chat_model("ollama:nemotron-mini:4b"),
}

messages = Path("./messages/executor")
messages.mkdir(exist_ok=True, parents=True)
workspaces = Path("./workspaces/executor")

for name, llm in models.items():
    executor = ExecutionAgent(llm=llm, workspace=workspaces / name)
    executor.invoke("Hi")
    executor.ursa_logger.save_messages(messages / f"{name}.json")
