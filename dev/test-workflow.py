from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.workflows import PlanningExecutorWorkflow

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

messages = Path("./messages/workflows")
messages.mkdir(exist_ok=True)
workspaces = Path("./workspaces/workflows")

for name, llm in models.items():
    planner = PlanningAgent(llm=llm, workspace=workspaces / name)
    executor = ExecutionAgent(llm=llm, ursa_logger=planner.ursa_logger)
    workflow = PlanningExecutorWorkflow(
        planner=planner,
        executor=executor,
        ursa_logger=planner.ursa_logger,
    )
    workflow(
        "Write a python script <10 lines to compute Pi "
        "using Monte Carlo; use standard lib only."
        "Use at most two steps."
    )
    workflow.ursa_logger.save_messages(messages / f"{name}.json")
