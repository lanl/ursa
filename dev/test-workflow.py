import tempfile
from pathlib import Path

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.util.traced import UrsaOllama, UrsaOpenAI
from ursa.workflows import PlanningExecutorWorkflow

models = {
    "ollama-nemotron-nano": UrsaOllama(
        model="nemotron-3-nano:4b", reasoning=True
    ),
    "ollama-nemotron-3-super": UrsaOllama(
        model="nemotron-3-super:120b", reasoning=True
    ),
    "openai-gpt-5-nano": UrsaOpenAI(
        model="gpt-5-nano", reasoning={"effort": "low", "summary": "auto"}
    ),
    # NOTE: This model does not reason
    "ollama-nemotron-mini": UrsaOllama(model="nemotron-mini:4b"),
}

messages = Path("./messages/workflows")
messages.mkdir(exist_ok=True)


for name, llm in models.items():
    workspace = Path(tempfile.mkdtemp())
    planner = PlanningAgent(llm=llm, workspace=workspace)
    executor = ExecutionAgent(llm=llm, workspace=workspace)
    workflow = PlanningExecutorWorkflow(planner=planner, executor=executor)
    workflow(
        "Write a python script <10 lines to compute Pi "
        "using Monte Carlo; use standard lib only."
        "Use at most two steps."
    )
    llm.save_messages(messages / f"{name}.json")
