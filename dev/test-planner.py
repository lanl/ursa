import tempfile
from pathlib import Path

from ursa.agents import PlanningAgent
from ursa.util.traced import UrsaOllama, UrsaOpenAI

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

messages = Path("./messages/planner")
messages.mkdir(exist_ok=True, parents=True)

for name, llm in models.items():
    print("Curent model:", name)
    planner = PlanningAgent(llm=llm, workspace=Path(tempfile.mkdtemp()))
    planner.invoke("Hi")
    llm.save_messages(messages / f"{name}.json")
