import tempfile
from pathlib import Path

from ursa.agents import ExecutionAgent
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

messages = Path("messages/executor")
messages.mkdir(parents=True, exist_ok=True)

for name, llm in models.items():
    executor = ExecutionAgent(llm=llm, workspace=Path(tempfile.mkdtemp()))
    executor.invoke(
        "Write a python script to print the first 10 positive integer."
    )
    llm.save_messages(messages / f"{name}.json")
