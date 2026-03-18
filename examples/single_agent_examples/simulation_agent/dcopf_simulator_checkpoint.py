import sqlite3
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.experimental.agents.simulator_agent import SimulatorAgent

problem = (
    "Your task is to perform a do another parameter sweep of dcopf using an open source "
    "code for optimizing power systems in Julia, PowerModels.jl. "
    "The parameter sweep will be performed on the load parameters 50 times by choosing "
    "a random number between 0.5 and 2.0 and multiplying the load by this factor (wider ranges than before)."
    "Produce a plot with opf objective value on the x axis and load factor on the y axis."
    "Highlight any difference between the results of this sweep and the previous sweep you performed in a file called changes.md"
)
workspace = "ursa_simulator_test"

db_path = Path(workspace) / "checkpoint.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(str(db_path), check_same_thread=False)
checkpointer = SqliteSaver(conn)

embedding = init_embeddings("openai:text-embedding-3-small")
model = init_chat_model(model="openai:gpt-5.2")

simulator = SimulatorAgent(
    llm=model,
    workspace=workspace,
    embedding=embedding,
    checkpointer=checkpointer,
    use_web=True,
)
simulator.thread_id = "dcopf_test_executor"

result = simulator.invoke(problem)

print("COMPLETE ------------")
# print("==============\n==============\n\n".join([x.text for x in result["messages"]]))
print(result["messages"][-1].text)
