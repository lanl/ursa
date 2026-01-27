import sqlite3
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.experimental.agents.simulator_agent import SimulatorAgent

problem = (
    "Your task is to perform a parameter sweep of dcopf using an open source "
    "code for optimizing power systems in Julia, PowerModels.jl. "
    "The parameter sweep will be performed on the load parameters 10 times by choosing "
    "a random number between 0.8 and 1.2 and multiplying the load by this factor."
    "I require that each parameter configuration be stored in its own input file, ieee14."
    "I require that the code used to perform the task be stored."
    "I require that the code be executed and output saved to a csv file. "
    "Produce a plot with opf objective value on the x axis and load factor on the y axis."
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
    thread_id="dcopf_test_executor",
    use_web=True,
)

result = simulator.invoke(problem)
