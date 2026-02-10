import sqlite3
import os
from ursa.agents import DSIAgent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from pathlib import Path

# Get the data
current_file = Path(__file__).resolve()
current_dir = current_file.parent
run_path = os.getcwd()

dataset_path = str( current_dir / "data/oceans_11/ocean_11_datasets.db" )
print(dataset_path)

model = ChatOpenAI( model="gpt-5.1", max_tokens=100000, timeout=None, max_retries=2)

workspace = "dsi_agent_example"
os.makedirs(workspace, exist_ok=True)
rdb_path = Path(workspace) / "dsi_agent_checkpoint.db"
rdb_path.parent.mkdir(parents=True, exist_ok=True)
rconn = sqlite3.connect(str(rdb_path), check_same_thread=False)
dsiagent_checkpointer = SqliteSaver(rconn)


ai = DSIAgent(llm=model, database_path=dataset_path, output_mode="console", checkpointer=dsiagent_checkpointer, run_path=run_path)

print("\nQuery: Tell me about the datasets you have.")
ai.ask("Tell me about the datasets you have.")

print("\nQuery: Do you have any implosion data?")
ai.ask("Do you have any implosion data?")

print("\nQuery: Tell me everything you have about that Ignition dataset")
ai.ask("Tell me everything you have about that Ignition dataset")

print("\nQuery: Can you find some arxiv papers related to this?")
ai.ask("can you find some arxiv papers related to this")

print("\nQuery: Can you find some OSTI papers related to this?")
ai.ask("can you find some OSTI papers related to this")

print("\nQuery: Can you find a websearch on implosion?")
ai.ask("can you find some websearch on implosion?")