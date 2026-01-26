from ursa.agents import DSIAgent
from langchain_openai import ChatOpenAI
from pathlib import Path

# Get the data
current_file = Path(__file__).resolve()
current_dir = current_file.parent

dataset_path = current_dir / "data/oceans_11/ocean_11_datasets.db"
print(dataset_path)

model = ChatOpenAI( model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2)


ai = DSIAgent(llm=model, db_index_name=dataset_path, output_mode="console")

print("\nQuery: Tell me about the datasets you have.")
ai.ask("Tell me about the datasets you have.")

print("\nQuery: Do you have any implosion data?")
ai.ask("Do you have any implosion data?")

print("\nQuery: Can you find some arxiv papers related to this?")
ai.ask("can you find some arxiv papers related to this")