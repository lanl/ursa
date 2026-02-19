# DSIAgent Documentation

`DSIAgent` is a class that manages access to DSI databases and also manages path location of databases.


## Basic Usage

```python
from ursa.agents import DSIAgent

# Initialize the agent
ai = DSIAgent()

# Run a query
ai.ask("load the dataset at data/oceans_11/ocean_11_datasets.db" )
ai.ask("Tell me about the datasets you have.")

# Print the summary
print(result)
```

## Parameters

When initializing `DSIAgent`, you can customize its behavior with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | `init_chat_model("openai:gpt-5-mini")` | The LLM model to use  |
| `database_path` | str | True | Path to the dataset to load |
| `process_images` | bool | True | Whether to extract and describe images from papers |
| `output_mode` | str | True | Jupyter for jupyter notebooks or console |
| `run_path` | str | True | path where to run |


## Advanced Usage

### Customizing the Agent

```python
import os
from ursa.agents import DSIAgent
from langchain_openai import ChatOpenAI
from pathlib import Path

# Get the data
current_file = Path(__file__).resolve()
current_dir = current_file.parent
run_path = os.getcwd()

dataset_path = str( current_dir / "data/oceans_11/ocean_11_datasets.db" )

model = ChatOpenAI( model="gpt-5.1", max_tokens=100000, timeout=None, max_retries=2)

workspace = "dsi_agent_example"
os.makedirs(workspace, exist_ok=True)
rdb_path = Path(workspace) / "dsi_agent_checkpoint.db"
rdb_path.parent.mkdir(parents=True, exist_ok=True)

ai = DSIAgent(llm=model, database_path=dataset_path, output_mode="console", run_path=run_path)

print("\nQuery: Tell me about the datasets you have.")
ai.ask("Tell me about the datasets you have.")
```



