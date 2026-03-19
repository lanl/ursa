# DSIAgent Documentation

`DSIAgent` is a class that manages access to DSI owned databases and also manages path location of databases. The [DSI](https://github.com/lanl/dsi) framework support a number of databases, such as Sqlite, DuckDB, ... . More information is available [here](https://lanl.github.io/dsi/introduction.html).


## Basic Usage

```python
ai = DSIAgent(
    llm=model,
    database_path=dataset_path,
    output_mode="console",
    checkpointer=dsiagent_checkpointer,
    run_path=run_path,
)

print("\nQuery: Tell me about the datasets you have.")
response = ai.ask("Tell me about the datasets you have.")
print(response)
```

See more examples in the [DSI examples folder](../examples/single_agent_examples/dsi_agent/) and [below](#using-python).


## Parameters

When initializing `DSIAgent`, you can customize its behavior with these parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | `BaseChatModel` | `init_chat_model("openai:gpt-5-mini")` | The LLM model to use  |
| `database_path` | str | True | Path to the dataset to load |
| `process_images` | bool | True | Whether to extract and describe images from papers |
| `output_mode` | str | True | Jupyter for jupyter notebooks or console |
| `checkpointer`|  ursa.util.Checkpointer | dsiagent_checkpointer | Path to a checkpoint is **required**
| `run_path` | str | True | path where to run |


## Advanced Usage

### From the URSA CLI
```bash
ursa % ursa

  __  ________________ _
 / / / / ___/ ___/ __ `/
/ /_/ / /  (__  ) /_/ /
\__,_/_/  /____/\__,_/

╭───────────────────────╮
│ LLM endpoint: Default │
│ LLM model: gpt-5.2    │
╰───────────────────────╯
For help, type: ? or help. Exit with Ctrl+d.
ursa> dsi
dsi: can you load the dsi dataset at examples/single_agent_examples/dsi_agent/data/oceans_11/ocean_11_datasets.db 
No DSI database provided. Please load one
Loaded the DSI dataset successfully.                                                                                                                      

 • Active DSI database: /Users/pascalgrosset/projects/ursa/examples/single_agent_examples/dsi_agent/data/oceans_11/ocean_11_datasets.db                   

ursa> dsi
dsi: what are the datasets available
Datasets available in ocean_11_datasets.db:                                                                                                               

  1 3D FLASH Computation of National Ignition Facility Shot (Theme: fusion) — nif.db                                                                      
  2 Bowtie Dataset (Theme: manufacturing)                                                                                                                 
  3 Deep Water Impact Ensemble Dataset (Theme: physics)                                                                                                   
  4 Gray-Scott reaction-diffusion dataset (Theme: physics)                                                                                                
  5 Heat Equations (Theme: physics)                                                                                                                       
  6 Higrad Firetex Wildfire Simulations (Theme: physics)                                                                                                  
  7 Monopoly Dataset (Theme: manufacturing)                                                                                                               
  8 Poisson's Equations for Electrostatics (Theme: Physics) — poisson.db                                                                                  
  9 The High Explosives & Affected Targets (HEAT) Dataset (Theme: eulerian)                                                                               

ursa> dsi
dsi: do you have any ignition datasets
Yes—there is 1 ignition-related dataset in this DB:                                                                                                       

 • 3D FLASH Computation of National Ignition Facility Shot (Theme: fusion) — linked DSI DB: nif.db                                                        

ursa> dsi
dsi: tell me more about it
Here’s more about “3D FLASH Computation of National Ignition Facility Shot”:                                                                              

 • Theme: fusion                                                                                                                                          
 • Keywords: NIF, fusion, 3d, simulation                                                                                                                  
 • Last updated: 2023-03-22                                                                                                                               
 • What it is: A 3D FLASH radiation-hydrodynamics simulation of a cylindrical implosion experiment at the National Ignition Facility, shot N210201-001.   
 • What’s in it: 61 HDF5 output files covering 0–30 ns with outputs every 0.5 ns. Each file includes cell-wise fields such as:                            
    • Material concentrations: ablt (ablator CH), cham (helium), mark (aluminum marker), foam (CH foam)                                                   
    • Physical state: dens (g/cm³), pres (erg/cm³), tele (K), tion (K)                                                                                    
    • Drive deposition: depo (erg/g)                                                                                                                      
 • Domain / resolution notes: cubic domain with adaptive mesh refinement; minimum cell size ~8 microns.                                                   
 • Access rights: unclassified (CUI: none)                                                                                                                
 • Contact: Joshua Paul Sauppe — jpsauppe@lanl.gov                                                                                                        
 • Originating organization: DOE:LANL                                                                                                                     
 • Linked DSI database: nif.db                                                                                                                            

If you want, I can load nif.db next (from the same folder) and list/preview the actual data tables it contains.
```

### Using python

```python
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from ursa.agents import DSIAgent
from ursa.util import Checkpointer

# Get the data
current_file = Path(__file__).resolve()
current_dir = current_file.parent
run_path = os.getcwd()

dataset_path = str(current_dir / "data/oceans_11/ocean_11_datasets.db")
print(dataset_path)

model = ChatOpenAI(
    model="gpt-5.4", max_completion_tokens=10000, timeout=None, max_retries=2
)

workspace = Path("dsi_agent_example")
dsiagent_checkpointer = Checkpointer.from_workspace(workspace)


ai = DSIAgent(
    llm=model,
    database_path=dataset_path,
    output_mode="console",
    checkpointer=dsiagent_checkpointer,
    run_path=run_path,
)

print("\nQuery: Tell me about the datasets you have.")
response = ai.ask("Tell me about the datasets you have.")
print(response)
```



