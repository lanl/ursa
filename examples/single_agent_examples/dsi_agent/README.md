# DSI-Explorer Agent example
The dsi_agent example allows data from the [https://github.com/lanl/dsi]( Data Science Infrastructure Project) (DSI) toi be queried using AI.

To use the jupyter-notebook, do the following: 

1. install jupyter-packages:
```bash
uv pip install -U -r examples/single_agent_examples/dsi_agent/requirements.txt
```

2. register the environment with jupyter notebook:
```bash
uv run python -m ipykernel install --user --name venv_ursa --display-name "venv_ursa" # register the environment with Jupyter notebook 
```

3. run:
```
jupyter lab&
```

Copy the link from the terminal and paste in a browser.


4. Open the dsi_explorer notebook and select the venv_ursa kernel