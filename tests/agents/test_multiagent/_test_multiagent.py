# NOTE: This will be helpful for prompting.
#     https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide

from pathlib import Path

from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ursa.experimental.agents.multiagent import Ursa


def generate_data(data_path: Path):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 100)
    y = rng.normal(2 * x + 1, 0.1)
    pd.DataFrame(dict(x=x, y=y)).to_csv(data_path, index=False)


# TODO: Need to make `uv run` a SAFE command.
query_1 = """
I have a file `data/data.csv`.

**First**, read the first few lines of the file to understand the format.
Do this quickly; don't go overboard.

**Then**, write a plan (with at most 4 steps) to perform simple linear
regression on this data in python.  The plan MUST NOT include code; though it
may include instruction to write code. The analysis should be **very minimal**
and AS CONCISE AS POSSIBLE.  I care only about the coefficients (including an
intercept). Do not provide other information or plots.

**Then**, EXECUTE THE PLAN using execute_plan_tool. Write all code to
`analysis.py`. DO NOT write anything to `data/`. Do not write any other
files. I want a single file with the entire analysis.

**Finally**, edit `analysis.py` to make it AS CONCISE AS POSSIBLE. Don't
include code for assert, raising errors, exception handling, plots, etc. I want
ONLY a very minimal script that reads the data and then prints the linear
model's coefficients. Remember, I want A SINGLE FILE with the entire analysis
(in `analysis.py`).
"""

# An alternate query to test.
query_2 = """
I have a file `data/data.csv`.

Please write a very minimal python script to perform linear regression on this
data.  The analysis shoud be as concise as possible. I care only about the
coefficients (including an intercept).  Do not provide other information or
plots. Write the analysis to `analysis.py`. Run the code to ensure it works.
"""


def test_multiagent(chat_model):
    # Generate data if not already present.
    workspace = Path(__file__).parent / "workspace"
    data_dir = workspace / "data"
    data_csv = data_dir / "data.csv"
    if not data_csv.exists():
        data_dir.mkdir(exist_ok=True, parents=True)
        generate_data(data_dir / "data.csv")

    # Initialize agent.
    agent = Ursa(
        chat_model,
        max_reflection_steps=0,
        workspace=workspace,
        checkpointer=InMemorySaver(),
    ).create()

    # Store results (AI output) in this list.
    results = []

    def run(query: str):
        print(f"Task:\n{query}")
        results.append(
            result := agent.invoke(
                {"messages": [HumanMessage(query)]},
                {
                    "configurable": {
                        "thread_id": "ursa",
                    },
                    "recursion_limit": 50,
                },
            )
        )
        return result

    run(query_1)

    for result in results:
        for msg in result["messages"]:
            msg.pretty_print()
