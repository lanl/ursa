# NOTE: This will be helpful for prompting.
# https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide

import os
from pathlib import Path

import httpx
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from ursa.experimental.agents.multiagent import Ursa

aiportal = False

if aiportal:
    llm = init_chat_model(
        model=os.environ["CLAUDE"],
        base_url=os.environ["AIPORTAL_API_URL"],
        api_key=os.environ["AIPORTAL_API_KEY"],
        model_provider="openai",
        model_kwargs={"extra_headers": {"disable_fallbacks": "true"}},
        http_client=httpx.Client(verify=False),
    )
else:
    llm = init_chat_model("openai:gpt-5.2")


agent = Ursa(
    llm,
    max_reflection_steps=0,
    workspace=Path("dev-workspace"),
    checkpointer=InMemorySaver(),
).create()

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


# TODO: Need to make `uv run` a SAFE command.
query = """
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
run(query)

for result in results:
    for msg in result["messages"]:
        msg.pretty_print()
