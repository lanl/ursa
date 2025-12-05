import os

import httpx
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import SecretStr

from ursa.experimental.agents.multiagent import Ursa

aiportal = False

if aiportal:
    llm = ChatOpenAI(
        model=os.environ["CLAUDE"],
        # model="gpt-oss-120b",
        base_url=os.environ["AIPORTAL_API_URL"],
        api_key=SecretStr(os.environ["AIPORTAL_API_KEY"]),
        http_client=httpx.Client(verify=False),
    )
else:
    # llm = init_chat_model("ollama:ministral-3:14b")
    llm = init_chat_model("openai:gpt-5-nano")


agent = Ursa(
    llm,
    max_reflection_steps=0,
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


# run(
#     "Write and execute a very minimal python script to compute Pi using Monte Carlo."
# )

# run("What did you just do?")

# print(results)


# run(
#     "Write a plan to write a very minimal python script to compute Pi using Monte Carlo."
#     "After planning, please execute the plan step by step. Save any code to disk."
# )

# run("Can you now execute the plan?")

query = """
I have a file `data/data.csv`. 

**First**, read the first few lines of the file to understand the format.
Do this quickly; don't go overboard.

**Then**, write a plan (with at most 3 steps) to perform simple linear
regression on this data in python. The linear regression must have a slope and
intercept. The plan MUST NOT include code; though it may include instruction
to write code.  The analysis should be **very minimal** and AS CONCISE AS
POSSIBLE.

**Finally**, EXECUTE THE PLAN using execute_plan_tool. Write any code to
*`output/`. DO NOT write anything to `data/`.
"""
run(query)

for result in results:
    for msg in result["messages"]:
        msg.pretty_print()
