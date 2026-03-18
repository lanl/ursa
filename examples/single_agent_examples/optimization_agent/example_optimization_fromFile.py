import os
import pprint

from langchain_openai import ChatOpenAI

from ursa.agents.optimization_agent import OptimizationAgent

model = ChatOpenAI(
    model="gpt-5-mini", max_tokens=10000, timeout=None, max_retries=2
)


filename = "data/3-infeasible/description.txt"
filename = "data/2/description.txt"
abspath = os.path.join(
    os.getcwd(),
    "examples/single_agent_examples/optimization_agent",
    filename,
)

fopen = open(abspath)
ftext = fopen.read()

problem_string = f"""
Here is an optimization problem: {ftext}

Keep your answers short.

Formulate this problem mathematically.
"""

execution_agent = OptimizationAgent(llm=model)

inputs = {"user_input": problem_string}


print("Started execution: \n")

result = execution_agent.invoke(inputs)
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Output of the LLM:\n")
pprint.pprint(result)
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Summary:\n")
print(f"{result['summary']}\n")


print("End execution\n")
