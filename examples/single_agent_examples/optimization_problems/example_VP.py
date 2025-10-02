from langchain_openai import ChatOpenAI
from ursa.agents.optimization_agent   import OptimizationAgent

import pprint


model = ChatOpenAI(
    model       = "gpt-4o",
    max_tokens  = 10000,
    timeout     = None,
    max_retries = 2)

problem_string = f'''
Consider the 1D1V Vlasov-poisson equation with two-stream instability case. Find the external electric field with optimal norm that stabilizes this system. 
Discretize the PDE using an accurate and stable numerical scheme.
'''

execution_agent = OptimizationAgent(llm=model)

inputs = {"user_input": problem_string}


print("Started execution: \n")

result = execution_agent.action.invoke(inputs, {"recursion_limit":500})
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Output of the LLM:\n")
pprint.pprint(result)
print("------------------------------------------\n")
print("------------------------------------------\n")
print("Summary:\n")
print(f"{result["summary"]}\n")


print("End execution\n")


