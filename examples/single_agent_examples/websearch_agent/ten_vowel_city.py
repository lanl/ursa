from langchain.chat_models import init_chat_model

from ursa.agents import WebSearchAgent

##### Run a simple example of a WebSearch Agent.

# Define a simple problem
problem = "Find a city with as least 10 vowels in its name."

# Choose the LLM and
model = init_chat_model(model="openai:gpt-5-mini", max_completion_tokens=20000)

# Initialize the agent
websearcher = WebSearchAgent(llm=model, enable_metrics=True)

# Solve the problem
websearch_output = websearcher.invoke(problem)

# Print results
print("Final summary: \n", websearch_output["messages"][-1].content)
# for x in websearch_output["messages"]:
#     print(x.content)

print("Citations: \n", [x for x in websearch_output.get("urls_visited", [])])
