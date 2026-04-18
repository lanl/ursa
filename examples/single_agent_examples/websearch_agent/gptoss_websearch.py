import asyncio

from langchain.chat_models import init_chat_model

from ursa.agents import WebSearchAgent

# vLLM server: http://localhost:8000
model = init_chat_model(
    model="gpt-oss-20b",
    model_provider="openai",
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
    max_tokens=16384,
    temperature=0.1,
)

problem = "Find a city with at least 10 vowels in its name."

websearcher = WebSearchAgent(llm=model, enable_metrics=True)

websearch_output = asyncio.run(websearcher.ainvoke(problem))

print("Final summary:\n", websearch_output["final_summary"])
print("\nCitations:\n", [x for x in websearch_output.get("urls_visited", [])])
