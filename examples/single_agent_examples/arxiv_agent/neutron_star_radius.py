import asyncio
from pathlib import Path

from langchain_openai import ChatOpenAI

from ursa.agents import ArxivAgent


async def main():
    llm = ChatOpenAI(model="gpt-5-mini", max_completion_tokens=20000)

    Path("workspace").mkdir(exist_ok=True)
    agent = ArxivAgent(
        llm=llm,
        summarize=True,
        process_images=True,
        max_results=3,
        database_path="workspace/arxiv_papers_neutron_star",
        summaries_path="workspace/arxiv_summaries_neutron_star",
        vectorstore_path="workspace/arxiv_vectorstores_neutron_star",
        download=True,
        enable_metrics=True,
    )

    result = await agent.ainvoke(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )

    print(result["final_summary"])


if __name__ == "__main__":
    asyncio.run(main())
