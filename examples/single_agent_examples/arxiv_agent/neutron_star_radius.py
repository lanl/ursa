from pathlib import Path

from langchain_litellm import ChatLiteLLM

from ursa.agents import ArxivAgent


def main():
    llm = ChatLiteLLM(model="openai/o3", max_completion_tokens=20000)

    Path("workspace").mkdir(exist_ok=True)
    agent = ArxivAgent(
        llm=llm,
        summarize=True,
        process_images=True,
        max_results=3,
        database_path="workspace/arxiv_papers_neutron_star",
        summaries_path="workspace/arxiv_summaries_neutron_star",
        vectorstore_path="workspace/arxiv_vectorstores_neutron_star",
        download_papers=True,
    )

    result = agent.run(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )

    print(result)


if __name__ == "__main__":
    main()
