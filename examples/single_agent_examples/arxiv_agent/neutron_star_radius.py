from pathlib import Path

from langchain_openai import ChatOpenAI

from ursa.agents import ArxivAgent
from ursa.observability.timing import render_session_summary

tid = "run-" + __import__("uuid").uuid4().hex[:8]


def main():
    llm = ChatOpenAI(model="o3", max_completion_tokens=20000)

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
    agent.thread_id = tid

    result = agent.invoke(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )

    print(result)

    render_session_summary(tid)


if __name__ == "__main__":
    main()
