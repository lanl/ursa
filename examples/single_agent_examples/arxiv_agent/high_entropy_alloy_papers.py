from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ursa.agents import ArxivAgent
from ursa.observability.timing import render_session_summary

tid = "run-" + __import__("uuid").uuid4().hex[:8]


def main():
    callback_handler = OpenAICallbackHandler()

    llm = ChatOpenAI(
        model="gpt-5-mini",
        max_tokens=10000,
        timeout=None,
        max_retries=2,
        callbacks=[callback_handler],
    )

    agent = ArxivAgent(
        llm=llm,
        summarize=True,
        process_images=False,
        max_results=10,
        rag_embedding=OpenAIEmbeddings(),
        database_path="arxiv_HEA_papers",
        summaries_path="arxiv_HEA_summaries",
        vectorstore_path="arxiv_HEA_vectorstores",
        download_papers=True,
    )
    agent.thread_id = tid

    # t0 = time.time()

    results = agent.invoke(
        arxiv_search_query="High Entropy Alloys",
        context="Find High entropy alloys suitable for application under extreme conditions. For candidates that you identify, provide the starting structure, crystal structure, lattice parameters, and space group.",
    )

    print(results)

    render_session_summary(tid)


if __name__ == "__main__":
    main()
