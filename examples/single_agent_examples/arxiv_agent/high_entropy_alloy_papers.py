import time

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ursa.agents import ArxivAgent


def main():
    callback_handler = OpenAICallbackHandler()

    llm = ChatOpenAI(
        model="o3",
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

    t0 = time.time()

    agent.run(
        arxiv_search_query="High Entropy Alloys",
        context="Find High entropy alloys suitable for application under extreme conditions. For candidates that you identify, provide the starting structure, crystal structure, lattice parameters, and space group.",
    )

    t1 = time.time()

    print(f"Time Taken: {t1 - t0}")
    print(f"Total Tokens Used: {callback_handler.total_tokens}")
    print(f"Prompt Tokens: {callback_handler.prompt_tokens}")
    print(f"Completion Tokens: {callback_handler.completion_tokens}")
    print(f"Successful Requests: {callback_handler.successful_requests}")
    print(f"Total Cost (USD): ${callback_handler.total_cost}")


if __name__ == "__main__":
    main()
