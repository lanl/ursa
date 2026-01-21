import asyncio
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from ursa.agents import ArxivAgent, ExecutionAgent
from ursa.observability.timing import render_session_summary


async def main():
    tid = f"run-{uuid4().hex[:8]}"
    model = init_chat_model(model="openai:gpt-5", max_completion_tokens=20000)

    embedding = init_embeddings("openai:text-embedding-3-large")

    agent = ArxivAgent(
        llm=model,
        summarize=True,
        process_images=False,
        max_results=20,
        rag_embedding=embedding,
        database_path="database_materials1",
        summaries_path="database_summaries_materials1",
        vectorstore_path="vectorstores_materials1",
        download=True,
        thread_id=tid,
    )

    result = await agent.ainvoke(
        arxiv_search_query="high entropy alloy hardness",
        context="What data and uncertainties are reported for hardness of the high entropy alloy and how that that compare to other alloys?",
    )
    print(result["final_summary"])
    executor = ExecutionAgent(llm=model, thread_id=tid)

    exe_plan = f"""
    The following is the summaries of research papers on the high entropy alloy hardness: 
    {result["final_summary"]}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical.
    """

    _ = executor.invoke(exe_plan)

    render_session_summary(tid)


if __name__ == "__main__":
    asyncio.run(main())
