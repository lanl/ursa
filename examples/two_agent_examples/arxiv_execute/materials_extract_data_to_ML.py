from uuid import uuid4

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ursa.agents import ArxivAgent, ExecutionAgent
from ursa.observability.timing import render_session_summary


def main():
    tid = f"run-{uuid4().hex[:8]}"
    model = ChatOpenAI(
        model="o3",
        max_tokens=50000,
    )

    agent = ArxivAgent(
        llm=model,
        summarize=True,
        process_images=False,
        max_results=40,
        database_path="arxiv_papers_materials2",
        summaries_path="arxiv_summaries_materials2",
        vectorstore_path="arxiv_vectorstores_materials2",
        download_papers=True,
        thread_id=tid,
    )

    result = agent.invoke(
        arxiv_search_query="high entropy alloy, yield strength, interstitial",
        context="Extract data that can be used to visualize how yield strength increase (%) depends on the interstital doping atomic percentage.",
    )
    print(result)
    executor = ExecutionAgent(llm=model, thread_id=tid)
    exe_plan = f"""
    The following is the summaries of research papers on how yield strength increase depends on interstital doping percentage: 
    {result}

    Fit a machine learning surrogate to predict yield strength increase (%) from interstital doping atomic percentage and any other relevant features.
    Summarize the results in a markdown document. Include one or more plots of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical. Ensure it is well cited from the reviewed works.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.invoke(init, {"recursion_limit": 10000})

    for x in final_results["messages"]:
        print(x.content)

    render_session_summary(tid)


if __name__ == "__main__":
    main()
