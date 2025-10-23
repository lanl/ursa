from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ursa.agents import ArxivAgent, ExecutionAgent
from ursa.observability.timing import render_session_summary

tid = "run-" + __import__("uuid").uuid4().hex[:8]


def main():
    model = ChatOpenAI(
        model="o3",
        max_tokens=50000,
    )

    agent = ArxivAgent(
        llm=model,
        summarize=True,
        process_images=False,
        max_results=20,
        database_path="database_materials1",
        summaries_path="database_summaries_materials1",
        vectorstore_path="vectorstores_materials1",
        download_papers=True,
    )
    agent.thread_id = tid

    result = agent.invoke(
        arxiv_search_query="high entropy alloy hardness",
        context="What data and uncertainties are reported for hardness of the high entropy alloy and how that that compare to other alloys?",
    )
    print(result)
    executor = ExecutionAgent(llm=model)
    executor.thread_id = tid
    exe_plan = f"""
    The following is the summaries of research papers on the high entropy alloy hardness: 
    {result}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.invoke(init)

    for x in final_results["messages"]:
        print(x.content)

    render_session_summary(tid)


if __name__ == "__main__":
    main()
