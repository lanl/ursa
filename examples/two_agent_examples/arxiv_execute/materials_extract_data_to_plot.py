from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from ursa.agents import ArxivAgent, ExecutionAgent


def main():
    model = ChatLiteLLM(
        model="openai/o3",
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

    result = agent.run(
        arxiv_search_query="high entropy alloy hardness",
        context="What data and uncertainties are reported for hardness of the high entropy alloy and how that that compare to other alloys?",
    )
    print(result)
    executor = ExecutionAgent(llm=model)
    exe_plan = f"""
    The following is the summaries of research papers on the high entropy alloy hardness: 
    {result}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.action.invoke(init)

    for x in final_results["messages"]:
        print(x.content)


if __name__ == "__main__":
    main()
