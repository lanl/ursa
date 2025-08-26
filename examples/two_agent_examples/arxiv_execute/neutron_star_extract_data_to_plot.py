from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

from ursa.agents import ArxivAgent, ExecutionAgent


def main():
    model = ChatLiteLLM(
        model="openai/gpt-5",
        max_completion_tokens=50000,
    )

    agent = ArxivAgent(
        llm=model,
        summarize=True,
        process_images=False,
        max_results=5,
        database_path="arxiv_papers_neutron_star",
        summaries_path="arxiv_summaries_neutron_star",
        vectorstore_path="arxiv_vectorstores_neutron_star",
        download_papers=True,
    )

    result = agent.run(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )
    print(result)
    executor = ExecutionAgent(llm=model)
    exe_plan = f"""
    The following is the summaries of research papers on the contraints on neutron
    star radius: 
    {result}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is 
    critical.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.action.invoke(init)

    for x in final_results["messages"]:
        print(x.content)


if __name__ == "__main__":
    main()
