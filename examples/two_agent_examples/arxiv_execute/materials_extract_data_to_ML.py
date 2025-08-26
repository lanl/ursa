from langchain_community.callbacks import get_openai_callback
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
        max_results=40,
        database_path="arxiv_papers_materials2",
        summaries_path="arxiv_summaries_materials2",
        vectorstore_path="arxiv_vectorstores_materials2",
        download_papers=True,
    )

    result = agent.run(
        arxiv_search_query="high entropy alloy, yield strength, interstitial",
        context="Extract data that can be used to visualize how yield strength increase (%) depends on the interstital doping atomic percentage.",
    )
    print(result)
    executor = ExecutionAgent(llm=model)
    exe_plan = f"""
    The following is the summaries of research papers on how yield strength increase depends on interstital doping percentage: 
    {result}

    Fit a machine learning surrogate to predict yield strength increase (%) from interstital doping atomic percentage and any other relevant features.
    Summarize the results in a markdown document. Include one or more plots of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is critical. Ensure it is well cited from the reviewed works.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.action.invoke(init, {"recursion_limit": 10000})

    for x in final_results["messages"]:
        print(x.content)


if __name__ == "__main__":
    with get_openai_callback() as cbh:
        main()
    print(f"Total Tokens Used: {cbh.total_tokens}")
    print(f"Prompt Tokens: {cbh.prompt_tokens}")
    print(f"Completion Tokens: {cbh.completion_tokens}")
