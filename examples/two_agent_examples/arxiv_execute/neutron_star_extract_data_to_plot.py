from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ArxivAgent, ExecutionAgent
from ursa.observability.timing import render_session_summary


def main():
    tid = f"run-{uuid4().hex[:8]}"
    model = init_chat_model(
        model="openai:gpt-5-mini",
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
        download=True,
    )
    agent.thread_id = tid

    result = agent.invoke(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )
    print(result)
    executor = ExecutionAgent(llm=model)
    executor.thread_id = tid
    exe_plan = f"""
    The following is the summaries of research papers on the contraints on neutron
    star radius: 
    {result}

    Summarize the results in a markdown document. Include a plot of the data extracted from the papers. This 
    will be reviewed by experts in the field so technical accuracy and clarity is 
    critical.
    """

    init = {"messages": [HumanMessage(content=exe_plan)]}

    final_results = executor.invoke(init)

    for x in final_results["messages"]:
        print(x.content)

    render_session_summary(tid)


if __name__ == "__main__":
    main()
