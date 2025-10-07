from ursa.agents.arxiv_agent import ArxivAgent
from ursa.observability.timing import render_session_summary


def test_arxiv_agent_runs():
    agent = ArxivAgent(enable_metrics=True)
    result = agent.invoke(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )
    print(result)
    render_session_summary(agent.thread_id)
