from langchain.chat_models import init_chat_model

from ursa.agents import ArxivAgentLegacy
from ursa.observability.timing import render_session_summary


def test_arxiv_agent_legacy():
    agent = ArxivAgentLegacy(llm=init_chat_model(model="openai:gpt-5-nano"))
    result = agent.invoke(
        arxiv_search_query="Experimental Constraints on neutron star radius",
        context="What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
    )
    print(result)
    render_session_summary(agent.thread_id)
