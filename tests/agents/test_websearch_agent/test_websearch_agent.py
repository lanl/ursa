# from langchain_openai import ChatOpenAI
import os

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from ursa.agents import WebSearchAgentLegacy
from ursa.observability.timing import render_session_summary


def test_websearch_agent():
    model = init_chat_model(os.getenv("URSA_TEST_LLM", "openai:gpt-5-nano"))

    websearcher = WebSearchAgentLegacy(llm=model)
    # problem = "Who are the 2025 Detroit Tigers top 10 prospects and what year were they born?"
    problem = "Who won the 2025 International Chopin Competition? Who are his/her piano teachers?"
    inputs = {
        "messages": [HumanMessage(content=problem)],
        "model": model,
    }
    result = websearcher.invoke(inputs)
    msg = result["messages"][-1]
    msg.pretty_print()
    assert "eric lu" in msg.content.lower()

    print("\n\nURLs visited:")
    for i, url in enumerate(result["urls_visited"], start=1):
        print(f"{i}. {url}")

    render_session_summary(websearcher.thread_id)
