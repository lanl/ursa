from langchain.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ursa.agents import WebSearchAgent


def test_websearch_agent():
    model = ChatOpenAI(model="gpt-4o-mini")
    websearcher = WebSearchAgent(llm=model, enable_metrics=True)
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
