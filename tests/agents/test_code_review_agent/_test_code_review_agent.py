# WIP
import os

from langchain.chat_models import init_chat_model

from ursa.agents import CodeReviewAgent


def test_code_review_agent():
    code_review_agent = CodeReviewAgent(
        llm=init_chat_model(
            model=os.getenv("URSA_TEST_LLM", "openai:gpt-5-nano")
        )
    )
    initial_state = {
        "messages": [],
        "project_prompt": "Find a city with as least 10 vowels in its name.",
        "code_files": ["vowel_count.py"],
        "edited_files": [],
        "iteration": 0,
    }
    result = code_review_agent.action.invoke(initial_state)

    for x in result["messages"]:
        print(x.content)

    return result
