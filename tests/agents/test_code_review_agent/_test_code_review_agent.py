from ursa.agents import CodeReviewAgent


def test_code_review_agent(chat_model, tmpdir):
    code_review_agent = CodeReviewAgent(llm=chat_model, workspace=tmpdir)
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
