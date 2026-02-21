from pathlib import Path

from langchain.chat_models import init_chat_model

from ursa.agents import GitGoAgent


def main():
    repo_root = Path(".").resolve()
    agent = GitGoAgent(
        llm=init_chat_model("openai:gpt-5-mini"),
        workspace=repo_root,
    )

    prompt = (
        "Show git status, list tracked .go files, and summarize any main.go you find."
    )
    result = agent.invoke(prompt)
    print(result["messages"][-1].text)


if __name__ == "__main__":
    main()
