import sys

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary

tid = "run-" + __import__("uuid").uuid4().hex[:8]


def main(mode: str):
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        model = init_chat_model(
            model="openai:gpt-5-mini"
            if mode == "prod"
            else "ollama:llama3.1:8b",
            max_completion_tokens=10000 if mode == "prod" else 4000,
            max_retries=2,
        )
        init = {"messages": [HumanMessage(content=problem)]}

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)
        planner.thread_id = tid
        executor.thread_id = tid

        # Solve the problem
        planning_output = planner.invoke(init)
        print(planning_output["messages"][-1].content)
        planning_output["workspace"] = "workspace_cityVowels"
        final_results = executor.invoke(
            planning_output, config={"recursion_limit": 100000}
        )
        for x in final_results["messages"]:
            print(x.content)
        # print(final_results["messages"][-1].content)

        render_session_summary(tid)

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    mode = "dev" if sys.argv[-1] == "dev" else "prod"
    final_output = main(mode=mode)  # dev or prod
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print(final_output)
