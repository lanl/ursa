import sqlite3
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.workflows import PlanningExecutorWorkflow


def main():
    """Run a simple example of an agent."""
    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        workspace = "city_vowel_test"

        planner_model = ChatOpenAI(
            model="o3",
            max_tokens=10000,
            max_retries=2,
        )
        executor_model = ChatOpenAI(
            model="o3",
            max_tokens=10000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem}\n")

        # Setup checkpointing
        db_path = Path(workspace) / "checkpoint.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        checkpointer = SqliteSaver(conn)

        # Init the agents with the model and checkpointer
        executor = ExecutionAgent(
            llm=executor_model,
            checkpointer=checkpointer,
            enable_metrics=True,
            thread_id="city_vowel_test_executor",
        )
        planner = PlanningAgent(
            llm=planner_model,
            checkpointer=checkpointer,
            enable_metrics=True,
            thread_id="city_vowel_test_planner",
        )

        agent = PlanningExecutorWorkflow(
            llm=executor_model,
            planner=planner,
            executor=executor,
            workspace=workspace,
            enable_metrics=True,
            thread_id="city_vowel_test_workflow",
        )

        final_results = agent.invoke(problem)

        render_session_summary(agent.thread_id)

        return final_results

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    final_output = main()  # dev or prod
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print(final_output)
