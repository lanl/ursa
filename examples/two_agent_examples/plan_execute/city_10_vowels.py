from uuid import uuid4

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.observability.timing import render_session_summary
from ursa.workflows import PlanningExecutorWorkflow


def main():
    """Run a simple example of an agent."""

    tid = "run-" + uuid4().hex[:8]

    try:
        # Define a simple problem
        problem = "Find a city with as least 10 vowels in its name."
        workspace = "city_vowel_test"

        planner_model = init_chat_model(
            model="openai:gpt-5-mini",
            max_completion_tokens=10000,
            max_retries=2,
        )

        executor_model = init_chat_model(
            model="openai:gpt-5-mini",
            max_completion_tokens=10000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem}\n")

        # Init the agents with the model and checkpointer
        executor = ExecutionAgent(
            llm=executor_model,
            enable_metrics=True,
            thread_id=tid + "_executor",
            workspace=workspace,
        )

        planner = PlanningAgent(
            llm=planner_model,
            enable_metrics=True,
            thread_id=tid + "_planner",
            workspace=workspace,
        )

        workflow = PlanningExecutorWorkflow(
            planner=planner, executor=executor, workspace=workspace
        )

        final_results = workflow(problem)

        render_session_summary(planner.thread_id)
        render_session_summary(executor.thread_id)

        return final_results

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    final_output = main()
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    print(final_output)
