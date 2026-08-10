from uuid import uuid4

from langchain.chat_models import init_chat_model

from ursa.agents import PlanningExecutionAgent
from ursa.observability.timing import render_session_summary
from ursa.util.events import configure_event_logging

configure_event_logging()


def main():
    """Find a city using one persistent planning/execution agent."""
    thread_id = "run-" + uuid4().hex[:8]
    agent = None
    try:
        problem = "Find a city with at least 10 vowels in its name."
        workspace = "city_vowel_test"
        model = init_chat_model(
            model="openai:gpt-5.4-mini",
            max_completion_tokens=10000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem}\n")
        agent = PlanningExecutionAgent(
            llm=model,
            enable_metrics=True,
            thread_id=thread_id,
            workspace=workspace,
        )
        result = agent.invoke(problem)
        render_session_summary(thread_id)
        return result
    except Exception as exc:
        print(f"Error in example: {exc!s}")
        import traceback

        traceback.print_exc()
        return {"error": str(exc)}
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    final_output = main()
    print("=" * 80)
    print(final_output)
