from langchain.chat_models import init_chat_model

from ursa.agents import PlanningExecutionAgent
from ursa.util.events import configure_event_logging

configure_event_logging()

problem = """
Design, run, and visualize the effects of the counter-rotating states in the
quantum Rabi model using the QuTiP Python package. Compare the result with the
rotating-wave approximation.

Write a Python file that creates a compelling example, runs it (installing QuTiP
if necessary), visualizes the results, and saves outputs for a website. Also
write a pedagogical description that defines the technical terms. Finally,
create a webpage that presents the output clearly.
"""


def main():
    """Solve the task with one parent planning/execution agent."""
    agent = None
    try:
        workspace = "qutip_workspace"
        model = init_chat_model(
            model="openai:gpt-5.4-mini",
            max_completion_tokens=20000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem}\n")
        agent = PlanningExecutionAgent(
            llm=model,
            workspace=workspace,
            enable_metrics=True,
            thread_id="quantum_rabi_workflow",
        )
        return agent.invoke(problem)
    except Exception as exc:
        print(f"Error in example: {exc!s}")
        import traceback

        traceback.print_exc()
        return {"error": str(exc)}
    finally:
        if agent is not None:
            agent.close()


if __name__ == "__main__":
    main()
