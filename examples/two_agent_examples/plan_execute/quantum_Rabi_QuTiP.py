from ursa.cli.config import UrsaConfig
from ursa.util.events import configure_event_logging
from ursa.workflows import PlanningExecutionAgent

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
        config = UrsaConfig.user_config()
        model = config.llm_model.init_chat_model()

        print(f"\nSolving problem: {problem}\n")
        agent = PlanningExecutionAgent(
            llm=model,
            workspace=workspace,
            enable_metrics=True,
            agent_name="quantum_rabi_bot",
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
