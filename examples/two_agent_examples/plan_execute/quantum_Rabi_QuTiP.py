from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent, PlanningAgent
from ursa.workflows import PlanningExecutorWorkflow

problem = """
Design, run and visualize the effects of the counter-rotating states in the quantum Rabi model using the QuTiP
python package. Compare with the Rotating wave approximation.

Write a python file to:
  - Create a compelling example
  - Build the case in python with the QuTiP package, installing QuTiP if necessary.
  - Visualize the results and create outputs for future website visualization.
  - Write a pedogogical description of the example, its motivation, and the results. Define technical terms.

Then create a webpage to present the output in a clear and engaging manner. 
"""


def main():
    """
    Run an example where a planning agent generates a multistep plan and the execution agent is
    queried to solve the problem step by step.
    """
    try:
        workspace = "qutip_workspace"

        model = init_chat_model(
            model="openai:gpt-5-mini",
            max_completion_tokens=20000,
            max_retries=2,
        )

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model, workspace=workspace)
        executor = ExecutionAgent(llm=model, workspace=workspace)

        workflow = PlanningExecutorWorkflow(
            planner=planner,
            executor=executor,
            workspace=workspace,
            enable_metrics=True,
            thread_id="city_vowel_test_workflow",
        )

        # Solve the problem
        final_results = workflow.invoke(problem)

        return final_results

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main()
