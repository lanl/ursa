from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from ursa.agents import ExecutionAgent, PlanningAgent

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
        model = init_chat_model(
            model="openai:gpt-5-mini",
            max_completion_tokens=20000,
            max_retries=2,
        )

        init = {"messages": [HumanMessage(content=problem)]}

        print(f"\nSolving problem: {problem}\n")

        # Initialize the agent
        planner = PlanningAgent(llm=model)
        executor = ExecutionAgent(llm=model)

        # Solve the problem
        planning_output = planner.invoke(init)
        print(planning_output["messages"][-1].content)
        last_step_string = "This is the first step."
        for x in planning_output["plan_steps"]:
            plan_string = str(x)
            execute_string = """
            Execute this step and report results for the executor of the next step. 
            Do not use placeholders. 
            Run commands to execute code generated for the step if applicable.
            Only address the current step. Stay in your lane.
            """
            step_prompt = f"""
                You are contributing to a larger solution aimed at {problem}.
                If there are previous steps, the summary of the most recent step is: {last_step_string}.
                The current substep is: {plan_string}.

                {execute_string}
            """
            final_results = executor.invoke(
                {
                    "messages": [HumanMessage(content=step_prompt)],
                    "workspace": "workspace_qutip",
                },
                {"recursion_limit": 999999},
            )
            last_step_string = final_results["messages"][-1].content
            print(last_step_string)

        return final_results["messages"][-1].content

    except Exception as e:
        print(f"Error in example: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": str(e)}


if __name__ == "__main__":
    main()
