from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from ursa.agents import PlanningAgent
from ursa.observability.timing import render_session_summary

# def test_planning_agent():
planning_agent = PlanningAgent(
    llm=init_chat_model(model="openai:gpt-5-nano"), enable_metrics=True
)

problem_string = "Calculate Pi to 1000 decimal places."
# problem_string = "Create a one step plan for computing 1+1."

inputs = {
    "messages": [HumanMessage(content=problem_string)],
    "reflection_steps": 3,
}

result = planning_agent(inputs)

result["messages"][-2].pretty_print()

result["messages"][-1].pretty_print()

render_session_summary(planning_agent.thread_id)
