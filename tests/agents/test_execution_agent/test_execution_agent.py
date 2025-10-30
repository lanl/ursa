from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary


def test_execution_agent():
    execution_agent = ExecutionAgent(
        llm=init_chat_model(model="openai:gpt-4o-mini"), enable_metrics=True
    )
    problem_string = "Write and execute a minimal python script to print the first 10 integers."
    inputs = {
        "messages": [HumanMessage(content=problem_string)],
        "workspace": Path(".ursa/test-execution-agent"),
    }
    result = execution_agent.invoke(
        inputs,
        config={"configurable": {"thread_id": execution_agent.thread_id}},
    )
    result["messages"][-1].pretty_print()
