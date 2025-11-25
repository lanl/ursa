from math import sqrt
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool

from ursa.agents import ExecutionAgent
from ursa.observability.timing import render_session_summary


def test_execution_agent():
    execution_agent = ExecutionAgent(
        llm=init_chat_model(model="openai:gpt-5-nano")
    )
    problem_string = "Write and execute a minimal python script to print the first 10 integers."
    inputs = {
        "messages": [HumanMessage(content=problem_string)],
        "workspace": Path(".ursa/test-execution-agent"),
    }
    result = execution_agent(inputs)
    result["messages"][-1].pretty_print()
    render_session_summary(execution_agent.thread_id)


def test_execution_agent_with_extra_tools():
    execution_agent = ExecutionAgent(
        llm=init_chat_model(model="openai:gpt-5-nano"),
        extra_tools=[do_magic],
    )
    problem = (
        "Do magic with the integers 3 and 4. "
        "Don't give me verbose output. "
        "Don't provide a summary. "
        "Just give me the answer as a single float."
    )
    inputs = {
        "messages": [HumanMessage(content=problem)],
        "workspace": Path(".ursa/test-execution-agent"),
    }
    result = execution_agent(inputs)
    (msg := result["messages"][-1]).pretty_print()
    render_session_summary(execution_agent.thread_id)
    assert "5.0" in msg.content


@tool
def do_magic(a: int, b: int) -> float:
    """Do magic with integers a and b.

    Args:
        a: first integer
        b: second integer
    """
    return sqrt(a**2 + b**2)
