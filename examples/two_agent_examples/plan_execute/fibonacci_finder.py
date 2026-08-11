"""Plan, implement, and benchmark techniques for computing Fibonacci numbers."""

from pathlib import Path
from uuid import uuid4

from langchain.chat_models import init_chat_model

from ursa.observability.timing import render_session_summary
from ursa.util import Checkpointer
from ursa.util.events import configure_event_logging
from ursa.workflows import PlanningExecutionAgent

configure_event_logging()

thread_id = "run-" + uuid4().hex[:8]
workspace = Path("example_fibonacci_finder")
index_to_find = 35
problem = (
    f"Create a single Python script to compute the Fibonacci number at position "
    f"{index_to_find}. Compute the answer through more than one distinct "
    "technique, benchmark and compare the approaches, then explain which is best."
)

# The parent owns one checkpointer and one model. Its native planner and executor
# subgraphs inherit both resources and use isolated checkpoint namespaces.
checkpointer = Checkpointer.from_workspace(workspace)
model = init_chat_model(model="openai:o4-mini")
agent = PlanningExecutionAgent(
    llm=model,
    enable_metrics=True,
    thread_id=thread_id,
    workspace=workspace,
    checkpointer=checkpointer,
)

agent.invoke(problem)
render_session_summary(thread_id)
agent.close()
