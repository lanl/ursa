"""Plan, implement, and benchmark techniques for computing Fibonacci numbers."""

from pathlib import Path

from ursa.cli.config import UrsaConfig
from ursa.observability.timing import render_session_summary
from ursa.util.events import configure_event_logging
from ursa.workflows import PlanningExecutionAgent

configure_event_logging()

index_to_find = 35
problem = (
    f"Create a single Python script to compute the Fibonacci number at position "
    f"{index_to_find}. Compute the answer through more than one distinct "
    "technique, benchmark and compare the approaches, then explain which is best."
)

# The parent owns one checkpointer and one model. Its native planner and executor
# subgraphs inherit both resources and use isolated checkpoint namespaces.
workspace = Path("example_fibonacci_finder")
config = UrsaConfig.user_config()
model = config.llm_model.init_chat_model()
agent = PlanningExecutionAgent(
    llm=model,
    agent_name="fibonnaci_bot",
    enable_metrics=True,
    workspace=workspace,
)

agent.invoke(problem)
render_session_summary(agent.thread_id)
agent.close()
