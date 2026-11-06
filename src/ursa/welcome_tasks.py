"""Curated starter tasks for the URSA welcome screen.

Each group pairs an agent (or workflow) with a small library of example
tasks. The dashboard welcome screen chooses one task from each group when
it loads and lets users rotate through the examples without leaving the
page.

Add new examples by appending to an existing group's ``tasks``, or add a
whole new group to ``WELCOME_TASK_LIBRARY``. The data is plain Python with
no dashboard dependencies, so other parts of URSA (for example, the TUI)
can reuse the same library later.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "WelcomeTask",
    "WelcomeTaskGroup",
    "WELCOME_TASK_LIBRARY",
    "dashboard_payload",
]


@dataclass(frozen=True)
class WelcomeTask:
    """One example prompt a user can send directly to an agent."""

    title: str
    prompt: str


@dataclass(frozen=True)
class WelcomeTaskGroup:
    """Example tasks for one agent or workflow."""

    agent_id: str
    label: str
    purpose: str
    tasks: tuple[WelcomeTask, ...]

    def payload(self) -> dict[str, object]:
        """JSON-ready mapping with the dashboard client's camelCase keys."""
        return {
            "agentId": self.agent_id,
            "label": self.label,
            "purpose": self.purpose,
            "tasks": [
                {"title": task.title, "prompt": task.prompt}
                for task in self.tasks
            ],
        }


WELCOME_TASK_LIBRARY: tuple[WelcomeTaskGroup, ...] = (
    WelcomeTaskGroup(
        agent_id="chat_agent",
        label="Base",
        purpose="Explore and explain",
        tasks=(
            WelcomeTask(
                title="Build intuition for Bayesian optimization",
                prompt=(
                    "Explain Expected Improvement and Upper Confidence "
                    "Bound. Use one concrete optimization example, compare "
                    "how the acquisition functions balance exploration and "
                    "exploitation, and end with practical guidance about "
                    "when to choose each one."
                ),
            ),
            WelcomeTask(
                title="Making a graph for a presentation",
                prompt=(
                    "Make a professional, clean PNG visualizing the "
                    "spacings between the first 10,000 prime numbers. "
                    "Include clear labels and a short Markdown note "
                    "explaining the main patterns visible in the plot."
                ),
            ),
        ),
    ),
    WelcomeTaskGroup(
        agent_id="execution_agent",
        label="Execute",
        purpose="Analyze and create",
        tasks=(
            WelcomeTask(
                title="Benchmark coding algorithms",
                prompt=(
                    "Create a python function that finds the sum of the "
                    "first N positive integers with a for loop. Time how "
                    "long it takes to sum the first 10,000 and print the "
                    "results to the console. Then add a new function that "
                    "computes the same value using the built-in sum "
                    "function, no loops. Compare the timing for these two "
                    "methods on the first 100,000 integers, and check the "
                    "results match. Finally, add a third function that "
                    "uses a static formula the compute the same value. "
                    "Compare the timing for all three methods on the first "
                    "million integers, and check the results match."
                ),
            ),
            WelcomeTask(
                title="Compare EI and UCB experimentally",
                prompt=(
                    "Compare Expected Improvement and Upper Confidence "
                    "Bound for optimizing the six-hump camel function and "
                    "write a report summarizing the results. The report "
                    "should include one professional, clean PNG image "
                    "visualizing the results."
                ),
            ),
        ),
    ),
    WelcomeTaskGroup(
        agent_id="planning_executor_workflow",
        label="Plan + Execute",
        purpose="Plan a larger project",
        tasks=(
            WelcomeTask(
                title="Benchmark approaches for calculating the Nth Fibonacci number",
                prompt=(
                    "Create a single python script to compute the Fibonacci "
                    "number at position 35 in the sequence."
                    "Compute the answer through more than one distinct"
                    "technique, benchmark and compare the approaches then "
                    "explain which one is the best. Produce a reproducible"
                    "code base for the benchmark with clean, clear documentation."
                ),
            ),
            WelcomeTask(
                title="Teach ML through a model comparison",
                prompt=(
                    "Download the Boston Housing Dataset and compare 5 "
                    "different machine-learning predictors on hold-out "
                    "prediction performance. Choose the models to make "
                    "central ideas from machine learning clear to a "
                    "student audience. Then write a presentation geared at "
                    "an undergraduate audience using these results."
                ),
            ),
            WelcomeTask(
                title="Plan and deliver a reproducible data study",
                prompt=(
                    "Find a small public tabular dataset suitable for "
                    "teaching classification. Plan and run a reproducible "
                    "comparison of five models that illuminate different "
                    "machine-learning ideas, then create an "
                    "undergraduate-level presentation with a professional "
                    "results visualization and clear takeaways."
                ),
            ),
        ),
    ),
)


def dashboard_payload() -> list[dict[str, object]]:
    """Build the library in the shape the dashboard client script expects."""
    return [group.payload() for group in WELCOME_TASK_LIBRARY]
