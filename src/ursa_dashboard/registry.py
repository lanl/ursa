from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .adapters import (
    AgentAdapter,
    BaseAgentInProcessAdapter,
    DirectInvokeAdapter,
)
from .models import (
    AgentCapabilities,
    AgentParam,
    AgentSpec,
    ParamConstraint,
    ParamSource,
)


@dataclass(frozen=True)
class AgentEntry:
    spec: AgentSpec
    # Build an adapter for this agent.
    build_adapter: Callable[[Any, dict[str, Any]], AgentAdapter]
    # Convert UI run-input params into the object passed to adapter.invoke(...).
    build_inputs: Callable[[dict[str, Any]], Any]


REGISTRY: dict[str, AgentEntry] = {}


def _common_llm_params() -> list[AgentParam]:
    return [
        AgentParam(
            name="llm_base_url",
            title="LLM Base URL",
            description="OpenAI-compatible base URL.",
            type="string",
            required=False,
            default="http://127.0.0.1:8000/v1",
            advanced=True,
            source=ParamSource.llm,
            target="base_url",
        ),
        AgentParam(
            name="llm_model",
            title="LLM Model",
            description="Model name.",
            type="string",
            required=False,
            default="gpt-5-mini",
            advanced=True,
            source=ParamSource.llm,
            target="model",
        ),
    ]


def _runner_params() -> list[AgentParam]:
    return [
        AgentParam(
            name="timeout_seconds",
            title="Timeout (seconds)",
            description="Force-stop the run after this many seconds.",
            type="integer",
            required=False,
            default=3600,
            advanced=True,
            source=ParamSource.runner,
            target="timeout_seconds",
            constraints=ParamConstraint(minimum=1),
        )
    ]


def _prompt_param(*, title: str = "Prompt") -> AgentParam:
    return AgentParam(
        name="prompt",
        title=title,
        description="What you want the agent to do.",
        type="string",
        required=True,
        source=ParamSource.run_input,
        target="prompt",
        constraints=ParamConstraint(minLength=1),
    )


def _lazy_class(class_path: str):
    mod_name, cls_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _baseagent_adapter_builder(
    class_path: str, *, supports_streaming: bool = False
):
    """Return a build_adapter(llm, agent_init_kwargs) closure."""

    def build_adapter(llm: Any, agent_init: dict[str, Any]) -> AgentAdapter:
        cls = _lazy_class(class_path)

        def agent_factory(workspace_dir: Path, _inputs: Any):
            # Most URSA agents accept workspace via BaseAgent(**kwargs).
            return cls(llm=llm, workspace=str(workspace_dir), **agent_init)

        return BaseAgentInProcessAdapter(
            agent_factory,
            supports_streaming=supports_streaming,
        )

    return build_adapter


def _think_plan_execute_workflow_builder() -> Callable[
    [Any, dict[str, Any]], AgentAdapter
]:
    """Build the one-runtime planning/execution BaseAgent adapter."""

    return _baseagent_adapter_builder(
        "ursa.workflows.think_plan_execute.ThinkPlanningExecutionAgent"
    )

def _planning_executor_workflow_builder() -> Callable[
    [Any, dict[str, Any]], AgentAdapter
]:
    """Build the one-runtime planning/execution BaseAgent adapter."""

    return _baseagent_adapter_builder(
        "ursa.workflows.planning_execution_workflow.PlanningExecutionAgent"
    )


def register(entry: AgentEntry) -> None:
    agent_id = entry.spec.agent_id
    if agent_id in REGISTRY:
        raise ValueError(f"Duplicate agent_id registered: {agent_id}")
    REGISTRY[agent_id] = entry


# -----------------------------
# Built-in registry entries
# -----------------------------

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="chat_agent",
            display_name="Chat + Execute",
            description="General chat interface to an LLM.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=False,
            ),
            parameters=[_prompt_param()]
            + _common_llm_params()
            + _runner_params(),
            tags=["general"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.chat_agent.ChatAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="planning_agent",
            display_name="Plan",
            description="Creates a step-by-step plan using structured output and optional self-reflection.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=False,
            ),
            parameters=[
                _prompt_param(title="Goal"),
                AgentParam(
                    name="max_reflection_steps",
                    title="Max reflection steps",
                    description="Number of reflection passes to improve the plan.",
                    type="integer",
                    required=False,
                    default=1,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="max_reflection_steps",
                    constraints=ParamConstraint(minimum=0, maximum=10),
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["planning"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.planning_agent.PlanningAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="prompting_agent",
            display_name="Prompt Refinement",
            description="Iterates with the user to refine a rough request into clean, self-contained instructions for a downstream agentic workflow. It can reference available ChatAgent and ExecutionAgent tools when drafting prompts; web/arXiv/OSTI tools are reflected only when web access is enabled.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=False,
            ),
            parameters=[
                _prompt_param(title="Prompt to refine"),
                AgentParam(
                    name="use_web",
                    title="Include web tool context",
                    description="Include web/arXiv/OSTI tools in the downstream-tool context used while drafting the prompt.",
                    type="boolean",
                    required=False,
                    default=False,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="use_web",
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["planning", "prompting"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.prompting_agent.PromptingAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="execution_agent",
            display_name="Execution + Reflect",
            description="Tool-using agent that can write/edit files and run shell commands. Web/arXiv/OSTI search tools are available only when the dashboard is started with --use-web or agent_init.use_web=true.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(),
                AgentParam(
                    name="tokens_before_summarize",
                    title="Tokens before summarize",
                    description="Conversation token budget before context is summarized.",
                    type="integer",
                    required=False,
                    default=50000,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="tokens_before_summarize",
                    constraints=ParamConstraint(minimum=1000),
                ),
                AgentParam(
                    name="messages_to_keep",
                    title="Messages to keep",
                    description="How many recent messages to keep verbatim when summarizing.",
                    type="integer",
                    required=False,
                    default=20,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="messages_to_keep",
                    constraints=ParamConstraint(minimum=0, maximum=200),
                ),
                AgentParam(
                    name="safe_codes",
                    title="Safe code types",
                    description="Code languages that can be executed by the shell tool.",
                    type="array",
                    required=False,
                    default=["python", "julia"],
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="safe_codes",
                ),
                AgentParam(
                    name="log_state",
                    title="Log state",
                    description="Emit extra internal state logs to stdout.",
                    type="boolean",
                    required=False,
                    default=False,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="log_state",
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["tools"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.execution_agent.ExecutionAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="planning_executor_workflow",
            display_name="Plan -> Execute",
            description="Uses one persistent planning/execution agent with native planner and executor subgraphs. Best for longer, complex tasks. Web/arXiv/OSTI tools are opt-in via --use-web or agent_init.use_web=true.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(title="Task"),
                # Planner and executor settings belong to the same agent runtime.
                AgentParam(
                    name="max_reflection_steps",
                    title="Max reflection steps",
                    description="Number of reflection passes for the planner.",
                    type="integer",
                    required=False,
                    default=1,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="max_reflection_steps",
                    constraints=ParamConstraint(minimum=0, maximum=10),
                ),
                AgentParam(
                    name="tokens_before_summarize",
                    title="Tokens before summarize",
                    description="Conversation token budget before context is summarized (executor).",
                    type="integer",
                    required=False,
                    default=50000,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="tokens_before_summarize",
                    constraints=ParamConstraint(minimum=1000),
                ),
                AgentParam(
                    name="messages_to_keep",
                    title="Messages to keep",
                    description="How many recent messages to keep verbatim when summarizing (executor).",
                    type="integer",
                    required=False,
                    default=20,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="messages_to_keep",
                    constraints=ParamConstraint(minimum=0, maximum=200),
                ),
                AgentParam(
                    name="safe_codes",
                    title="Safe code types",
                    description="Code languages that can be executed by the shell tool (executor).",
                    type="array",
                    required=False,
                    default=["python", "julia"],
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="safe_codes",
                ),
                AgentParam(
                    name="log_state",
                    title="Log state",
                    description="Emit extra internal state logs to stdout (executor).",
                    type="boolean",
                    required=False,
                    default=False,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="log_state",
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["workflow", "planning", "tools"],
        ),
        build_adapter=_planning_executor_workflow_builder(),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="hypothesize-plan-execute",
            display_name="Hypothesize -> Plan -> Execute",
            description="Uses one persistent building a hypothesis space and then doing planning/execution with native planner and executor subgraphs. Best for longer, complex tasks. Web/arXiv/OSTI tools are opt-in via --use-web or agent_init.use_web=true.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(title="Task"),
                # Planner and executor settings belong to the same agent runtime.
                AgentParam(
                    name="max_reflection_steps",
                    title="Max reflection steps",
                    description="Number of reflection passes for the planner.",
                    type="integer",
                    required=False,
                    default=1,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="max_reflection_steps",
                    constraints=ParamConstraint(minimum=0, maximum=10),
                ),
                AgentParam(
                    name="tokens_before_summarize",
                    title="Tokens before summarize",
                    description="Conversation token budget before context is summarized (executor).",
                    type="integer",
                    required=False,
                    default=50000,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="tokens_before_summarize",
                    constraints=ParamConstraint(minimum=1000),
                ),
                AgentParam(
                    name="messages_to_keep",
                    title="Messages to keep",
                    description="How many recent messages to keep verbatim when summarizing (executor).",
                    type="integer",
                    required=False,
                    default=20,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="messages_to_keep",
                    constraints=ParamConstraint(minimum=0, maximum=200),
                ),
                AgentParam(
                    name="safe_codes",
                    title="Safe code types",
                    description="Code languages that can be executed by the shell tool (executor).",
                    type="array",
                    required=False,
                    default=["python", "julia"],
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="safe_codes",
                ),
                AgentParam(
                    name="log_state",
                    title="Log state",
                    description="Emit extra internal state logs to stdout (executor).",
                    type="boolean",
                    required=False,
                    default=False,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="log_state",
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["workflow", "planning", "tools"],
        ),
        build_adapter=_think_plan_execute_workflow_builder(),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="deep_review_agent",
            display_name="Propose -> Critique -> Adversarial Review",
            description="Iteratively drafts, critiques, and refines a solution with adversarial review. Workspace file tools are available by default; web/arXiv/OSTI search tools are opt-in via use_web.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(title="Research question"),
                AgentParam(
                    name="max_iterations",
                    title="Max iterations",
                    description="Number of draft/critique/refinement loops.",
                    type="integer",
                    required=False,
                    default=3,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="max_iterations",
                    constraints=ParamConstraint(minimum=1, maximum=20),
                ),
                AgentParam(
                    name="use_web",
                    title="Enable web search tools",
                    description="Expose web/arXiv/OSTI search tools to the autonomous reviewer. When false, Deep Review cannot perform web searches.",
                    type="boolean",
                    required=False,
                    default=False,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="use_web",
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["research", "review"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.deep_review_agent.DeepReviewAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="hypothesizer_agent",
            display_name="Hypothesize",
            description="Maintains a persistent hypothesis space in an experience artifact for reuse by other agents.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[_prompt_param(title="Question, evidence, or update")]
            + _common_llm_params()
            + _runner_params(),
            tags=["research", "hypotheses", "experiences"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.hypothesizer_agent.HypothesizerAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

