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
            name="llm_api_key",
            title="LLM API Key",
            description="API key (if required).",
            type="string",
            required=False,
            default="",
            advanced=True,
            hidden=True,
            source=ParamSource.llm,
            target="api_key",
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
        AgentParam(
            name="llm_max_tokens",
            title="Max Tokens",
            description="Max tokens for the response.",
            type="integer",
            required=False,
            default=4096,
            advanced=True,
            source=ParamSource.llm,
            target="max_tokens",
            constraints=ParamConstraint(minimum=1),
        ),
        AgentParam(
            name="llm_temperature",
            title="Temperature",
            description="Sampling temperature.",
            type="number",
            required=False,
            default=0.2,
            advanced=True,
            source=ParamSource.llm,
            target="temperature",
            constraints=ParamConstraint(minimum=0.0, maximum=2.0),
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


def _planning_executor_workflow_builder() -> Callable[
    [Any, dict[str, Any]], AgentAdapter
]:
    """Build adapter for PlanningExecutorWorkflow.

    The workflow composes a PlanningAgent + ExecutionAgent. We create both using the
    same LLM config and workspace.
    """

    def build_adapter(llm: Any, agent_init: dict[str, Any]) -> AgentAdapter:
        if llm is None:
            raise ValueError(
                "PlanningExecutorWorkflow requires an enabled LLM configuration (llm.disabled=false)."
            )

        PlanningAgent = _lazy_class("ursa.agents.planning_agent.PlanningAgent")
        ExecutionAgent = _lazy_class(
            "ursa.agents.execution_agent.ExecutionAgent"
        )

        # User request had a typo (worksflows). The correct module is ursa.workflows.
        try:
            PlanningExecutorWorkflow = _lazy_class(
                "ursa.workflows.planning_execution_workflow.PlanningExecutorWorkflow"
            )
        except Exception:
            PlanningExecutorWorkflow = _lazy_class(
                "ursa.worksflows.planning_execution_workflow.PlanningExecutorWorkflow"
            )

        def agent_factory(workspace_dir: Path, _inputs: Any):
            planner_init = dict(agent_init)
            executor_init = dict(agent_init)

            # Split init kwargs between the two agents to avoid unexpected-kw errors.
            for k in [
                "tokens_before_summarize",
                "messages_to_keep",
                "safe_codes",
                "log_state",
            ]:
                planner_init.pop(k, None)
            executor_init.pop("max_reflection_steps", None)

            planner = PlanningAgent(
                llm=llm, workspace=str(workspace_dir), **planner_init
            )
            executor = ExecutionAgent(
                llm=llm, workspace=str(workspace_dir), **executor_init
            )
            return PlanningExecutorWorkflow(
                planner=planner,
                executor=executor,
                workspace=str(workspace_dir),
            )

        # IMPORTANT: do not redirect stdout/stderr inside the adapter; the dashboard
        # runner captures worker stdout/stderr directly.
        return DirectInvokeAdapter(agent_factory)

    return build_adapter


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
            display_name="Chat Agent",
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
            display_name="Planning Agent",
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
            agent_id="execution_agent",
            display_name="Execution Agent",
            description="Tool-using agent that can write/edit files, run shell commands, and search the web/arXiv/OSTI.",
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
            display_name="Planning + Execution Workflow",
            description="Runs a PlanningAgent to break the task into steps, then an ExecutionAgent to execute each step. Best for longer, complex tasks.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(title="Task"),
                # Reuse common agent_init knobs (applies to both planner + executor)
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

### register(
###     AgentEntry(
###         spec=AgentSpec(
###             agent_id="web_search_agent",
###             display_name="Web Search Agent",
###             description="Searches the web, downloads pages, and optionally summarizes/RAGs across results.",
###             capabilities=AgentCapabilities(
###                 supports_streaming=False,
###                 supports_cancellation=False,
###                 produces_artifacts=True,
###             ),
###             parameters=
###                 [
###                     _prompt_param(title="Query"),
###                     AgentParam(
###                         name="max_results",
###                         title="Max results",
###                         description="Maximum number of items to fetch.",
###                         type="integer",
###                         required=False,
###                         default=5,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="max_results",
###                         constraints=ParamConstraint(minimum=1, maximum=50),
###                     ),
###                     AgentParam(
###                         name="download",
###                         title="Download",
###                         description="Download/scrape results (otherwise rely on cache).",
###                         type="boolean",
###                         required=False,
###                         default=True,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="download",
###                     ),
###                     AgentParam(
###                         name="summarize",
###                         title="Summarize",
###                         description="Generate summaries for each item.",
###                         type="boolean",
###                         required=False,
###                         default=True,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="summarize",
###                     ),
###                     AgentParam(
###                         name="process_images",
###                         title="Process images (PDF)",
###                         description="Extract and describe images from PDFs.",
###                         type="boolean",
###                         required=False,
###                         default=True,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="process_images",
###                     ),
###                     AgentParam(
###                         name="num_threads",
###                         title="Threads",
###                         description="Parallel download/summarization workers.",
###                         type="integer",
###                         required=False,
###                         default=4,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="num_threads",
###                         constraints=ParamConstraint(minimum=1, maximum=32),
###                     ),
###                     AgentParam(
###                         name="user_agent",
###                         title="HTTP User-Agent",
###                         description="User-Agent string for requests.",
###                         type="string",
###                         required=False,
###                         default="Mozilla/5.0",
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="user_agent",
###                     ),
###                 ]
###                 + _common_llm_params()
###                 + _runner_params(),
###             tags=["search", "acquisition"],
###         ),
###         build_adapter=_baseagent_adapter_builder(
###             "ursa.agents.acquisition_agents.WebSearchAgent"
###         ),
###         build_inputs=lambda p: p["prompt"],
###     )
### )

### register(
###     AgentEntry(
###         spec=AgentSpec(
###             agent_id="arxiv_agent",
###             display_name="arXiv Agent",
###             description="Searches arXiv, downloads PDFs, and summarizes across papers.",
###             capabilities=AgentCapabilities(
###                 supports_streaming=False,
###                 supports_cancellation=False,
###                 produces_artifacts=True,
###             ),
###             parameters=
###                 [
###                     _prompt_param(title="Query"),
###                     AgentParam(
###                         name="max_results",
###                         title="Max results",
###                         description="Maximum number of papers to fetch.",
###                         type="integer",
###                         required=False,
###                         default=3,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="max_results",
###                         constraints=ParamConstraint(minimum=1, maximum=50),
###                     ),
###                     AgentParam(
###                         name="download",
###                         title="Download",
###                         description="Download PDFs (otherwise rely on cache).",
###                         type="boolean",
###                         required=False,
###                         default=True,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="download",
###                     ),
###                     AgentParam(
###                         name="process_images",
###                         title="Process images (PDF)",
###                         description="Extract and describe images from PDFs.",
###                         type="boolean",
###                         required=False,
###                         default=True,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="process_images",
###                     ),
###                 ]
###                 + _common_llm_params()
###                 + _runner_params(),
###             tags=["search", "papers"],
###         ),
###         build_adapter=_baseagent_adapter_builder(
###             "ursa.agents.acquisition_agents.ArxivAgent"
###         ),
###         build_inputs=lambda p: p["prompt"],
###     )
### )

### register(
###     AgentEntry(
###         spec=AgentSpec(
###             agent_id="osti_agent",
###             display_name="OSTI Agent",
###             description="Searches OSTI.gov records, downloads reports when available, and summarizes.",
###             capabilities=AgentCapabilities(
###                 supports_streaming=False,
###                 supports_cancellation=False,
###                 produces_artifacts=True,
###             ),
###             parameters=
###                 [
###                     _prompt_param(title="Query"),
###                     AgentParam(
###                         name="max_results",
###                         title="Max results",
###                         description="Maximum number of records to fetch.",
###                         type="integer",
###                         required=False,
###                         default=5,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="max_results",
###                         constraints=ParamConstraint(minimum=1, maximum=50),
###                     ),
###                     AgentParam(
###                         name="api_base",
###                         title="OSTI API base",
###                         description="Base URL for OSTI API.",
###                         type="string",
###                         required=False,
###                         default="https://www.osti.gov/api/v1/records",
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="api_base",
###                     ),
###                 ]
###                 + _common_llm_params()
###                 + _runner_params(),
###             tags=["search", "papers"],
###         ),
###         build_adapter=_baseagent_adapter_builder(
###             "ursa.agents.acquisition_agents.OSTIAgent"
###         ),
###         build_inputs=lambda p: p["prompt"],
###     )
### )
###
### register(
###     AgentEntry(
###         spec=AgentSpec(
###             agent_id="rag_agent",
###             display_name="RAG Agent",
###             description="Retrieval-Augmented Generation over ingested documents stored in a per-workspace vectorstore.",
###             capabilities=AgentCapabilities(
###                 supports_streaming=False,
###                 supports_cancellation=False,
###                 produces_artifacts=True,
###             ),
###             parameters=
###                 [
###                     _prompt_param(title="Question"),
###                     AgentParam(
###                         name="return_k",
###                         title="Top-K",
###                         description="How many chunks to retrieve.",
###                         type="integer",
###                         required=False,
###                         default=10,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="return_k",
###                         constraints=ParamConstraint(minimum=1, maximum=100),
###                     ),
###                     AgentParam(
###                         name="chunk_size",
###                         title="Chunk size",
###                         description="Text chunk size for splitting.",
###                         type="integer",
###                         required=False,
###                         default=1000,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="chunk_size",
###                         constraints=ParamConstraint(minimum=100, maximum=5000),
###                     ),
###                     AgentParam(
###                         name="chunk_overlap",
###                         title="Chunk overlap",
###                         description="Overlap between chunks.",
###                         type="integer",
###                         required=False,
###                         default=200,
###                         advanced=True,
###                         source=ParamSource.agent_init,
###                         target="chunk_overlap",
###                         constraints=ParamConstraint(minimum=0, maximum=1000),
###                     ),
###                 ]
###                 + _common_llm_params()
###                 + _runner_params(),
###             tags=["rag"],
###         ),
###         build_adapter=_baseagent_adapter_builder("ursa.agents.rag_agent.RAGAgent"),
###         build_inputs=lambda p: p["prompt"],
###     )
### )

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="hypothesizer_agent",
            display_name="Hypothesizer Agent",
            description="Iteratively generates and refines hypotheses with web searches.",
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
                    description="Number of hypothesis/refinement loops.",
                    type="integer",
                    required=False,
                    default=3,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="max_iterations",
                    constraints=ParamConstraint(minimum=1, maximum=20),
                ),
            ]
            + _common_llm_params()
            + _runner_params(),
            tags=["research"],
        ),
        build_adapter=_baseagent_adapter_builder(
            "ursa.agents.hypothesizer_agent.HypothesizerAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)


# -----------------------------
# Demo agents (internal / hidden by default)
# -----------------------------

# These are used for smoke-testing and troubleshooting the dashboard itself.
# They are *not* listed in the UI unless URSA_DASHBOARD_INCLUDE_DEMO_AGENTS=1.


def _demo_adapter_builder(class_path: str):
    def build_adapter(_llm: Any, agent_init: dict[str, Any]) -> AgentAdapter:
        cls = _lazy_class(class_path)

        def agent_factory(workspace_dir: Path, _inputs: Any):
            return cls(workspace=str(workspace_dir), **agent_init)

        return DirectInvokeAdapter(agent_factory)

    return build_adapter


register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="demo_quick",
            display_name="Demo: Quick",
            description="Writes a couple of small artifacts and exits. Does not require any LLM credentials.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=False,
                produces_artifacts=True,
            ),
            parameters=[_prompt_param()] + _runner_params(),
            tags=["demo"],
        ),
        build_adapter=_demo_adapter_builder(
            "ursa_dashboard.demo_agents.DemoQuickAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)

register(
    AgentEntry(
        spec=AgentSpec(
            agent_id="demo_slow",
            display_name="Demo: Slow (cancel/stream)",
            description="Prints progress and updates an artifact over time. Useful to demo streaming logs and cancellation.",
            capabilities=AgentCapabilities(
                supports_streaming=False,
                supports_cancellation=True,
                produces_artifacts=True,
            ),
            parameters=[
                _prompt_param(),
                AgentParam(
                    name="steps",
                    title="Steps",
                    description="How many progress steps to run.",
                    type="integer",
                    required=False,
                    default=60,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="steps",
                    constraints=ParamConstraint(minimum=1, maximum=600),
                ),
                AgentParam(
                    name="sleep_s",
                    title="Sleep per step (seconds)",
                    description="Delay per step.",
                    type="number",
                    required=False,
                    default=0.25,
                    advanced=True,
                    source=ParamSource.agent_init,
                    target="sleep_s",
                    constraints=ParamConstraint(minimum=0.0, maximum=10.0),
                ),
            ]
            + _runner_params(),
            tags=["demo"],
        ),
        build_adapter=_demo_adapter_builder(
            "ursa_dashboard.demo_agents.DemoSlowAgent"
        ),
        build_inputs=lambda p: p["prompt"],
    )
)
