"""Base agent class providing telemetry, configuration, and execution abstractions.

This module defines the BaseAgent abstract class, which serves as the foundation for all
agent implementations in the Ursa framework. It provides:

- Standardized initialization with LLM configuration
- Telemetry and metrics collection
- Thread and checkpoint management
- Input normalization and validation
- Execution flow control with invoke/stream methods
- Graph integration utilities for LangGraph compatibility
- Runtime enforcement of the agent interface contract

Agents built on this base class benefit from consistent behavior, observability, and
integration capabilities while only needing to implement the core _invoke method.
"""

import asyncio
import inspect
import re
import sqlite3
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    final,
)

from langchain.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain.tools import BaseTool, ToolException
from langchain_core.load import dumps
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.runnables.config import merge_configs
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import (
    CompiledStateGraph,
    StateGraph,
    coerce_to_runnable,
)
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from langgraph.store.base import BaseStore
from langgraph.store.sqlite import SqliteStore
from langgraph.store.sqlite.aio import AsyncSqliteStore
from langgraph.types import Overwrite

from ursa.observability.timing import (
    Telemetry,  # for timing / telemetry / metrics
)
from ursa.security import (
    DEFAULT_GROUP_NAME,
    enforce_model_group_policy,
    group_agents_dir,
    group_root_dir,
    validate_group_name,
)
from ursa.util import Checkpointer
from ursa.util.events import DEFAULT_EVENT_LOGGING_HANDLER, AgentEvents

InputLike = str | Mapping[str, Any]
TState = TypeVar("TState", bound=Mapping[str, Any])


@dataclass(frozen=True, kw_only=True)
class AgentContext:
    """Immutable context provided during graph execution"""

    llm: BaseChatModel
    """ Chat model for use during tool calls """

    workspace: Path
    """ Workspace path for the agent """

    agent_name: str
    """ Name for persisting the agent """

    group: str
    """ Name for the group for the agent to operate in. Controls data and endpoint availability"""

    den: Path
    """ Path to the persistent storage for the agent"""

    tool_character_limit: int = 30000
    """ Suggested limit on tool call responses """

    rag_tools: tuple[str, ...] = ()
    """Persisted URSA RAG collections bound to this agent as tools."""

    rag_tool_group: str | None = None
    """Group used to resolve persisted RAG tool collections."""

    rag_tool_return_k: int = 10
    """Number of chunks retrieved by persisted RAG tools."""


def _to_snake(s: str) -> str:
    """Convert a string to snake_case format.

    This function transforms various string formats (CamelCase, PascalCase, etc.) into
    snake_case. It handles special cases like acronyms at the beginning of strings
    (e.g., "RAGAgent" becomes "rag_agent") and replaces hyphens and spaces with
    underscores.

    Args:
        s: The input string to convert to snake_case.

    Returns:
        The snake_case version of the input string.
    """
    s = re.sub(
        r"^([A-Z]{2,})([A-Z][a-z])",
        lambda m: m.group(1)[0] + m.group(1)[1:].lower() + m.group(2),
        str(s),
    )  # RAGAgent -> RagAgent
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)  # CamelCase -> snake_case
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


class BaseAgent(Generic[TState], ABC):
    """Abstract base class for all agent implementations in the Ursa framework.

    BaseAgent provides a standardized foundation for building LLM-powered agents with
    built-in telemetry, configuration management, and execution flow control. It handles
    common tasks like input normalization, thread management, metrics collection, and
    LangGraph integration.

    Subclasses only need to implement the _invoke method to define their core
    functionality, while inheriting standardized invocation patterns, telemetry, and
    graph integration capabilities. The class enforces a consistent interface through
    runtime checks that prevent subclasses from overriding critical methods like
    invoke().

    The agent supports both direct invocation with inputs and streaming responses, with
    automatic tracking of token usage, execution time, and other metrics. It also
    provides utilities for integrating with LangGraph through node wrapping and
    configuration.

    Subclass Inheritance Guidelines:
        - Must Override: _invoke() - Define your agent's core functionality
        - Can Override: _stream() - Enable streaming support
                        _normalize_inputs() - Customize input handling
                        Various helper methods (_default_node_tags, etc.)
        - Never Override: invoke() - Final method with runtime enforcement
                          stream() - Handles telemetry and delegates to _stream
                          __call__() - Delegates to invoke
                          Other public methods (build_config, write_state, add_node)

    To create a custom agent, inherit from this class and implement the _invoke method:

    ```python
    class MyAgent(BaseAgent):
        def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
            # Process inputs and return results
            ...
    ```
    """

    # This will be shared across all BaseAgent instances.
    _invoke_depth: int = 0

    _TELEMETRY_KW = {
        "raw_debug",
        "save_json",
        "save_otel",
        "metrics_path",
        "save_raw_snapshot",
        "save_raw_records",
        "otel_endpoint",
        "otel_headers",
    }

    _CONTROL_KW = {"config", "recursion_limit", "tags", "metadata", "callbacks"}

    state_type: type[TState] = dict

    def __init__(
        self,
        llm: BaseChatModel,
        workspace: Optional[Path] = None,
        agent_name: Optional[str] = None,
        group: Optional[str] = "default",
        checkpointer: Optional[BaseCheckpointSaver] = None,
        enable_metrics: bool = True,
        metrics_dir: str = "ursa_metrics",  # dir to save metrics, with a default
        autosave_metrics: bool = True,
        otel_metrics: bool = False,
        thread_id: Optional[str] = None,
        tokens_before_summarize: int = 50000,
        messages_to_keep: int = 20,
        max_single_tool_message_tokens: int = 100000,
        rag_tools: str | Sequence[str] | None = None,
        rag_tool_group: str | None = None,
        rag_tool_embedding: Embeddings | None = None,
        rag_tool_return_k: int = 10,
    ):
        """Initializes the base agent with a language model and optional configurations.

        Args:
            llm: a BaseChatModel instance.
            checkpointer: Optional checkpoint saver for persisting agent state.
            enable_metrics: Whether to collect performance and usage metrics.
            metrics_dir: Directory path where metrics will be saved.
            autosave_metrics: Whether to automatically save metrics to disk.
            thread_id: Unique identifier for this agent instance. Generated if not
                       provided.
        """
        self.llm: BaseChatModel = llm
        self.workspace = Path(workspace or ".")
        self.agent_name = agent_name
        self.group = validate_group_name(group)
        self.tokens_before_summarize = tokens_before_summarize
        self.messages_to_keep = messages_to_keep
        self.max_single_tool_message_tokens = max_single_tool_message_tokens
        from ursa.rag.persistence import normalize_rag_tool_names

        self.rag_tools = tuple(normalize_rag_tool_names(rag_tools))
        self.rag_tool_group = rag_tool_group
        self.rag_tool_embedding = rag_tool_embedding
        self.rag_tool_return_k = rag_tool_return_k
        enforce_model_group_policy(self.llm, self.group)
        if (
            not group_root_dir(self.group).exists()
            and self.group != DEFAULT_GROUP_NAME
        ):
            raise ValueError(
                (
                    f"Group '{self.group}' does not exist. "
                    f"Please use `ursa create-group {self.group} <group_config_file>` to create"
                )
            )
        set_checkpointer = True if checkpointer else False
        set_name = True if agent_name else False
        self.checkpointer = checkpointer
        self._checkpointer_was_provided = set_checkpointer
        self._async_checkpointer: BaseCheckpointSaver | None = None
        self._async_storage: BaseStore | None = None
        self._async_compiled_graph: CompiledStateGraph | None = None
        persist_agent = (agent_name is not None) or set_checkpointer
        den_name = str(self.workspace)
        if persist_agent:
            if set_checkpointer:
                if set_name:
                    print(
                        (
                            "[WARNING]: Both checkpointer and den persistence set."
                            " Using checkpointer, but only use one in the future."
                        )
                    )
                self.den = self.workspace
            else:
                den_name = Path(self.group) / "agents" / agent_name
                self.den = group_agents_dir(self.group) / agent_name
                if not self.den.exists():
                    print(f"[Agent Created]: {den_name}")
                self.checkpointer = Checkpointer.from_workspace(self.den)
        else:
            # Keep current behavior if the user is not persisting.
            self.den = self.workspace
        self.thread_id = thread_id or "ursa"
        self.telemetry = Telemetry(
            enable=enable_metrics,
            output_dir=self.den.joinpath(metrics_dir),
            save_json_default=autosave_metrics,
        )

        self.workspace.mkdir(exist_ok=True, parents=True)
        self.den.mkdir(exist_ok=True, parents=True)

    @property
    def name(self) -> str:
        """Agent name."""
        return self.__class__.__name__

    @property
    def context(self) -> AgentContext:
        """Immutable run-scoped information provided to the Agent's graph"""
        return AgentContext(
            llm=self.llm,
            workspace=self.workspace,
            agent_name=self.agent_name,
            group=self.group,
            den=self.den,
            rag_tools=self.rag_tools,
            rag_tool_group=self.rag_tool_group,
            rag_tool_return_k=self.rag_tool_return_k,
        )

    @staticmethod
    def _dedupe_callbacks(callbacks: list[Any]) -> list[Any]:
        """Preserve callback order while removing duplicate handler instances."""
        deduped: list[Any] = []
        seen: set[int] = set()
        for callback in callbacks:
            marker = id(callback)
            if marker in seen:
                continue
            deduped.append(callback)
            seen.add(marker)
        return deduped

    def events(self, config: RunnableConfig | None = None) -> AgentEvents:
        """Return a helper for emitting structured agent progress events."""
        return AgentEvents(agent=self.name, config=config)

    def add_node(
        self,
        f: Callable[..., Mapping[str, Any]],
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> StateGraph:
        """Add a node to the state graph with token usage tracking.

        This method adds a function as a node to the state graph, wrapping it to track
        token usage during execution. The node is identified by either the provided
        node_name or the function's name.

        Args:
            f: The function to add as a node. Should return a mapping of string keys to
                any values.
            node_name: Optional name for the node. If not provided, the function's name
                will be used.
            agent_name: Optional agent name for tracking. If not provided, the agent's
                name in snake_case will be used.

        Returns:
            The updated StateGraph with the new node added.
        """
        _node_name = node_name or f.__name__
        _agent_name = agent_name or _to_snake(self.name)
        wrapped_node = self._wrap_node(f, _node_name, _agent_name)
        return self.graph.add_node(_node_name, wrapped_node, **kwargs)

    def write_state(self, filename: str, state: dict) -> None:
        """Writes agent state to a JSON file.

        Serializes the provided state dictionary to JSON format and writes it to the
        specified file. The JSON is written with non-ASCII characters preserved.

        Args:
            filename: Path to the file where state will be written.
            state: Dictionary containing the agent state to be serialized.
        """
        json_state = dumps(state, ensure_ascii=False)
        with open(filename, "w") as f:
            f.write(json_state)

    def build_config(self, **overrides) -> dict:
        """Constructs a config dictionary for agent operations with telemetry support.

        This method creates a standardized configuration dictionary that includes thread
        identification, telemetry callbacks, and other metadata needed for agent
        operations. The configuration can be customized through override parameters.

        Args:
            **overrides: Optional configuration overrides that can include keys like
                'recursion_limit', 'configurable', 'metadata', 'tags', etc.

        Returns:
            dict: A complete configuration dictionary with all necessary parameters.
        """
        # Create the base configuration with essential fields.
        base = {
            "configurable": {"thread_id": self.thread_id},
            "metadata": {
                "thread_id": self.thread_id,
                "telemetry_run_id": self.telemetry.context.get("run_id"),
            },
            "tags": [self.name],
            "callbacks": [
                *self.telemetry.callbacks,
                DEFAULT_EVENT_LOGGING_HANDLER,
            ],
        }

        # Try to determine the model name from either direct or nested attributes
        model_name = getattr(self, "llm_model", None) or getattr(
            getattr(self, "llm", None), "model", None
        )

        # Add model name to metadata if available
        if model_name:
            base["metadata"]["model"] = model_name

        # Handle configurable dictionary overrides by merging with base configurable
        if "configurable" in overrides and isinstance(
            overrides["configurable"], dict
        ):
            base["configurable"].update(overrides.pop("configurable"))

        # Handle metadata dictionary overrides by merging with base metadata
        if "metadata" in overrides and isinstance(overrides["metadata"], dict):
            base["metadata"].update(overrides.pop("metadata"))

        # Merge tags from caller-provided overrides, avoid duplicates
        if "tags" in overrides and isinstance(overrides["tags"], list):
            base["tags"] = base["tags"] + [
                t for t in overrides.pop("tags") if t not in base["tags"]
            ]

        if "callbacks" in overrides:
            callbacks = merge_configs(
                {"callbacks": base["callbacks"]},
                {"callbacks": overrides.pop("callbacks")},
            ).get("callbacks")
            if isinstance(callbacks, list):
                callbacks = self._dedupe_callbacks(callbacks)
            base["callbacks"] = callbacks

        # Apply any remaining overrides directly to the base configuration
        base.update(overrides)

        return base

    def _invoke_engine(
        self,
        invoke_method,
        inputs: Optional[InputLike] = None,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_headers: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ):
        BaseAgent._invoke_depth += 1
        invoke_config = dict(config or {})

        try:
            # Start telemetry tracking for the top-level invocation
            if BaseAgent._invoke_depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # Handle the case where inputs are provided as keyword arguments
            if inputs is None:
                # Separate kwargs into input parameters and control parameters
                kw_inputs: dict[str, Any] = {}
                control_kwargs: dict[str, Any] = {}
                for k, v in kwargs.items():
                    if k in self._TELEMETRY_KW or k in self._CONTROL_KW:
                        control_kwargs[k] = v
                    else:
                        kw_inputs[k] = v
                inputs = kw_inputs

                # Only control kwargs remain for further processing
                kwargs = control_kwargs

            # Handle the case where inputs are provided as a positional argument
            else:
                # Ensure no ambiguous keyword arguments are present
                for k in kwargs.keys():
                    if not (k in self._TELEMETRY_KW or k in self._CONTROL_KW):
                        raise TypeError(
                            f"Unexpected keyword argument '{k}'. "
                            "Pass inputs as a single mapping or omit the positional "
                            "inputs and pass them as keyword arguments."
                        )

            # Allow subclasses to normalize or transform the input format
            normalized = self._normalize_inputs(inputs)

            # Delegate to the subclass implementation with the normalized inputs
            # and any control parameters
            return invoke_method(normalized, **invoke_config, **kwargs)

        finally:
            # Clean up the invocation depth tracking
            BaseAgent._invoke_depth -= 1

            # For the top-level invocation, finalize telemetry and generate outputs
            if BaseAgent._invoke_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    otel_endpoint=otel_endpoint,
                    otel_headers=otel_headers,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    async def _ainvoke_engine(
        self,
        invoke_method,
        inputs: Optional[InputLike] = None,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        otel_endpoint: Optional[str] = None,
        otel_headers: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ):
        BaseAgent._invoke_depth += 1
        invoke_config = dict(config or {})

        try:
            if BaseAgent._invoke_depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            if inputs is None:
                kw_inputs: dict[str, Any] = {}
                control_kwargs: dict[str, Any] = {}
                for k, v in kwargs.items():
                    if k in self._TELEMETRY_KW or k in self._CONTROL_KW:
                        control_kwargs[k] = v
                    else:
                        kw_inputs[k] = v
                inputs = kw_inputs
                kwargs = control_kwargs
            else:
                for k in kwargs.keys():
                    if not (k in self._TELEMETRY_KW or k in self._CONTROL_KW):
                        raise TypeError(
                            f"Unexpected keyword argument '{k}'. "
                            "Pass inputs as a single mapping or omit the "
                            "positional inputs and pass them as keyword "
                            "arguments."
                        )

            normalized = self._normalize_inputs(inputs)
            return await invoke_method(normalized, **invoke_config, **kwargs)

        finally:
            BaseAgent._invoke_depth -= 1

            if BaseAgent._invoke_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    otel_endpoint=otel_endpoint,
                    otel_headers=otel_headers,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    # NOTE: The `invoke` method uses the PEP 570 `/,*` notation to explicitly state which
    # arguments can and cannot be passed as positional or keyword arguments.
    @final
    def invoke(
        self,
        inputs: Optional[InputLike] = None,
        /,
        *,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Executes the agent with the provided inputs and configuration.

        This is the main entry point for agent execution. It handles input normalization,
        telemetry tracking, and proper execution context management. The method supports
        flexible input formats - either as a positional argument or as keyword arguments.

        Args:
            inputs: Optional positional input to the agent. If provided, all non-control
                keyword arguments will be rejected to avoid ambiguity.
            raw_debug: If True, displays raw telemetry data for debugging purposes.
            save_json: If True, saves telemetry data as JSON.
            save_otel: If True, saves telemetry data to OpenTelemetry endpoint.
            metrics_path: Optional file path where telemetry metrics should be saved.
            save_raw_snapshot: If True, saves a raw snapshot of the telemetry data.
            save_raw_records: If True, saves raw telemetry records.
            config: Optional configuration dictionary to override default settings.
            **kwargs: Additional keyword arguments that can be either:
                - Input parameters (when no positional input is provided)
                - Control parameters recognized by the agent

        Returns:
            The result of the agent's execution.

        Raises:
            TypeError: If both positional inputs and non-control keyword arguments are
                provided simultaneously.
        """
        return self._invoke_engine(
            invoke_method=self._invoke,
            inputs=inputs,
            raw_debug=raw_debug,
            save_json=save_json,
            save_otel=save_otel,
            metrics_path=metrics_path,
            save_raw_snapshot=save_raw_snapshot,
            save_raw_records=save_raw_records,
            config=config,
            **kwargs,
        )

    # NOTE: The `ainvoke` method uses the PEP 570 `/,*` notation to explicitly state which
    # arguments can and cannot be passed as positional or keyword arguments.
    @final
    async def ainvoke(
        self,
        inputs: Optional[InputLike] = None,
        /,
        *,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        save_otel: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchrnously executes the agent with the provided inputs and configuration.

        (Async version of `invoke`.)

        This is the main entry point for agent execution. It handles input normalization,
        telemetry tracking, and proper execution context management. The method supports
        flexible input formats - either as a positional argument or as keyword arguments.

        Args:
            inputs: Optional positional input to the agent. If provided, all non-control
                keyword arguments will be rejected to avoid ambiguity.
            raw_debug: If True, displays raw telemetry data for debugging purposes.
            save_json: If True, saves telemetry data as JSON.
            save_otel: If True, saves telemetry data to OpenTelemetry endpoint.
            metrics_path: Optional file path where telemetry metrics should be saved.
            save_raw_snapshot: If True, saves a raw snapshot of the telemetry data.
            save_raw_records: If True, saves raw telemetry records.
            config: Optional configuration dictionary to override default settings.
            **kwargs: Additional keyword arguments that can be either:
                - Input parameters (when no positional input is provided)
                - Control parameters recognized by the agent

        Returns:
            The result of the agent's execution.

        Raises:
            TypeError: If both positional inputs and non-control keyword arguments are
                provided simultaneously.
        """
        return await self._ainvoke_engine(
            invoke_method=self._ainvoke,
            inputs=inputs,
            raw_debug=raw_debug,
            save_json=save_json,
            save_otel=save_otel,
            metrics_path=metrics_path,
            save_raw_snapshot=save_raw_snapshot,
            save_raw_records=save_raw_records,
            config=config,
            **kwargs,
        )

    def format_query(self, prompt: str, state: TState | None = None) -> TState:
        """Format a plain text prompt into the agent's input schema
        possibly incorporating the prior state.

        Agents should override this method for their operation
        """

        if state is not None and "messages" in state:
            state["messages"].append(HumanMessage(content=str(prompt)))
            return state
        return self._normalize_inputs(prompt)

    def format_result(self, result: TState) -> str:
        """Extracts a plain text response from the Agent's output schema

        Agents should override this method for their operation
        """

        if "messages" in result:
            if isinstance(result["messages"], list) and isinstance(
                result["messages"][-1], BaseMessage
            ):
                return result["messages"][-1].text
        raise NotImplementedError()

    def _message_tool_call_ids(self, msg: Any) -> list[str]:
        """Return LangChain tool-call IDs requested by a message, if any."""
        return [
            call["id"]
            for call in getattr(msg, "tool_calls", []) or []
            if "id" in call
        ]

    def _patch_dangling(
        self,
        state: Mapping[str, Any],
        summarized: bool = False,
        config: RunnableConfig | None = None,
    ) -> tuple[Mapping[str, Any], bool]:
        """Patch missing tool responses in a LangGraph message state.

        Tool-calling chat models require every AI tool call to be followed by a
        corresponding ``ToolMessage``. If a tool result is missing because a run
        timed out, failed, or summarization trimmed it away, later model calls can
        fail with provider-side message validation errors. This utility inserts a
        synthetic response for any dangling tool call ID and truncates extremely
        large tool messages.

        Args:
            state: A graph state containing a ``messages`` list.
            summarized: Existing overwrite flag from prior state mutation.

        Returns:
            ``(new_state, full_overwrite_required)``. When the boolean is true,
            callers using an ``add_messages`` reducer should return an
            ``Overwrite(new_state["messages"])`` update rather than appending.
        """
        if "messages" not in state:
            return state, summarized

        new_state = deepcopy(state)
        events = self.events(config)
        dangling_response = (
            "Response Not Found from tool. "
            "May have timed out or been forgotten due to summarization."
        )

        pending_tool_ids: list[str] = []
        for msg in new_state["messages"]:
            if isinstance(msg, ToolMessage):
                if (
                    self.max_single_tool_message_tokens is not None
                    and count_tokens_approximately([msg])
                    > self.max_single_tool_message_tokens
                ):
                    msg.content = "Message too long - truncated."
                    summarized = True
                try:
                    pending_tool_ids.remove(msg.tool_call_id)
                except ValueError:
                    # Orphan tool messages are not ideal, but do not create the
                    # provider-side validation error that missing tool responses do.
                    pass
                continue

            pending_tool_ids.extend(self._message_tool_call_ids(msg))

        if pending_tool_ids:
            summarized = True
            events.emit(
                "Dangling tool calls patched",
                stage="dangling_tool_calls",
                tool_call_ids=list(pending_tool_ids),
            )
            for tool_id in pending_tool_ids:
                for msg_ind, msg in enumerate(new_state["messages"]):
                    if tool_id in self._message_tool_call_ids(msg):
                        new_state["messages"].insert(
                            msg_ind + 1,
                            ToolMessage(
                                content=dangling_response,
                                tool_call_id=tool_id,
                            ),
                        )
                        break

        return new_state, summarized

    def _summarize_context(
        self,
        state: Mapping[str, Any],
        *,
        system_prompt: str | None = None,
        summary_prompt: str | None = None,
        config: RunnableConfig | None = None,
    ) -> tuple[Mapping[str, Any], bool]:
        """Summarize long ``state['messages']`` histories.

        This preserves the first/system message and the last
        ``self.messages_to_keep`` messages, replacing the middle of the
        transcript with an LLM-generated summary when the approximate token count
        exceeds ``self.tokens_before_summarize``.

        The method also avoids creating dangling tool-call pairs when possible by
        moving tool responses that correspond to summarized-away calls into the
        summarized block.
        """
        if "messages" not in state:
            return state, False

        new_state = deepcopy(state)
        events = self.events(config)
        messages = list(new_state.get("messages") or [])
        if len(messages) <= 1:
            return new_state, False

        tokens_before = count_tokens_approximately(messages[1:])
        if tokens_before <= self.tokens_before_summarize:
            return new_state, False

        events.emit(
            message=f"Summarizing ~{tokens_before} token history to compress context",
            stage="summarize_context",
        )
        keep_count = max(0, int(self.messages_to_keep or 0))
        body_messages = messages[1:]
        if keep_count == 0:
            conversation_to_summarize = body_messages
            conversation_to_keep = []
        elif len(body_messages) > keep_count:
            conversation_to_summarize = body_messages[:-keep_count]
            conversation_to_keep = body_messages[-keep_count:]
        else:
            # Nothing can be summarized while still preserving the requested tail.
            return new_state, False

        tool_ids: list[str] = []
        for msg in conversation_to_summarize:
            tool_ids.extend(self._message_tool_call_ids(msg))
            if isinstance(msg, ToolMessage):
                try:
                    tool_ids.remove(msg.tool_call_id)
                except ValueError:
                    pass

        if tool_ids:
            events.emit(
                "Preserving tool responses during summarization",
                stage="summarize_context",
                tool_call_ids=list(tool_ids),
            )
            keep_copy = list(conversation_to_keep)
            for msg in keep_copy:
                if (
                    isinstance(msg, ToolMessage)
                    and msg.tool_call_id in tool_ids
                ):
                    conversation_to_summarize.append(msg)
                    conversation_to_keep.remove(msg)
                    tool_ids.remove(msg.tool_call_id)

        if tool_ids:
            events.emit(
                "Dangling tool calls found during summarization",
                stage="summarize_context",
                tool_call_ids=list(tool_ids),
            )

        summarize_system_message = SystemMessage(
            content="""
        Your only tasks is to provide a detailed, comprehensive summary of the following
        conversation.

        Your summary will be the only information retained from the conversation, so ensure
        it contains all details that need to be remembered to meet the goals of the work.

        The conversation to summarize is:
        """
        )
        to_summarize = [summarize_system_message] + conversation_to_summarize
        to_summarize += [
            HumanMessage(
                content="Summarize that conversation per your instruction."
            )
        ]
        summary = self.tool_llm.invoke(to_summarize)

        first_message = messages[0]
        if system_prompt is not None:
            first_message = SystemMessage(content=system_prompt)

        new_state["messages"] = [first_message, summary] + conversation_to_keep
        return new_state, True

    def prepare_messages_context(
        self,
        state: Mapping[str, Any],
        *,
        system_prompt: str | None = None,
        summary_prompt: str | None = None,
        patch_dangling: bool = True,
        summarize: bool = True,
    ) -> tuple[Mapping[str, Any], bool]:
        """Apply standard message-history maintenance before an LLM call.

        Returns ``(new_state, full_overwrite_required)``. Graph nodes using the
        ``add_messages`` reducer should use :meth:`messages_update` with the
        returned flag to avoid appending duplicate history after summarization or
        dangling-tool patching.
        """
        new_state = deepcopy(state)
        full_overwrite = False
        if summarize:
            new_state, full_overwrite = self._summarize_context(
                new_state,
                system_prompt=system_prompt,
                summary_prompt=summary_prompt,
            )
        if patch_dangling:
            new_state, full_overwrite = self._patch_dangling(
                new_state, full_overwrite
            )
        return new_state, full_overwrite

    def messages_update(
        self,
        state: Mapping[str, Any],
        message_update: Any,
        *,
        full_overwrite: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a LangGraph-safe update for states with a ``messages`` reducer."""
        update = dict(extra or {})
        if full_overwrite:
            messages = list(state.get("messages") or [])
            if message_update is None:
                update["messages"] = Overwrite(messages)
            else:
                if isinstance(message_update, list):
                    messages.extend(message_update)
                else:
                    messages.append(message_update)
                update["messages"] = Overwrite(messages)
        else:
            update["messages"] = message_update
        return update

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        """Normalizes various input formats into a standardized mapping.

        This method converts different input types into a consistent dictionary format
        that can be processed by the agent. String inputs are wrapped as messages, while
        mappings are passed through unchanged.

        Args:
            inputs: The input to normalize. Can be a string (which will be converted to a
                message) or a mapping (which will be returned as-is).

        Returns:
            A mapping containing the normalized inputs, with keys appropriate for agent
            processing.

        Raises:
            TypeError: If the input type is not supported (neither string nor mapping).
        """
        if isinstance(inputs, str):
            # Adjust to your message type
            return {"messages": [HumanMessage(content=inputs)]}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    @cached_property
    def compiled_graph(self) -> CompiledStateGraph:
        """Return the sync compiled StateGraph application for the agent."""
        graph = self.build_graph()
        compiled = graph.compile(
            checkpointer=self.checkpointer,
            store=self.storage,
        ).with_config({"recursion_limit": 50000})
        return self._finalize_graph(compiled)

    async def _aget_async_compiled_graph(self) -> CompiledStateGraph:
        """Return an async-compatible compiled graph for the agent.

        LangGraph SQLite checkpointers are not interchangeable between sync
        and async execution. A graph compiled with ``SqliteSaver`` cannot be
        awaited with ``ainvoke``. Build a separate graph using async SQLite
        persistence resources for async runs.
        """
        if self._async_compiled_graph is None:
            graph = self.build_graph()
            compiled = graph.compile(
                checkpointer=await self._aget_async_checkpointer(),
                store=await self._aget_async_storage(),
            ).with_config({"recursion_limit": 50000})
            self._async_compiled_graph = self._finalize_graph(compiled)
        return self._async_compiled_graph

    async def _aget_async_checkpointer(self) -> BaseCheckpointSaver | None:
        """Return an async-compatible checkpointer for async graph runs."""
        if self.checkpointer is None:
            return None

        if isinstance(self.checkpointer, AsyncSqliteSaver):
            return self.checkpointer

        if self._async_checkpointer is not None:
            return self._async_checkpointer

        if isinstance(self.checkpointer, SqliteSaver):
            if self._checkpointer_was_provided:
                raise RuntimeError(
                    "This agent was configured with a synchronous SqliteSaver "
                    "checkpointer. Async agent execution requires an async "
                    "checkpointer, such as AsyncSqliteSaver, or URSA-managed "
                    "persistence via agent_name."
                )
            self._async_checkpointer = await Checkpointer.async_from_workspace(
                self.den
            )
            return self._async_checkpointer

        if self._checkpointer_has_async_methods(self.checkpointer):
            return self.checkpointer

        raise RuntimeError(
            "The configured checkpointer does not appear to support LangGraph "
            "async checkpoint methods. Use an async checkpointer for "
            "agent.ainvoke(...), or use URSA-managed persistence via "
            "agent_name."
        )

    @staticmethod
    def _checkpointer_has_async_methods(checkpointer: Any) -> bool:
        required = ("aget_tuple", "alist", "aput", "aput_writes")
        return all(
            inspect.iscoroutinefunction(getattr(type(checkpointer), name, None))
            or inspect.isasyncgenfunction(
                getattr(type(checkpointer), name, None)
            )
            for name in required
        )

    @cached_property
    def storage(self) -> BaseStore:
        """Create a sync SQLite-backed LangGraph store."""
        store_path = self.den / "graph_store.sqlite"
        conn = sqlite3.connect(
            store_path, check_same_thread=False, isolation_level=None
        )
        store = SqliteStore(conn)
        store.setup()
        self.hook_storage_setup(store)
        return store

    async def _aget_async_storage(self) -> BaseStore:
        """Create an async SQLite-backed LangGraph store."""
        if self._async_storage is None:
            import aiosqlite

            store_path = self.den / "graph_store.sqlite"
            conn = await aiosqlite.connect(store_path, isolation_level=None)
            store = AsyncSqliteStore(conn)
            await store.setup()
            await self.ahook_storage_setup(store)
            self._async_storage = store
        return self._async_storage

    def hook_storage_setup(self, store: BaseStore) -> None:
        pass

    async def ahook_storage_setup(self, store: BaseStore) -> None:
        pass

    def close(self) -> None:
        """Close sync SQLite resources owned by this agent when possible."""
        if "storage" in self.__dict__:
            self.storage.conn.close()
        if isinstance(self.checkpointer, SqliteSaver):
            self.checkpointer.conn.close()

    async def aclose(self) -> None:
        """Close async SQLite resources owned by this agent when possible."""
        if self._async_storage is not None:
            await self._async_storage.__aexit__(None, None, None)
            await self._async_storage.conn.close()
            self._async_storage = None
            self._async_compiled_graph = None
        if isinstance(self._async_checkpointer, AsyncSqliteSaver):
            await self._async_checkpointer.conn.close()
            self._async_checkpointer = None
            self._async_compiled_graph = None

    @final
    def build_graph(self) -> StateGraph:
        """Build and return the StateGraph backing this agent."""
        self.graph = StateGraph(
            self.state_type,
            context_schema=AgentContext,
        )
        self._build_graph()
        return self.graph

    @abstractmethod
    def _build_graph(self) -> None:
        """Construct the StateGraph for this agent without compiling.

        Called during `__post_init__()` after the Agent has been fully
        Initialized (`__post_init__` is called after `__init__`) to
        instantiate `self.graph`

        Agents should implement this to define their their behavior.

        Agents should treat `self.graph` as read-only
        """
        ...

    def _finalize_graph(
        self, graph_app: CompiledStateGraph
    ) -> CompiledStateGraph:
        """Hook for subclasses to wrap or modify the compiled graph."""
        return graph_app

    def _tool_is_async_only(self, tool: Any) -> bool:
        """Return True for tools that can only be invoked asynchronously.

        MCP tools are commonly exposed as StructuredTool instances with a
        coroutine implementation but no synchronous function implementation.
        Those raise errors like:

            "StructuredTool does not support sync invocation."

        when called via `.invoke()`.
        """

        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        return func is None and coroutine is not None

    def _has_async_only_tools(self) -> bool:
        tools_obj = getattr(self, "tools", None)
        if not tools_obj:
            return False

        try:
            tool_iter = (
                tools_obj.values() if isinstance(tools_obj, dict) else tools_obj
            )
        except Exception:
            return False

        return any(self._tool_is_async_only(t) for t in tool_iter)

    def _invoke(self, input, **config):
        config = self.build_config(**config)

        # If we have async-only tools (e.g. MCP StructuredTools), we must run the
        # graph via `ainvoke` so ToolNode dispatches tools asynchronously.
        if self._has_async_only_tools():
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(self._ainvoke(input, **config))

            raise RuntimeError(
                "This agent has async-only tools, but `.invoke()` was called "
                "from an async context (a running event loop was detected). "
                "Use `await agent.ainvoke(...)` instead."
            )

        try:
            return self.compiled_graph.invoke(
                input, config=config, context=self.context
            )
        except Exception as e:
            # Fallback: if a tool raises the canonical sync-invoke error, retry
            # with ainvoke for backwards compatibility.
            if "does not support sync invocation" in str(e):
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    return asyncio.run(self._ainvoke(input, **config))
            raise

    async def _ainvoke(self, input, **config):
        config = self.build_config(**config)
        graph = await self._aget_async_compiled_graph()
        try:
            return await graph.ainvoke(
                input, config=config, context=self.context
            )
        finally:
            await self.aclose()

    def _stream(self, input, **config):
        config = self.build_config(**config)
        yield from self.compiled_graph.stream(
            input, config=config, context=self.context
        )

    def __call__(self, inputs: InputLike, /, **kwargs: Any) -> Any:
        """Specify calling behavior for class instance."""
        return self.invoke(inputs, **kwargs)

    # Runtime enforcement: forbid subclasses from overriding invoke
    def __init_subclass__(cls, **kwargs):
        """Ensure subclass does not override key method."""
        super().__init_subclass__(**kwargs)
        if "invoke" in cls.__dict__:
            err_msg = (
                f"{cls.__name__} must not override BaseAgent.invoke(); "
                "implement _invoke() only."
            )
            raise TypeError(err_msg)

        # Init graph after subclass has been fully constructed
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.__post_init__()

        cls.__init__ = __init__

    def __post_init__(self):
        self.build_graph()

    def stream(
        self,
        inputs: InputLike,
        config: Any | None = None,  # allow positional/keyword like LangGraph
        /,
        *,
        raw_debug: bool = False,
        save_json: bool | None = None,
        save_otel: bool | None = None,
        metrics_path: str | None = None,
        save_raw_snapshot: bool | None = None,
        save_raw_records: bool | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Streams agent responses with telemetry tracking.

        This method serves as the public streaming entry point for agent interactions.
        It wraps the actual streaming implementation with telemetry tracking to capture
        metrics and debugging information.

        Args:
            inputs: The input to process, which will be normalized internally.
            config: Optional configuration for the agent, compatible with LangGraph
                positional/keyword argument style.
            raw_debug: If True, renders raw debug information in telemetry output.
            save_json: If True, saves telemetry data as JSON.
            metrics_path: Optional file path where metrics should be saved.
            save_raw_snapshot: If True, saves raw snapshot data in telemetry.
            save_raw_records: If True, saves raw record data in telemetry.
            **kwargs: Additional keyword arguments passed to the streaming
                implementation.

        Returns:
            An iterator yielding the agent's responses.

        Note:
            This method tracks invocation depth to properly handle nested agent calls
            and ensure telemetry is only rendered once at the top level.
        """
        # Track invocation depth to handle nested agent calls
        BaseAgent._invoke_depth += 1

        try:
            # Start telemetry tracking for top-level invocations only
            if BaseAgent._invoke_depth == 1:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # Normalize inputs and delegate to the actual streaming implementation
            normalized = self._normalize_inputs(inputs)
            stream_config = dict(config or {})
            yield from self._stream(normalized, **stream_config, **kwargs)

        finally:
            # Decrement invocation depth when exiting
            BaseAgent._invoke_depth -= 1

            # Render telemetry data only for top-level invocations
            if BaseAgent._invoke_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    save_otel=save_otel,
                    filepath=metrics_path,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    def _default_node_tags(
        self, name: str, extra: Sequence[str] | None = None
    ) -> list[str]:
        """Generate default tags for a graph node.

        Args:
            name: The name of the node.
            extra: Optional sequence of additional tags to include.

        Returns:
            list[str]: A list of tags for the node, including the agent name, 'graph',
                the node name, and any extra tags provided.
        """
        # Start with standard tags: agent name, graph indicator, and node name
        tags = [self.name, "graph", name]

        # Add any extra tags if provided
        if extra:
            tags.extend(extra)

        return tags

    def _node_cfg(self, name: str, *extra_tags: str) -> dict:
        """Build a consistent configuration for a node/runnable.

        Creates a configuration dict that can be reapplied after operations like
        .map(), subgraph compile, etc.

        Args:
            name: The name of the node.
            *extra_tags: Additional tags to include in the node configuration.

        Returns:
            dict: A configuration dictionary with run_name, tags, and metadata.
        """
        # Determine the namespace - use first extra tag if available, otherwise
        # convert agent name to snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Combine all tags: agent name, graph indicator, node name, and any extra tags
        tags = [self.name, "graph", name, *extra_tags]

        # Return the complete configuration dictionary
        return dict(
            run_name="node",  # keep "node:" prefixing in the timer
            tags=tags,
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )

    def ns(self, runnable_or_fn, name: str, *extra_tags: str):
        """Return a runnable with node configuration applied.

        Applies the agent's node configuration to a runnable or callable. This method
        should be called again after operations like .map() or subgraph .compile() as
        these operations may drop configuration.

        Args:
            runnable_or_fn: A runnable or callable to configure.
            name: The name to assign to this node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured runnable with the agent's node configuration applied.
        """
        # Convert input to a runnable if it's not already one
        r = coerce_to_runnable(runnable_or_fn, name=name, trace=True)
        # Apply node configuration and return the configured runnable
        return r.with_config(**self._node_cfg(name, *extra_tags))

    def _wrap_node(self, fn_or_runnable, name: str, *extra_tags: str):
        """Wrap a function or runnable as a node in the graph.

        This is a convenience wrapper around the ns() method.

        Args:
            fn_or_runnable: A function or runnable to wrap as a node.
            name: The name to assign to this node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured runnable with the agent's node configuration applied.
        """
        return self.ns(fn_or_runnable, name, *extra_tags)

    def _wrap_cond(self, fn: Any, name: str, *extra_tags: str):
        """Wrap a conditional function as a routing node in the graph.

        Creates a runnable lambda with routing-specific configuration.

        Args:
            fn: The conditional function to wrap.
            name: The name of the routing node.
            *extra_tags: Additional tags to apply to the node.

        Returns:
            A configured RunnableLambda with routing-specific metadata.
        """
        # Use the first extra tag as namespace, or fall back to agent name in snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Create and return a configured RunnableLambda for routing
        return RunnableLambda(fn).with_config(
            run_name="node",
            tags=[
                self.name,
                "graph",
                f"route:{name}",
                *extra_tags,
            ],
            metadata={
                "langgraph_node": f"route:{name}",
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )

    def _named(self, runnable: Any, name: str, *extra_tags: str):
        """Apply a specific name and configuration to a runnable.

        Configures a runnable with a specific name and the agent's metadata.

        Args:
            runnable: The runnable to configure.
            name: The name to assign to this runnable.
            *extra_tags: Additional tags to apply to the runnable.

        Returns:
            A configured runnable with the specified name and agent metadata.
        """
        # Use the first extra tag as namespace, or fall back to agent name in snake_case
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)

        # Apply configuration and return the configured runnable
        return runnable.with_config(
            run_name=name,
            tags=[self.name, "graph", name, *extra_tags],
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )


def _default_tool_error_handler(e: Exception):
    if isinstance(e, ToolInvocationError):
        return e.message
    elif isinstance(e, ToolException):
        return str(e)
    raise e


class AgentWithTools:
    """Mixin that equips an agent with LangGraph tools management."""

    def __init__(
        self,
        *args,
        tools: list[BaseTool] | dict[str, BaseTool] | None = None,
        safe_codes: list[str] | None = None,
        handle_tool_errors=_default_tool_error_handler,
        **kwargs,
    ):
        self._tools: dict[str, BaseTool] = {}
        self.safe_codes = set(safe_codes or [])
        self.handle_tool_errors = handle_tool_errors
        self.tool_node = ToolNode([])
        self._apply_tools(tools, rebuild_graph=False)
        super().__init__(*args, **kwargs)
        self.tool_llm = self.llm.model_copy()

        if getattr(self, "rag_tools", ()):
            from ursa.rag.tools import build_rag_tools

            rag_tools = build_rag_tools(
                names=self.rag_tools,
                group=self.rag_tool_group or self.group or "default",
                llm=self.llm,
                embedding=getattr(self, "rag_tool_embedding", None),
                return_k=self.rag_tool_return_k,
            )
            if rag_tools:
                merged = dict(self._tools)
                merged.update({tool.name: tool for tool in rag_tools})
                self._apply_tools(merged, rebuild_graph=False)

    def __post_init__(self):
        super().__post_init__()

    def hook_storage_setup(self, store: BaseStore) -> None:
        if store is None:
            return
        for safe_code in self.safe_codes:
            store.put(
                ("workspace", "safe_codes"),
                safe_code,
                {},
            )

    async def ahook_storage_setup(self, store: BaseStore) -> None:
        if store is None:
            return
        for safe_code in self.safe_codes:
            await store.aput(
                ("workspace", "safe_codes"),
                safe_code,
                {},
            )

    @property
    def tools(self) -> dict[str, BaseTool]:
        return dict(self._tools)

    @tools.setter
    def tools(self, tools: dict[str, BaseTool] | list[BaseTool] | None):
        self._apply_tools(tools)

    def add_tool(self, tools: BaseTool | list[BaseTool]) -> None:
        bundle = tools if isinstance(tools, list) else [tools]
        merged = dict(self._tools)
        merged.update({tool.name: tool for tool in bundle})
        self._apply_tools(merged)

    async def add_mcp_tools(
        self,
        client: MultiServerMCPClient,
        tool_name: None | str | list[str] = None,
    ) -> None:
        """Add tools from an MCP client to the agent

        Args:
           client: the MCP client to add tools from
           tool_name: if provided, only add named tools
        """
        tools = await client.get_tools()
        if tool_name is not None:
            tool_name = (
                tool_name if isinstance(tool_name, list) else [tool_name]
            )
            tools = [tool for tool in tools if tool.name in tool_name]
        self.add_tool(tools)

    def remove_tool(self, tool_names: str | list[str]) -> None:
        names = tool_names if isinstance(tool_names, list) else [tool_names]
        trimmed = {
            name: tool
            for name, tool in self._tools.items()
            if name not in names
        }
        self._apply_tools(trimmed)

    def _apply_tools(
        self,
        tools: dict[str, BaseTool] | list[BaseTool] | None,
        *,
        rebuild_graph: bool = True,
    ) -> None:
        if tools is None:
            mapping: dict[str, BaseTool] = {}
        elif isinstance(tools, dict):
            mapping = dict(tools)
        else:
            mapping = {tool.name: tool for tool in tools}

        self._tools = mapping
        self.tool_node = ToolNode(
            list(self._tools.values()),
            handle_tool_errors=self.handle_tool_errors,
        )

        if rebuild_graph and hasattr(self, "build_graph"):
            self.__dict__.pop("compiled_graph", None)
            if hasattr(self, "graph"):
                self.build_graph()
