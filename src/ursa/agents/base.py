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

import re
from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import (
    Any,
    Callable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Union,
    final,
)
from uuid import uuid4

from langchain.chat_models import BaseChatModel
from langchain_core.load import dumps
from langchain_core.messages import HumanMessage
from langchain_core.runnables import (
    RunnableLambda,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from ursa.observability.timing import (
    Telemetry,  # for timing / telemetry / metrics
)

InputLike = Union[str, Mapping[str, Any]]
_INVOKE_DEPTH = ContextVar("_INVOKE_DEPTH", default=0)


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


class BaseAgent(ABC):
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
                        Various helper methods (_default_node_tags, _as_runnable, etc.)
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

    _TELEMETRY_KW = {
        "raw_debug",
        "save_json",
        "metrics_path",
        "save_raw_snapshot",
        "save_raw_records",
    }

    _CONTROL_KW = {"config", "recursion_limit", "tags", "metadata", "callbacks"}

    def __init__(
        self,
        llm: BaseChatModel,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        enable_metrics: bool = True,
        metrics_dir: str = ".ursa_metrics",  # dir to save metrics, with a default
        autosave_metrics: bool = True,
        thread_id: Optional[str] = None,
    ):
        self.llm = llm
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
        self.thread_id = thread_id or uuid4().hex
        self.checkpointer = checkpointer
        self.telemetry = Telemetry(
            enable=enable_metrics,
            output_dir=metrics_dir,
            save_json_default=autosave_metrics,
        )

    @property
    def name(self) -> str:
        """Agent name."""
        return self.__class__.__name__

    def add_node(
        self,
        graph: StateGraph,
        f: Callable[..., Mapping[str, Any]],
        node_name: Optional[str] = None,
        agent_name: Optional[str] = None,
    ) -> StateGraph:
        """Add a node to the state graph with token usage tracking.

        This method adds a function as a node to the state graph, wrapping it to track
        token usage during execution. The node is identified by either the provided
        node_name or the function's name.

        Args:
            graph: The StateGraph to add the node to.
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

        return graph.add_node(_node_name, wrapped_node)

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
            "callbacks": self.telemetry.callbacks,
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

        # Apply any remaining overrides directly to the base configuration
        base.update(overrides)

        return base

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
        # Track invocation depth to manage nested agent calls
        depth = _INVOKE_DEPTH.get()
        _INVOKE_DEPTH.set(depth + 1)
        try:
            # Start telemetry tracking for the top-level invocation
            if depth == 0:
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
            return self._invoke(normalized, config=config, **kwargs)

        finally:
            # Clean up the invocation depth tracking
            new_depth = _INVOKE_DEPTH.get() - 1
            _INVOKE_DEPTH.set(new_depth)

            # For the top-level invocation, finalize telemetry and generate outputs
            if new_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    filepath=metrics_path,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

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

    @abstractmethod
    def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        """Subclasses implement the actual work against normalized inputs."""
        ...

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

    def stream(
        self,
        inputs: InputLike,
        config: Any | None = None,  # allow positional/keyword like LangGraph
        /,
        *,
        raw_debug: bool = False,
        save_json: bool | None = None,
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
        depth = _INVOKE_DEPTH.get()
        _INVOKE_DEPTH.set(depth + 1)

        try:
            # Start telemetry tracking for top-level invocations only
            if depth == 0:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # Normalize inputs and delegate to the actual streaming implementation
            normalized = self._normalize_inputs(inputs)
            yield from self._stream(normalized, config=config, **kwargs)

        finally:
            # Decrement invocation depth when exiting
            new_depth = _INVOKE_DEPTH.get() - 1
            _INVOKE_DEPTH.set(new_depth)

            # Render telemetry data only for top-level invocations
            if new_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    filepath=metrics_path,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    def _stream(
        self,
        inputs: Mapping[str, Any],
        *,
        config: Any | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Subclass method to be overwritten for streaming implementation."""
        raise NotImplementedError(
            f"{self.name} does not support streaming. "
            "Override _stream(...) in your agent to enable it."
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

    def _as_runnable(self, fn: Any):
        """Convert a function to a runnable if it isn't already.

        Args:
            fn: The function or object to convert to a runnable.

        Returns:
            A runnable object that can be used in the graph. If the input is already
            runnable (has .with_config and .invoke methods), it's returned as is.
            Otherwise, it's wrapped in a RunnableLambda.
        """
        # Check if the function already has the required runnable interface
        # If so, return it as is; otherwise wrap it in a RunnableLambda
        return (
            fn
            if hasattr(fn, "with_config") and hasattr(fn, "invoke")
            else RunnableLambda(fn)
        )

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
        r = self._as_runnable(runnable_or_fn)
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
