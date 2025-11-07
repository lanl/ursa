"""Base workflow class providing telemetry and execution abstractions.

This module defines the BaseWorkflow abstract class, which serves as the foundation for all
workflow implementations in the URSA framework. It provides:

- Telemetry and metrics collection
- Thread and checkpoint management
- Input normalization and validation
- Execution flow control with invoke methods
- Graph integration utilities for LangGraph compatibility
- Runtime enforcement of the agent interface contract

Workflows built on this base class benefit from consistent behavior, observability, and
integration capabilities while only needing to implement the core _invoke method.
"""

from abc import ABC, abstractmethod
from contextvars import ContextVar
from typing import (
    Any,
    Mapping,
    Optional,
    Union,
    final,
)
from uuid import uuid4

from ursa.observability.timing import (
    Telemetry,  # for timing / telemetry / metrics
)

InputLike = Union[str, Mapping[str, Any]]
_INVOKE_DEPTH = ContextVar("_INVOKE_DEPTH", default=0)


class BaseWorkflow(ABC):
    """Abstract base class for all workflow implementations in the URSA framework.

    BaseWorkflow provides a standardized foundation for building workflows consisting of
    LLM-powered agents with built-in telemetry, configuration management, and execution
    flow control. It handles common tasks like input normalization, thread management,
    metrics collection, and LangGraph integration.

    Subclasses only need to implement the _invoke method to define their core
    functionality, while inheriting standardized invocation patterns, telemetry, and
    graph integration capabilities. The class enforces a consistent interface through
    runtime checks that prevent subclasses from overriding critical methods like
    invoke().

    The agent supports direct invocation with inputs, with
    automatic tracking of token usage, execution time, and other metrics. It also
    provides utilities for integrating with LangGraph through node wrapping and
    configuration.

    Subclass Inheritance Guidelines:
        - Must Override: _invoke() - Define your agent's core functionality
        - Can Override: _normalize_inputs() - Customize input handling
                        Various helper methods
        - Never Override: invoke() - Final method with runtime enforcement
                          __call__() - Delegates to invoke
                          Other public methods (build_config, write_state, add_node)

    To create a custom agent, inherit from this class and implement the _invoke method:

    ```python
    class MyWorkflow(BaseWorkflow):
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
        enable_metrics: bool = False,  # default to enabling metrics
        metrics_dir: str = ".ursa_metrics",  # dir to save metrics, with a default
        autosave_metrics: bool = True,
        thread_id: Optional[str] = None,
        **kwargs,
    ):
        """Initializes the base agent with a language model and optional configurations.

        Args:
            enable_metrics: Whether to collect performance and usage metrics.
            metrics_dir: Directory path where metrics will be saved.
            autosave_metrics: Whether to automatically save metrics to disk.
            thread_id: Unique identifier for this agent instance. Generated if not
                       provided.
            **kwargs: Additional keyword arguments passed to the LLM initialization.
        """

        self.thread_id = thread_id or uuid4().hex
        self.telemetry = Telemetry(
            enable=enable_metrics,
            output_dir=metrics_dir,
            save_json_default=autosave_metrics,
        )

    @property
    def name(self) -> str:
        """Agent name."""
        return self.__class__.__name__

    def _adopt(
        self, child
    ):  # this should probably be in a more general place then here
        # import pdb; pdb.set_trace()
        child.telemetry = self.telemetry
        try:
            child.thread_id = self.thread_id  # if present in your base
        except AttributeError:
            pass

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
            return {"task": inputs}
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
