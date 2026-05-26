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
from collections.abc import Mapping
from typing import (
    Any,
    final,
)

InputLike = str | Mapping[str, Any]


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

    def __init__(self, **kwargs):
        pass

    # NOTE: The `invoke` method uses the PEP 570 `/,*` notation to explicitly state which
    # arguments can and cannot be passed as positional or keyword arguments.
    @final
    def invoke(
        self,
        inputs: InputLike | None = None,
        /,
        *,
        config: dict | None = None,
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

        normalized = self._normalize_inputs(inputs)

        # Delegate to the subclass implementation with the normalized inputs
        # and any control parameters
        return self._invoke(normalized, config=config, **kwargs)

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
