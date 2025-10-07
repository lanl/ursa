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

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.load import dumps
from langchain_core.runnables import (
    RunnableLambda,
)
from langchain_litellm import ChatLiteLLM
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph

from ursa.observability.timing import (
    Telemetry,  # for timing / telemetry / metrics
)

InputLike = Union[str, Mapping[str, Any]]
_INVOKE_DEPTH = ContextVar("_INVOKE_DEPTH", default=0)


def _to_snake(s: str) -> str:
    s = re.sub(
        r"^([A-Z]{2,})([A-Z][a-z])",
        lambda m: m.group(1)[0] + m.group(1)[1:].lower() + m.group(2),
        str(s),
    )  # RAGAgent -> RagAgent
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", s)  # CamelCase -> snake_case
    s = s.replace("-", "_").replace(" ", "_")
    return s.lower()


class BaseAgent(ABC):
    # llm: BaseChatModel
    # llm_with_tools: Runnable[LanguageModelInput, BaseMessage]

    def __init__(
        self,
        llm: str | BaseChatModel,
        checkpointer: BaseCheckpointSaver = None,
        enable_metrics: bool = False,  # default to enabling metrics
        metrics_dir: str = ".ursa_metrics",  # dir to save metrics, with a default
        autosave_metrics: bool = True,
        thread_id: Optional[str] = None,
        **kwargs,
    ):
        match llm:
            case BaseChatModel():
                self.llm = llm

            case str():
                self.llm_provider, self.llm_model = llm.split("/")
                self.llm = ChatLiteLLM(
                    model=llm,
                    max_tokens=kwargs.pop("max_tokens", 10000),
                    max_retries=kwargs.pop("max_retries", 2),
                    **kwargs,
                )

            case _:
                raise TypeError(
                    "llm argument must be a string with the provider and model, or a BaseChatModel instance."
                )

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
        """Add node to graph.

        This is used to track token usage and is simply the following.

        ```python
        _node_name = node_name or f.__name__
        return graph.add_node(
            _node_name, self._wrap_node(f, _node_name, self.name)
        )
        ```
        """
        _node_name = node_name or f.__name__
        _agent_name = agent_name or _to_snake(self.name)
        wrapped_node = self._wrap_node(f, _node_name, _agent_name)
        return graph.add_node(_node_name, wrapped_node)

    def write_state(self, filename, state):
        json_state = dumps(state, ensure_ascii=False)
        with open(filename, "w") as f:
            f.write(json_state)

    # BaseAgent
    def build_config(self, **overrides) -> dict:
        """
        Build a config dict that includes telemetry callbacks and the thread_id.
        You can pass overrides like recursion_limit=..., configurable={...}, etc.
        """
        base = {
            "configurable": {"thread_id": self.thread_id},
            "metadata": {
                "thread_id": self.thread_id,
                "telemetry_run_id": self.telemetry.context.get("run_id"),
            },
            # "configurable": {
            #     "thread_id": getattr(self, "thread_id", "default")
            # },
            # "metadata": {
            #     "thread_id": getattr(self, "thread_id", "default"),
            #     "telemetry_run_id": self.telemetry.context.get("run_id"),
            # },
            "tags": [self.name],
            "callbacks": self.telemetry.callbacks,
        }
        # include model name when we can
        model_name = getattr(self, "llm_model", None) or getattr(
            getattr(self, "llm", None), "model", None
        )
        if model_name:
            base["metadata"]["model"] = model_name

        if "configurable" in overrides and isinstance(
            overrides["configurable"], dict
        ):
            base["configurable"].update(overrides.pop("configurable"))
        if "metadata" in overrides and isinstance(overrides["metadata"], dict):
            base["metadata"].update(overrides.pop("metadata"))
        # merge tags if caller provides them
        if "tags" in overrides and isinstance(overrides["tags"], list):
            base["tags"] = base["tags"] + [
                t for t in overrides.pop("tags") if t not in base["tags"]
            ]
        base.update(overrides)
        return base

    # agents will invoke like this:
    # planning_output = planner.invoke(
    #     {"messages": [HumanMessage(content=problem)]},
    #     config={
    #         "recursion_limit": 999_999,
    #         "configurable": {"thread_id": planner.thread_id},
    #     },
    # )
    # they can also, separately, override these defaults about metrics
    # keys that are NOT inputs; they should not be folded into the inputs mapping
    _TELEMETRY_KW = {
        "raw_debug",
        "save_json",
        "metrics_path",
        "save_raw_snapshot",
        "save_raw_records",
    }
    _CONTROL_KW = {"config", "recursion_limit", "tags", "metadata", "callbacks"}

    @final
    def invoke(
        self,
        inputs: Optional[InputLike] = None,  # sentinel
        /,
        *,
        raw_debug: bool = False,
        save_json: Optional[bool] = None,
        metrics_path: Optional[str] = None,
        save_raw_snapshot: Optional[bool] = None,
        save_raw_records: Optional[bool] = None,
        config: Optional[dict] = None,
        **kwargs: Any,  # may contain inputs (keyword-inputs) and/or control kw
    ) -> Any:
        depth = _INVOKE_DEPTH.get()
        _INVOKE_DEPTH.set(depth + 1)
        try:
            if depth == 0:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )

            # If no positional inputs were provided, split kwargs into inputs vs control
            if inputs is None:
                kw_inputs: dict[str, Any] = {}
                control_kwargs: dict[str, Any] = {}
                for k, v in kwargs.items():
                    if k in self._TELEMETRY_KW or k in self._CONTROL_KW:
                        control_kwargs[k] = v
                    else:
                        kw_inputs[k] = v
                inputs = kw_inputs
                kwargs = control_kwargs  # only control kwargs remain

            # If both positional inputs and extra unknown kwargs-as-inputs are given, forbid merging
            else:
                # keep only control kwargs; anything else would be ambiguous
                for k in kwargs.keys():
                    if not (k in self._TELEMETRY_KW or k in self._CONTROL_KW):
                        raise TypeError(
                            f"Unexpected keyword argument '{k}'. "
                            "Pass inputs as a single mapping or omit the positional "
                            "inputs and pass them as keyword arguments."
                        )

            # subclasses may translate keys
            normalized = self._normalize_inputs(inputs)

            # forward config + any control kwargs (e.g., recursion_limit) to the agent
            return self._invoke(normalized, config=config, **kwargs)

        finally:
            new_depth = _INVOKE_DEPTH.get() - 1
            _INVOKE_DEPTH.set(new_depth)
            if new_depth == 0:
                self.telemetry.render(
                    raw=raw_debug,
                    save_json=save_json,
                    filepath=metrics_path,
                    save_raw_snapshot=save_raw_snapshot,
                    save_raw_records=save_raw_records,
                )

    def _normalize_inputs(self, inputs: InputLike) -> Mapping[str, Any]:
        if isinstance(inputs, str):
            # Adjust to your message type
            from langchain_core.messages import HumanMessage

            return {"messages": [HumanMessage(content=inputs)]}
        if isinstance(inputs, Mapping):
            return inputs
        raise TypeError(f"Unsupported input type: {type(inputs)}")

    @abstractmethod
    def _invoke(self, inputs: Mapping[str, Any], **config: Any) -> Any:
        """Subclasses implement the actual work against normalized inputs."""
        ...

    def __call__(self, inputs: InputLike, /, **kwargs: Any) -> Any:
        return self.invoke(inputs, **kwargs)

    # Runtime enforcement: forbid subclasses from overriding invoke
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "invoke" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override BaseAgent.invoke(); implement _invoke() only."
            )

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
        """Public streaming entry point. Telemetry-wrapped."""
        depth = _INVOKE_DEPTH.get()
        _INVOKE_DEPTH.set(depth + 1)
        try:
            if depth == 0:
                self.telemetry.begin_run(
                    agent=self.name, thread_id=self.thread_id
                )
            normalized = self._normalize_inputs(inputs)
            yield from self._stream(normalized, config=config, **kwargs)
        finally:
            new_depth = _INVOKE_DEPTH.get() - 1
            _INVOKE_DEPTH.set(new_depth)
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
        raise NotImplementedError(
            f"{self.name} does not support streaming. "
            "Override _stream(...) in your agent to enable it."
        )

    # def run(
    #     self,
    #     *args,
    #     raw_debug: bool = False,
    #     save_json: bool | None = None,
    #     metrics_path: str | None = None,
    #     save_raw_snapshot: bool | None = None,
    #     save_raw_records: bool | None = None,
    #     **kwargs
    # ):
    #     try:
    #         self.telemetry.begin_run(agent=self.name, thread_id=self.thread_id)
    #         result = self._run_impl(*args, **kwargs)
    #         return result
    #     finally:
    #         print(self.telemetry.render(
    #             raw=raw_debug,
    #             save_json=save_json,
    #             filepath=metrics_path,
    #             save_raw_snapshot=save_raw_snapshot,
    #             save_raw_records=save_raw_records,
    #         ))

    # @abstractmethod
    # def _run_impl(self, *args, **kwargs):
    #     raise NotImplementedError("Agents must implement _run_impl")

    def _default_node_tags(
        self, name: str, extra: Sequence[str] | None = None
    ) -> list[str]:
        tags = [self.name, "graph", name]
        if extra:
            tags.extend(extra)
        return tags

    def _as_runnable(self, fn: Any):
        # If it's already runnable (has .with_config/.invoke), return it; else wrap
        return (
            fn
            if hasattr(fn, "with_config") and hasattr(fn, "invoke")
            else RunnableLambda(fn)
        )

    def _node_cfg(self, name: str, *extra_tags: str) -> dict:
        """Build a consistent config for a node/runnable so we can reapply it after .map(), subgraph compile, etc."""
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)
        tags = [self.name, "graph", name, *extra_tags]
        return dict(
            run_name="node",  # keep "node:" prefixing in the timer; don't fight Rich labels here
            tags=tags,
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )

    def ns(self, runnable_or_fn, name: str, *extra_tags: str):
        """Return a runnable with our node config applied. Safe to call on callables or runnables.
        IMPORTANT: call this AGAIN after .map() / subgraph .compile() (they often drop config)."""
        r = self._as_runnable(runnable_or_fn)
        return r.with_config(**self._node_cfg(name, *extra_tags))

    def _wrap_node(self, fn_or_runnable, name: str, *extra_tags: str):
        return self.ns(fn_or_runnable, name, *extra_tags)

    def _wrap_cond(self, fn: Any, name: str, *extra_tags: str):
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)
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
        ns = extra_tags[0] if extra_tags else _to_snake(self.name)
        return runnable.with_config(
            run_name=name,
            tags=[self.name, "graph", name, *extra_tags],
            metadata={
                "langgraph_node": name,
                "ursa_ns": ns,
                "ursa_agent": self.name,
            },
        )
