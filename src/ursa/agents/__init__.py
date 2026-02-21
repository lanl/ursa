import importlib
from typing import Any

# Map public names to (module, attribute) for lazy loading
_lazy_attrs: dict[str, tuple[str, str]] = {
    "ArxivAgent": (".acquisition_agents", "ArxivAgent"),
    "OSTIAgent": (".acquisition_agents", "OSTIAgent"),
    "WebSearchAgent": (".acquisition_agents", "WebSearchAgent"),
    "ArxivAgentLegacy": (".arxiv_agent", "ArxivAgentLegacy"),
    "BaseAgent": (".base", "BaseAgent"),
    "BaseChatModel": (".base", "BaseChatModel"),
    "ChatAgent": (".chat_agent", "ChatAgent"),
    "CodeReviewAgent": (".code_review_agent", "CodeReviewAgent"),
    "ExecutionAgent": (".execution_agent", "ExecutionAgent"),
    "GitGoAgent": (".git_go_agent", "GitGoAgent"),
    "HypothesizerAgent": (".hypothesizer_agent", "HypothesizerAgent"),
    "LammpsAgent": (".lammps_agent", "LammpsAgent"),
    "MaterialsProjectAgent": (".mp_agent", "MaterialsProjectAgent"),
    "PlanningAgent": (".planning_agent", "PlanningAgent"),
    "RAGAgent": (".rag_agent", "RAGAgent"),
    "RecallAgent": (".recall_agent", "RecallAgent"),
    "WebSearchAgentLegacy": (".websearch_agent", "WebSearchAgentLegacy"),
}

__all__ = list(_lazy_attrs.keys())


def __getattr__(name: str) -> Any:
    """Dynamically import attributes on first access.

    This avoids importing all agent modules at package import time,
    so a failure in one agent does not prevent using others.
    """
    try:
        module_name, attr_name = _lazy_attrs[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None

    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr_name)
    # Cache the loaded attribute so subsequent access is fast
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Include lazy attributes in dir(package)
    return sorted(list(globals().keys()) + list(__all__))
