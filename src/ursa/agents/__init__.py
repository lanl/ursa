import importlib
from typing import Any

# Map public names to (module, attribute) for lazy loading
_lazy_attrs: dict[str, tuple[str, str]] = {
    "ArxivAgent": (".acquisition_agents", "ArxivAgent"),
    "OSTIAgent": (".acquisition_agents", "OSTIAgent"),
    "WebSearchAgent": (".acquisition_agents", "WebSearchAgent"),
    "ArxivAgentLegacy": (".arxiv_agent", "ArxivAgentLegacy"),
    "PaperMetadata": (".arxiv_agent", "PaperMetadata"),
    "PaperState": (".arxiv_agent", "PaperState"),
    "BaseAgent": (".base", "BaseAgent"),
    "BaseChatModel": (".base", "BaseChatModel"),
    "ChatAgent": (".chat_agent", "ChatAgent"),
    "ChatState": (".chat_agent", "ChatState"),
    "CodeReviewAgent": (".code_review_agent", "CodeReviewAgent"),
    "CodeReviewState": (".code_review_agent", "CodeReviewState"),
    "ExecutionAgent": (".execution_agent", "ExecutionAgent"),
    "ExecutionState": (".execution_agent", "ExecutionState"),
    "HypothesizerAgent": (".hypothesizer_agent", "HypothesizerAgent"),
    "HypothesizerState": (".hypothesizer_agent", "HypothesizerState"),
    "LammpsAgent": (".lammps_agent", "LammpsAgent"),
    "LammpsState": (".lammps_agent", "LammpsState"),
    "MaterialsProjectAgent": (".mp_agent", "MaterialsProjectAgent"),
    "PlanningAgent": (".planning_agent", "PlanningAgent"),
    "PlanningState": (".planning_agent", "PlanningState"),
    "RAGAgent": (".rag_agent", "RAGAgent"),
    "RAGState": (".rag_agent", "RAGState"),
    "RecallAgent": (".recall_agent", "RecallAgent"),
    "WebSearchAgentLegacy": (".websearch_agent", "WebSearchAgentLegacy"),
    "WebSearchState": (".websearch_agent", "WebSearchState"),
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
