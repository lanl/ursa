"""Git-aware coding agent with pluggable language support."""

from __future__ import annotations

from langchain.chat_models import BaseChatModel
from langchain_core.tools import BaseTool

from ursa.agents.execution_agent import ExecutionAgent
from ursa.prompt_library.git_prompts import compose_git_prompt

# Lazy import to avoid circular deps at module level
from ursa.prompt_library.go_prompts import go_language_prompt
from ursa.tools.git_tools import GIT_TOOLS
from ursa.tools.go_tools import GO_TOOLS

LANGUAGE_REGISTRY: dict[str, dict] = {
    "generic": {
        "tools": None,
        "prompt": None,
        "safe_codes": [],
    },
    "go": {
        "tools": GO_TOOLS,
        "prompt": go_language_prompt,
        "safe_codes": ["go"],
    },
}


class GitAgent(ExecutionAgent):
    """Execution agent with git tools and optional language-specific extensions.

    Use directly for language-agnostic git work, or pass ``language_tools``,
    ``language_prompt``, and ``safe_codes`` for a language-specific variant.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        language_tools: list[BaseTool] | None = None,
        language_prompt: str | None = None,
        safe_codes: list[str] | None = None,
        **kwargs,
    ):
        extra_tools: list[BaseTool] = list(GIT_TOOLS)
        if language_tools:
            extra_tools.extend(language_tools)

        super().__init__(
            llm=llm,
            extra_tools=extra_tools,
            safe_codes=safe_codes or [],
            **kwargs,
        )

        self.executor_prompt = compose_git_prompt(language_prompt or "")

        self.remove_tool([
            "run_command",
            "run_web_search",
            "run_osti_search",
            "run_arxiv_search",
        ])


def make_git_agent(
    llm: BaseChatModel,
    language: str = "generic",
    **kwargs,
) -> GitAgent:
    """Create a GitAgent preconfigured for a specific language.

    Supported languages are defined in ``LANGUAGE_REGISTRY``.  Pass
    ``"generic"`` (the default) for a git-only agent with no
    language-specific tools.
    """
    config = LANGUAGE_REGISTRY.get(language)
    if config is None:
        raise ValueError(
            f"Unknown language {language!r}. "
            f"Available: {sorted(LANGUAGE_REGISTRY)}"
        )
    return GitAgent(
        llm=llm,
        language_tools=config.get("tools"),
        language_prompt=config.get("prompt"),
        safe_codes=config.get("safe_codes"),
        **kwargs,
    )
