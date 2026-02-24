"""Git-aware Go coding agent -- backward-compatible wrapper around GitAgent."""

from langchain.chat_models import BaseChatModel

from ursa.agents.git_agent import GitAgent
from ursa.prompt_library.go_prompts import go_language_prompt
from ursa.tools.go_tools import GO_TOOLS


class GitGoAgent(GitAgent):
    """Execution agent specialized for git-managed Go repositories.

    Tools:
    - Git: status, diff, log, ls-files, add, commit, switch, create_branch
    - Go: build, test, vet, mod tidy, linting (golangci-lint with .golangci.yml support)
    - Code formatting: gofmt

    This is a convenience subclass of :class:`GitAgent` with the Go language
    tools and prompt pre-configured.
    """

    def __init__(self, llm: BaseChatModel, **kwargs):
        super().__init__(
            llm=llm,
            language_tools=GO_TOOLS,
            language_prompt=go_language_prompt,
            safe_codes=["go"],
            **kwargs,
        )
