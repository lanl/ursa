"""Git-aware Go coding agent based on ExecutionAgent."""

from langchain.chat_models import BaseChatModel

from ursa.agents.execution_agent import ExecutionAgent
from ursa.prompt_library.git_go_prompts import git_go_executor_prompt
from ursa.tools.git_tools import (
    git_add,
    git_commit,
    git_create_branch,
    git_diff,
    git_log,
    git_ls_files,
    git_status,
    git_switch,
    gofmt_files,
    go_build,
    go_test,
    go_vet,
    go_mod_tidy,
    golangci_lint,
)


class GitGoAgent(ExecutionAgent):
    """Execution agent specialized for git-managed Go repositories.
    
    Tools:
    - Git: status, diff, log, ls-files, add, commit, switch, create_branch
    - Go: build, test, vet, mod tidy, linting (golangci-lint with .golangci.yml support)
    - Code formatting: gofmt
    
    The agent uses a unified 1800-second timeout for all operations to accommodate
    large builds, test suites, and complex git operations.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        **kwargs,
    ):
        extra_tools = [
            git_status,
            git_diff,
            git_log,
            git_ls_files,
            git_add,
            git_commit,
            git_switch,
            git_create_branch,
            gofmt_files,
            go_build,
            go_test,
            go_vet,
            go_mod_tidy,
            golangci_lint,
        ]
        super().__init__(
            llm=llm,
            extra_tools=extra_tools,
            safe_codes=["go"],
            **kwargs,
        )
        self.executor_prompt = git_go_executor_prompt

        self.remove_tool(
            [
                "run_command",
                "run_web_search",
                "run_osti_search",
                "run_arxiv_search",
            ]
        )
