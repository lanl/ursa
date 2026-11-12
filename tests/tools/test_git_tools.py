from pathlib import Path

import pytest
from langchain.chat_models import BaseChatModel
from langchain.tools import ToolException

from tests.tools.utils import make_runtime
from ursa.tools.git_tools import git_status


def test_git_status_rejects_outside_repository_as_recoverable_tool_error(
    tmp_path: Path, chat_model: BaseChatModel
):
    outside = tmp_path.parent / "outside-workspace"
    outside.mkdir()
    symlink = tmp_path / "linked-outside-workspace"
    try:
        symlink.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")

    runtime = make_runtime(tmp_path, llm=chat_model)
    for repo_path in ("../outside-workspace", symlink.name):
        with pytest.raises(ToolException) as raised:
            git_status.func(runtime=runtime, repo_path=repo_path)

        assert (
            str(raised.value) == "repo_path must resolve inside the workspace"
        )
