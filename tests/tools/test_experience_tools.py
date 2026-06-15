from pathlib import Path
from unittest.mock import MagicMock

from ursa.agents.base import AgentContext
from ursa.tools.experience_tools import write_experience


def test_write_experience_reports_path_with_symlinked_den(
    tmp_path: Path,
) -> None:
    real_den = tmp_path / "real-den"
    real_den.mkdir()
    den_link = tmp_path / "den-link"
    den_link.symlink_to(real_den, target_is_directory=True)

    runtime = MagicMock()
    context = MagicMock(spec=AgentContext)
    context.den = den_link
    runtime.context = context
    runtime.store = None
    runtime.tool_call_id = "tool-call"
    runtime.config = {"metadata": {"thread_id": "thread"}}

    result = write_experience.func(
        filename="note.md",
        content="Important observation",
        append=False,
        runtime=runtime,
    )

    assert (
        result
        == "Experience file experiences/note.md overwritten successfully."
    )
    assert (real_den / "experiences" / "note.md").read_text(
        encoding="utf-8"
    ) == "Important observation\n"
