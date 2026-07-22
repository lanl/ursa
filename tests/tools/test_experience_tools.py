from dataclasses import replace
from pathlib import Path

from langchain.chat_models import BaseChatModel

from tests.tools.utils import invoke_with_event_recorder, make_runtime
from ursa.tools.experience_tools import read_experience, write_experience


def test_write_experience_reports_path_with_symlinked_den(
    tmp_path: Path,
    chat_model: BaseChatModel,
) -> None:
    real_den = tmp_path / "real-den"
    real_den.mkdir()
    den_link = tmp_path / "den-link"
    den_link.symlink_to(real_den, target_is_directory=True)

    runtime = make_runtime(real_den, llm=chat_model)
    runtime.context = replace(runtime.context, den=den_link)

    result, recorder = invoke_with_event_recorder(
        write_experience.func,
        filename="note.md",
        content="Important observation",
        append=False,
        runtime=runtime,
    )

    assert (
        result
        == "Experience file experiences/note.md overwritten successfully."
        or result
        == "Experience file experiences\\note.md overwritten successfully."
    )
    assert (real_den / "experiences" / "note.md").read_text(
        encoding="utf-8"
    ) == "Important observation\n"
    assert len(recorder.events) == 2
    _, finished = recorder.events[1]
    assert finished["message"] == "Experience written"
    assert finished["artifact"]["content"] == str(
        real_den / "experiences" / "note.md"
    )


def test_read_experience_emits_error_event_when_read_fails(
    tmp_path: Path,
    chat_model: BaseChatModel,
    monkeypatch,
) -> None:
    experiences = tmp_path / "experiences"
    experiences.mkdir()
    (experiences / "note.md").write_text("content", encoding="utf-8")
    runtime = make_runtime(tmp_path, llm=chat_model)
    runtime.context = replace(runtime.context, den=tmp_path)

    def fail_read(path: Path) -> str:
        raise OSError("unreadable")

    monkeypatch.setattr(
        "ursa.tools.experience_tools.read_text_from_file", fail_read
    )

    result, recorder = invoke_with_event_recorder(
        read_experience.func,
        filename="note.md",
        runtime=runtime,
    )

    assert result == "Failed to read note.md: unreadable"
    assert len(recorder.events) == 1
    _, event = recorder.events[0]
    assert event["phase"] == "error"
    assert event["message"] == "Failed to read experience"
    assert event["error"] == "unreadable"
