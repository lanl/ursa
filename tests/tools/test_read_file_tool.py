import os
from pathlib import Path

import pytest
from langchain.chat_models import BaseChatModel

from tests.event_fixtures import (
    fixed_monotonic_timestamp as fixed_monotonic_timestamp,
)
from tests.tools.utils import (
    invoke_with_event_recorder,
    invoke_with_parent_run,
    make_runtime,
)
from ursa.tools.read_file_tool import download_file_tool, read_file
from ursa.util import parse


@pytest.mark.parametrize("absolute", [False, True])
def test_read_file_reads_text_from_workspace(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
    absolute: bool,
):
    target = tmp_path / "example.txt"
    target.write_text("sample text", encoding="utf-8")
    result, recorder = invoke_with_event_recorder(
        read_file.func,
        filename=str(target) if absolute else target.name,
        runtime=make_runtime(
            tmp_path,
            llm=chat_model,
            tool_call_id="read-file-call",
        ),
    )

    assert result == "sample text"
    assert len(recorder.events) == 2
    _, started = recorder.events[0]
    assert started["phase"] == "start"
    assert started["message"] == "Reading file"
    _, finished = recorder.events[1]
    assert finished["phase"] == "end"
    assert finished["message"] == "File read"
    assert finished["artifact"] == {
        "content": str(target),
        "mime_type": "application/vnd.ursa.file-reference",
        "metadata": {
            "title": "File read",
            "path": str(target),
            "content_mime_type": "text/plain",
        },
    }
    assert isinstance(finished["elapsed_ms"], float)


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "symlink"])
def test_read_file_rejects_paths_outside_workspace(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel, path_kind: str
):
    monkeypatch.delenv("URSA_ALLOW_UNSAFE_READS", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("outside contents", encoding="utf-8")
    if path_kind == "symlink":
        link = workspace / "link.txt"
        try:
            link.symlink_to(secret)
        except OSError:
            pytest.skip("Symlink creation is unavailable")
        filename = link.name
    else:
        filename = "../secret.txt" if path_kind == "relative" else str(secret)

    monkeypatch.setattr(
        "ursa.util.parse.read_text_from_file",
        lambda path: pytest.fail("Rejected files must not be read"),
    )
    result = invoke_with_parent_run(
        lambda config: read_file.func(
            filename=filename,
            runtime=make_runtime(workspace, llm=chat_model, config=config),
        )
    )
    assert "resolves outside workspace directory" in result


@pytest.mark.parametrize(
    "setting, allowed",
    [
        ("1", True),
        ("true", True),
        ("yes", True),
        (" ON ", True),
        ("0", False),
        ("false", False),
        ("", False),
        ("invalid", False),
    ],
)
def test_read_file_outside_workspace_requires_explicit_opt_in(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
    setting: str,
    allowed: bool,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "external.txt"
    target.write_text("external input", encoding="utf-8")
    monkeypatch.setenv("URSA_ALLOW_UNSAFE_READS", setting)
    # The independent write override must not grant read access.
    monkeypatch.setenv("URSA_ALLOW_UNSAFE_WRITES", "1")
    result = invoke_with_parent_run(
        lambda config: read_file.func(
            filename=str(target),
            runtime=make_runtime(workspace, llm=chat_model, config=config),
        )
    )
    if allowed:
        assert result == "external input"
    else:
        assert "resolves outside workspace directory" in result


def test_read_file_uses_pdf_reader(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    called = {}

    def fake_pdf_reader(path: Path) -> str:
        called["path"] = path
        return "pdf contents"

    def fail_text_reader(path: Path) -> str:
        raise AssertionError("read_text_file should not be called for PDFs")

    monkeypatch.setattr("ursa.util.parse.read_text_from_file", fake_pdf_reader)
    monkeypatch.setattr(parse, "read_text_file", fail_text_reader)

    result = invoke_with_parent_run(
        lambda config: read_file.func(
            filename="report.pdf",
            runtime=make_runtime(
                tmp_path,
                llm=chat_model,
                tool_call_id="pdf-call",
                config=config,
            ),
        )
    )

    assert result == "pdf contents"
    assert called["path"] == tmp_path / "report.pdf"


def test_download_file_rejects_invalid_ascii_output_path_without_cleaning(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
):
    existing = tmp_path / "caf.txt"
    existing.write_text("original", encoding="utf-8")
    runtime = make_runtime(tmp_path, llm=chat_model)
    monkeypatch.setattr(
        "ursa.tools.read_file_tool.requests.get",
        lambda *args, **kwargs: pytest.fail(
            "requests.get should not run for invalid output_path"
        ),
    )

    result = download_file_tool.func(
        url="https://example.com/file.txt",
        output_path="caf\u00e9.txt",
        runtime=runtime,
    )

    assert result.startswith("Invalid output_path:")
    assert "U+00E9" in result
    assert "corrected ASCII string" in result
    assert existing.read_text(encoding="utf-8") == "original"


@pytest.mark.parametrize("path_kind", ["relative", "absolute", "symlink"])
def test_download_file_rejects_paths_outside_workspace(
    monkeypatch,
    tmp_path: Path,
    chat_model: BaseChatModel,
    path_kind: str,
):
    monkeypatch.delenv("URSA_ALLOW_UNSAFE_WRITES", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if path_kind == "symlink":
        link = workspace / "linked"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError:
            pytest.skip("Symlink creation is unavailable")
        output_path = "linked/file.txt"
    elif path_kind == "relative":
        output_path = "../outside/file.txt"
    else:
        output_path = str(outside / "file.txt")
    monkeypatch.setattr(
        "ursa.tools.read_file_tool.requests.get",
        lambda *args, **kwargs: pytest.fail(
            "Rejected destinations must not start a download"
        ),
    )

    result = download_file_tool.func(
        url="https://example.com/file.txt",
        output_path=output_path,
        runtime=make_runtime(workspace, llm=chat_model),
    )

    assert "resolves outside workspace directory" in result
    assert not (outside / "file.txt").exists()


def _read_skill(path, workspace, chat_model):
    return invoke_with_parent_run(
        lambda config: read_file.func(
            filename=str(path),
            runtime=make_runtime(workspace, llm=chat_model, config=config),
        )
    )


def test_read_file_accepts_each_configured_skills_directory(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    first = tmp_path / "first-skills"
    second = tmp_path / "second-skills"
    first.mkdir()
    second.mkdir()
    (first / "one.md").write_text("first skill", encoding="utf-8")
    (second / "two.md").write_text("second skill", encoding="utf-8")
    monkeypatch.setenv(
        "URSA_SKILLS_PATH", os.pathsep.join((str(first), str(second)))
    )

    assert _read_skill(first / "one.md", workspace, chat_model) == "first skill"
    assert (
        _read_skill(second / "two.md", workspace, chat_model) == "second skill"
    )


def test_read_file_resolves_configured_skills_symlink(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "real-skills"
    target.mkdir()
    skill = target / "skill.md"
    skill.write_text("symlinked skill", encoding="utf-8")
    link = tmp_path / "skills-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("Symlink creation is unavailable")
    monkeypatch.setenv("URSA_SKILLS_PATH", str(link))

    assert _read_skill(skill, workspace, chat_model) == "symlinked skill"


def test_read_file_accepts_cwd_local_skills_outside_workspace(
    monkeypatch, tmp_path: Path, chat_model: BaseChatModel
):
    workspace = tmp_path / "project" / "workspace"
    workspace.mkdir(parents=True)
    local_skills = tmp_path / ".ursa" / "skills"
    local_skills.mkdir(parents=True)
    skill = local_skills / "local.md"
    skill.write_text("local skill", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("URSA_SKILLS_PATH", raising=False)

    assert _read_skill(skill, workspace, chat_model) == "local skill"
