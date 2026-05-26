from __future__ import annotations

from pathlib import Path

from ursa_dashboard.artifacts import scan_artifacts


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_scan_artifacts_hides_dotfiles_and_hidden_directories(
    tmp_path: Path,
) -> None:
    _touch(tmp_path / "visible.txt")
    _touch(tmp_path / ".hidden.txt")
    _touch(tmp_path / ".git" / "config")
    _touch(tmp_path / ".venv" / "pyvenv.cfg")
    _touch(tmp_path / "outputs" / "result.txt")
    _touch(tmp_path / "outputs" / ".secret")
    _touch(tmp_path / "outputs" / ".cache" / "data.json")

    rel_paths = {entry["rel_path"] for entry in scan_artifacts(tmp_path)}

    assert rel_paths == {"outputs/result.txt", "visible.txt"}
