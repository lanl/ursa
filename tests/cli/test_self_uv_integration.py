"""Integration coverage for self-management against an isolated uv store."""

from __future__ import annotations

import re
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def _run(
    argv: list[str | Path], *, env: dict[str, str]
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(value) for value in argv],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.mark.slow
def test_self_commands_with_real_isolated_uv(tmp_path):
    """Exercise uv itself without reading or writing the user's tool store."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not installed")

    repository = Path(__file__).resolve().parents[2]
    tool_dir = tmp_path / "tools"
    bin_dir = tmp_path / "bin"
    env = {
        **os.environ,
        "UV_TOOL_DIR": str(tool_dir),
        "UV_TOOL_BIN_DIR": str(bin_dir),
        "UV_PYTHON_DOWNLOADS": "never",
    }

    _run(
        [uv, "tool", "install", repository, "--python", sys.executable],
        env=env,
    )
    ursa = bin_dir / ("ursa.exe" if os.name == "nt" else "ursa")

    status = _run([ursa, "self", "status"], env=env).stdout
    assert re.search(r"^Version: .+", status, re.MULTILINE)
    assert re.search(r"^Python: \d+\.\d+\.\d+ \([^)]+\)$", status, re.MULTILINE)
    assert re.search(r"^Python path: .+", status, re.MULTILINE)
    assert re.search(r"^Extras: none$", status, re.MULTILINE)
    assert re.search(r"^Additional packages: none$", status, re.MULTILINE)

    _run([ursa, "self", "update"], env=env)
    updated_status = _run([ursa, "self", "status"], env=env).stdout
    assert re.search(
        r"^Additional packages: none$", updated_status, re.MULTILINE
    )

    _run([ursa, "self", "modify", "--with", "pytest"], env=env)
    modified_status = _run([ursa, "self", "status"], env=env).stdout
    if os.name != "nt":
        assert re.search(
            r"^Additional packages: pytest$", modified_status, re.MULTILINE
        )

    _run([ursa, "self", "modify", "--clean"], env=env)
    cleaned_status = _run([ursa, "self", "status"], env=env).stdout
    assert re.search(
        r"^Additional packages: none$", cleaned_status, re.MULTILINE
    )
