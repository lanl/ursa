import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
INSTALL = ROOT / "install.sh"
POWERSHELL_INSTALL = ROOT / "install.ps1"
HARNESS = ROOT / "scripts/run_installer.sh"


def executable(path, text):
    path.write_text(text)
    path.chmod(0o755)


def fake_installation(tmp_path):
    bin_dir, tool_bin = tmp_path / "bin", tmp_path / "tools"
    bin_dir.mkdir()
    tool_bin.mkdir()
    log = tmp_path / "commands"
    executable(
        bin_dir / "uv",
        '#!/bin/sh\necho "uv $*" >> "$LOG"\n'
        '[ "$1 $2 $3" = "tool dir --bin" ] && echo "$TOOL_BIN" || :\n',
    )
    executable(
        tool_bin / "ursa",
        '#!/bin/sh\necho "ursa $*" >> "$LOG"\n'
        '[ "$1" != --version ] || echo "ursa 2.4.1"\n',
    )
    return {
        "HOME": str(tmp_path),
        "LOG": str(log),
        "PATH": f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }, log


def fake_uv_bootstrap(tmp_path):
    bin_dir, tool_bin = tmp_path / "bin", tmp_path / "tools"
    bin_dir.mkdir()
    tool_bin.mkdir()
    log = tmp_path / "commands"
    uv_source = tmp_path / "uv"
    executable(
        uv_source,
        '#!/bin/sh\necho "uv $*" >> "$LOG"\n'
        '[ "$1 $2 $3" = "tool dir --bin" ] && echo "$TOOL_BIN" || :\n',
    )
    executable(
        tool_bin / "ursa",
        '#!/bin/sh\necho "ursa $*" >> "$LOG"\n',
    )
    uv_installer = tmp_path / "uv-installer.sh"
    executable(
        uv_installer,
        '#!/bin/sh\nmkdir -p "$HOME/.local/bin"\n'
        'cp "$FAKE_UV" "$HOME/.local/bin/uv"\n'
        'echo "uv installer executed" >> "$LOG"\n',
    )
    executable(
        bin_dir / "curl",
        '#!/bin/sh\nwhile [ "$#" -gt 0 ]; do\n'
        '    if [ "$1" = -o ]; then cp "$UV_INSTALLER" "$2"; exit; fi\n'
        "    shift\n"
        "done\nexit 1\n",
    )
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    return (
        {
            "FAKE_UV": str(uv_source),
            "HOME": str(tmp_path),
            "LOG": str(log),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "TMPDIR": str(downloads),
            "TOOL_BIN": str(tool_bin),
            "UV_INSTALLER": str(uv_installer),
        },
        log,
        uv_installer,
        downloads,
    )


def run_install(env, *arguments, check=True, stdin=subprocess.DEVNULL):
    return subprocess.run(
        ["sh", INSTALL, *arguments],
        env=env,
        stdin=stdin,
        capture_output=True,
        text=True,
        check=check,
    )


def test_installer_syntax_and_docs():
    docs = (ROOT / "docs/getting-started/index.md").read_text()
    assert all(
        token in docs for token in ('=== "Linux/MacOS"', '=== "Windows"')
    )
    for script in (INSTALL, HARNESS):
        subprocess.run(["sh", "-n", script], check=True)
    powershell_text = POWERSHELL_INSTALL.read_text()
    assert powershell_text.index('"uv installer path: $uvInstaller"') < (
        powershell_text.index('Read-Host "Install uv? [y/N]"')
    )
    if powershell := shutil.which("pwsh") or shutil.which("powershell"):
        path = str(POWERSHELL_INSTALL).replace("'", "''")
        command = (
            "$e=$null;$t=$null;"
            f"[Management.Automation.Language.Parser]::ParseFile('{path}',"
            "[ref]$t,[ref]$e);if($e){exit 1}"
        )
        subprocess.run(
            [powershell, "-NoProfile", "-Command", command], check=True
        )


@pytest.mark.skipif(os.name != "posix", reason="PTY uses POSIX descriptors")
def test_uv_installation_reports_path_and_requires_confirmation(tmp_path):
    env, log, uv_installer, downloads = fake_uv_bootstrap(tmp_path)

    master, slave = os.openpty()
    try:
        os.write(master, b"yes\n")
        result = run_install(env, stdin=slave)
    finally:
        os.close(master)
        os.close(slave)

    output = result.stdout
    path_line = next(
        line
        for line in output.splitlines()
        if line.startswith("uv installer path: ")
    )
    downloaded_installer = Path(path_line.removeprefix("uv installer path: "))
    assert downloaded_installer.parent == downloads
    assert output.index(path_line) < output.index("Install uv? [y/N]")
    assert "uv installer executed" in log.read_text()
    assert not downloaded_installer.exists()
    assert not list(downloads.iterdir())

    shutil.rmtree(tmp_path / ".local")
    log.write_text("")
    master, slave = os.openpty()
    try:
        os.write(master, b"n\n")
        result = run_install(env, check=False, stdin=slave)
    finally:
        os.close(master)
        os.close(slave)

    assert result.returncode
    assert "uv installation declined" in result.stderr
    assert not log.exists() or "uv installer executed" not in log.read_text()
    assert not list(downloads.iterdir())


@pytest.mark.skipif(os.name != "posix", reason="PTY uses POSIX descriptors")
def test_installer_public_behavior(tmp_path):
    env, log = fake_installation(tmp_path)
    cases = (
        (("--version", "2.4.1"), "3.13", "ursa-ai==2.4.1"),
        (
            ("--branch", "feature/setup"),
            "3.13",
            "ursa-ai @ git+https://github.com/lanl/ursa.git@feature/setup",
        ),
        (
            ("--version", "1", "--version", "2", "--python", "3.11"),
            "3.11",
            "ursa-ai==2",
        ),
    )
    for arguments, python, requirement in cases:
        log.write_text("")
        result = run_install(env, *arguments)
        assert f"uv tool install --python {python}" in log.read_text()
        assert requirement in log.read_text()
        assert "URSA is installed" in result.stdout
    assert run_install(
        env, "--branch", "main", "--version", "1", check=False
    ).returncode

    defaults = tmp_path / "defaults.json"
    defaults.write_text("{}")
    master, slave = os.openpty()
    try:
        run_install(env, "--config", str(defaults), stdin=slave)
    finally:
        os.close(master)
        os.close(slave)
    assert f"ursa self setup --defaults-json {defaults}" in log.read_text()

    executable(
        tmp_path / "tools/ursa",
        '#!/bin/sh\n[ "$1" = --version ] && echo "ursa 2.4.1"\nexit 1\n',
    )
    result = run_install(env, check=False)
    assert "installed URSA (ursa 2.4.1)" in result.stderr
    assert "https://lanl.github.io/ursa/latest/getting-started" in result.stderr


@pytest.mark.skipif(os.name != "posix", reason="POSIX bind mounts")
def test_installer_harness_forwards_arguments(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "docker.log"
    executable(bin_dir / "docker", '#!/bin/sh\necho "$*" >> "$LOG"\n')
    config = tmp_path / "defaults.json"
    config.write_text("{}")
    subprocess.run(
        ["sh", HARNESS, "--config", str(config), "--branch", "main"],
        env={
            **os.environ,
            "LOG": str(log),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
        check=True,
    )
    command = log.read_text().splitlines()[-1]
    assert f"src={config},dst={config},readonly" in command
    assert command.endswith(f"--config {config} --branch main")
