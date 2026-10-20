import subprocess
from pathlib import Path

import pytest

from docs import hooks

INSTALL_SCRIPT = Path(__file__).parents[2] / "install.sh"
SITE_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "site.yml"


def _fake_installation(
    tmp_path: Path, *, path: str | None = None
) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    tool_bin = tmp_path / "tool-bin"
    bin_dir.mkdir()
    tool_bin.mkdir()
    log = tmp_path / "commands.log"
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'printf "uv %s\\n" "$*" >> "$INSTALL_LOG"\n'
        'case "$1 $2 $3" in\n'
        '  "tool dir --bin") printf "%s\\n" "$TOOL_BIN" ;;\n'
        '  "tool update-shell ") printf "source %s/env\\n" "$HOME/.local/bin"; '
        'exit "${UPDATE_SHELL_STATUS:-0}" ;;\n'
        "esac\n"
    )
    uv.chmod(0o755)
    ursa = tool_bin / "ursa"
    ursa.write_text('#!/bin/sh\nprintf "ursa %s\\n" "$*" >> "$INSTALL_LOG"\n')
    ursa.chmod(0o755)
    environment = {
        "HOME": str(tmp_path),
        "INSTALL_LOG": str(log),
        "PATH": path or f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }
    return environment, log, ursa


def _run_installer(environment, *, script=None, check=True):
    return subprocess.run(
        ["sh", INSTALL_SCRIPT] if script is None else ["sh"],
        env=environment,
        stdin=subprocess.DEVNULL if script is None else None,
        input=script,
        capture_output=True,
        text=True,
        check=check,
    )


def test_install_script_is_posix_shell():
    subprocess.run(["sh", "-n", INSTALL_SCRIPT], check=True)


@pytest.mark.parametrize("commit", [None, "0123456789abcdef"])
def test_install_script_uses_existing_uv_and_tool_bin(tmp_path, commit):
    environment, log, _ursa = _fake_installation(tmp_path)
    if commit is not None:
        environment["URSA_COMMIT"] = commit

    result = _run_installer(environment)

    commands = log.read_text()
    revision = commit or "main"
    assert (
        "uv tool install --python 3.13 --force --upgrade "
        "--compile-bytecode ursa-ai[dashboard] @ "
        f"git+https://github.com/lanl/ursa.git@{revision}" in commands
    )
    assert "ursa self modify" not in commands
    assert "uv tool dir --bin" in commands
    assert "URSA is installed" in result.stdout


def test_install_script_bootstraps_uv_when_missing(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    fake_uv = tmp_path / "bin" / "uv"
    uv_payload = tmp_path / "uv-payload"
    uv_payload.write_text(fake_uv.read_text())
    uv_payload.chmod(0o755)
    fake_uv.unlink()

    uv_installer = tmp_path / "uv-installer.sh"
    uv_installer.write_text(
        "#!/bin/sh\n"
        'mkdir -p "$HOME/.local/bin"\n'
        'cp "$UV_PAYLOAD" "$HOME/.local/bin/uv"\n'
        'chmod +x "$HOME/.local/bin/uv"\n'
    )
    curl = tmp_path / "bin" / "curl"
    curl.write_text('#!/bin/sh\ncp "$UV_INSTALLER" "$4"\n')
    curl.chmod(0o755)
    environment["UV_INSTALLER"] = str(uv_installer)
    environment["UV_PAYLOAD"] = str(uv_payload)

    result = _run_installer(environment)

    assert "Installing uv" in result.stdout
    assert (tmp_path / ".local" / "bin" / "uv").is_file()
    assert "uv tool install --python 3.13" in log.read_text()
    assert "URSA is installed" in result.stdout


def test_install_script_explains_release_mismatch(tmp_path):
    environment, _log, ursa = _fake_installation(tmp_path)
    ursa.write_text("#!/bin/sh\nexit 1\n")
    ursa.chmod(0o755)

    result = _run_installer(environment, check=False)

    assert result.returncode != 0
    assert "may target an unreleased URSA version" in result.stderr
    assert "https://lanl.github.io/ursa/latest/install.sh" in result.stderr


@pytest.mark.parametrize(
    ("update_status", "expected"),
    [
        (None, "source "),
        ("2", "Could not update this shell automatically"),
    ],
)
def test_install_script_repairs_uv_tool_bin_missing_from_path(
    tmp_path, update_status, expected
):
    bin_dir = tmp_path / "bin"
    environment, _log, _ursa = _fake_installation(
        tmp_path, path=f"{bin_dir}:/usr/bin:/bin"
    )
    if update_status is not None:
        environment["UPDATE_SHELL_STATUS"] = update_status

    result = _run_installer(environment)

    assert "Adding uv's tool directory to PATH" in result.stdout
    assert expected in result.stdout
    assert "Or restart your shell" in result.stdout
    assert "URSA is installed" in result.stdout


def test_install_warns_when_another_ursa_shadows_the_install(tmp_path):
    environment, _log, ursa = _fake_installation(tmp_path)
    old_ursa = Path(environment["PATH"].split(":", 1)[0]) / "ursa"
    old_ursa.write_text("#!/bin/sh\nexit 0\n")
    old_ursa.chmod(0o755)

    result = _run_installer(environment)

    assert f"Warning: {old_ursa} currently takes precedence over {ursa}" in (
        result.stdout
    )
    assert f"Run '{ursa}'" in result.stdout


def test_docs_builds_main_installer(monkeypatch):
    monkeypatch.setenv("MIKE_DOCS_VERSION", "main")
    assert (
        "git+https://github.com/lanl/ursa.git@main" in hooks._installer_script()
    )


def test_release_installer_ignores_commit_override_and_streams(
    monkeypatch, tmp_path
):
    environment, log, _ursa = _fake_installation(tmp_path)
    environment["URSA_COMMIT"] = "0123456789abcdef"
    monkeypatch.setenv("MIKE_DOCS_VERSION", "2.4.1")

    _run_installer(environment, script=hooks._installer_script())

    commands = log.read_text()
    assert "ursa-ai[dashboard]==2.4.1" in commands
    assert "0123456789abcdef" not in commands


def test_docs_rejects_unsafe_installer_version(monkeypatch):
    monkeypatch.setenv("MIKE_DOCS_VERSION", "1.0'; touch /tmp/owned; '")

    with pytest.raises(ValueError, match="Unsafe MIKE_DOCS_VERSION"):
        hooks._installer_script()


def test_docs_workflow_publishes_versioned_installers():
    workflow = SITE_WORKFLOW.read_text()

    assert "mike deploy" in workflow
    assert '"$version" latest' in workflow
    assert 'mike deploy "${push_args[@]}" main' in workflow
    assert 'cp "$GITHUB_WORKSPACE/install.sh"' not in workflow
