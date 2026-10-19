import subprocess
from pathlib import Path

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
        'if [ "$1 $2 $3" = "tool dir --bin" ]; then\n'
        '  printf "%s\\n" "$TOOL_BIN"\n'
        'elif [ "$1 $2" = "tool update-shell" ]; then\n'
        '  printf "source %s/env\\n" "$HOME/.local/bin"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    ursa = tool_bin / "ursa"
    ursa.write_text(
        "#!/bin/sh\n"
        'printf "ursa %s\\n" "$*" >> "$INSTALL_LOG"\n'
        'case "$*" in\n'
        '  "self modify --help") printf "%s\\n" "--installer-config" ;;\n'
        '  "self setup --help") printf "%s\\n" "--default-json" ;;\n'
        '  "--version") printf "%s\\n" "ursa 1.0.0" ;;\n'
        "esac\n"
    )
    ursa.chmod(0o755)
    environment = {
        "HOME": str(tmp_path),
        "INSTALL_LOG": str(log),
        "PATH": path or f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }
    return environment, log, ursa


def test_install_script_is_posix_shell_and_runs_guided_setup():
    subprocess.run(["sh", "-n", INSTALL_SCRIPT], check=True)
    script = INSTALL_SCRIPT.read_text()

    assert "https://astral.sh/uv/install.sh" in script
    assert "--python 3.13" in script
    assert "self modify --installer-config" in script
    assert "Usage: install.sh [--config FILE]" in script
    assert "--config-stdin" not in script
    assert "tool dir --bin" in script
    assert '"$ursa_executable" "$@" </dev/tty' in script
    assert "import json" not in script
    assert "python3" not in script


def test_install_script_uses_existing_uv_and_tool_bin(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )

    commands = log.read_text()
    assert (
        "uv tool install --python 3.13 --force --upgrade "
        "--compile-bytecode ursa-ai[dashboard] @ "
        "git+https://github.com/lanl/ursa.git@main" in commands
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
    curl.write_text(
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = "-o" ]; then cp "$UV_INSTALLER" "$2"; exit; fi\n'
        "  shift\n"
        "done\n"
        "exit 1\n"
    )
    curl.chmod(0o755)
    environment["UV_INSTALLER"] = str(uv_installer)
    environment["UV_PAYLOAD"] = str(uv_payload)

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Installing uv" in result.stdout
    assert (tmp_path / ".local" / "bin" / "uv").is_file()
    assert "uv tool install --python 3.13" in log.read_text()
    assert "URSA is installed" in result.stdout


def test_install_script_rejects_concurrent_installation(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    lock = tmp_path / ".cache" / "ursa" / "install.lock"
    lock.mkdir(parents=True)

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "another URSA installation is already running" in result.stderr
    assert lock.is_dir()
    assert not log.exists()


def test_install_script_applies_json_install_options_from_file(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    config = tmp_path / "installer.json"
    config.write_text(
        '{"python_version":"3.12","extras":["dashboard","fm"],'
        '"extra_packages":["acme-plugin>=1"],'
        '"inference_providers":[],"mcp_servers":{}}'
    )

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT, "--config", str(config)],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )

    commands = log.read_text()
    assert (
        "uv tool install --python 3.13 --force --upgrade --compile-bytecode "
        "ursa-ai @ git+https://github.com/lanl/ursa.git@main" in commands
    )
    assert f"ursa self modify --installer-config {config}" in commands
    assert f"ursa self setup --default-json {config}" in result.stdout


def test_install_script_explains_release_mismatch(tmp_path):
    environment, _log, ursa = _fake_installation(tmp_path)
    ursa.write_text("#!/bin/sh\nexit 0\n")
    ursa.chmod(0o755)
    config = tmp_path / "installer.json"
    config.write_text("{}")

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT, "--config", str(config)],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "may target an unreleased URSA version" in result.stderr
    assert "https://lanl.github.io/ursa/latest/install.sh" in result.stderr


def test_install_script_warns_and_lists_all_ursa_paths(tmp_path):
    bin_dir = tmp_path / "bin"
    old_bin = tmp_path / "old-bin"
    tool_bin = tmp_path / "tool-bin"
    old_bin.mkdir()
    environment, _log, installed_ursa = _fake_installation(
        tmp_path,
        path=f"{bin_dir}:{old_bin}:{tool_bin}:/usr/bin:/bin",
    )
    old_ursa = old_bin / "ursa"
    old_ursa.write_text('#!/bin/sh\nprintf "ursa 0.9.0\\n"\n')
    old_ursa.chmod(0o755)

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Warning: other URSA executables are also on PATH" in result.stdout
    assert f"- {old_ursa} (other installation; ursa 0.9.0)" in result.stdout
    assert f"- {installed_ursa} (installed now; ursa 1.0.0)" in result.stdout
    assert f"resolves 'ursa' to: {old_ursa}" in result.stdout


def test_install_script_repairs_uv_tool_bin_missing_from_path(tmp_path):
    bin_dir = tmp_path / "bin"
    environment, _log, _ursa = _fake_installation(
        tmp_path, path=f"{bin_dir}:/usr/bin:/bin"
    )

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Adding uv's tool directory to PATH" in result.stdout
    assert "source " in result.stdout
    assert "Or restart your shell" in result.stdout
    assert "URSA is installed" in result.stdout


def test_docs_builds_release_and_main_installers(monkeypatch):
    monkeypatch.setenv("MIKE_DOCS_VERSION", "2.4.1")
    release = hooks._installer_script()
    assert "URSA_INSTALL_REQUIREMENT='ursa-ai==2.4.1'" in release
    assert "URSA_DASHBOARD_REQUIREMENT='ursa-ai[dashboard]==2.4.1'" in release

    monkeypatch.setenv("MIKE_DOCS_VERSION", "main")
    main = hooks._installer_script()
    assert "git+https://github.com/lanl/ursa.git@main" in main


def test_docs_workflow_publishes_versioned_installers():
    workflow = SITE_WORKFLOW.read_text()

    assert "mike deploy" in workflow
    assert '"$version" latest' in workflow
    assert 'mike deploy "${push_args[@]}" main' in workflow
    assert 'cp "$GITHUB_WORKSPACE/install.sh"' not in workflow
