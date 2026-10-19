import json
import subprocess
from pathlib import Path

INSTALL_SCRIPT = Path(__file__).parents[2] / "install.sh"
SITE_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "site.yml"


def test_install_script_is_posix_shell_and_runs_guided_setup():
    subprocess.run(["sh", "-n", INSTALL_SCRIPT], check=True)
    script = INSTALL_SCRIPT.read_text()

    assert "https://astral.sh/uv/install.sh" in script
    assert "--python 3.13" in script
    assert "self modify --clean --extra dashboard" in script
    assert "self modify --installer-config" in script
    assert "--config-stdin" in script
    assert "tool dir --bin" in script
    assert '"$ursa_executable" "$@" </dev/tty' in script
    assert "import json" not in script
    assert "python3" not in script


def test_install_script_uses_existing_uv_and_tool_bin(tmp_path):
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
        "fi\n"
    )
    uv.chmod(0o755)
    ursa = tool_bin / "ursa"
    ursa.write_text('#!/bin/sh\nprintf "ursa %s\\n" "$*" >> "$INSTALL_LOG"\n')
    ursa.chmod(0o755)
    environment = {
        "HOME": str(tmp_path),
        "INSTALL_LOG": str(log),
        "PATH": f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }

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
        "--compile-bytecode ursa-ai" in commands
    )
    assert "ursa self modify --clean --extra dashboard" in commands
    assert "uv tool dir --bin" in commands
    assert "URSA is installed" in result.stdout


def test_install_script_applies_json_install_options_from_stdin(tmp_path):
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
        "fi\n"
    )
    uv.chmod(0o755)
    ursa = tool_bin / "ursa"
    ursa.write_text('#!/bin/sh\nprintf "ursa %s\\n" "$*" >> "$INSTALL_LOG"\n')
    ursa.chmod(0o755)
    payload = {
        "python_version": "3.12",
        "extras": ["dashboard", "fm"],
        "extra_packages": ["acme-plugin>=1", "other-package"],
        "inference_providers": [
            {
                "name": "company",
                "base_url": "https://models.example/v1",
            }
        ],
        "mcp_servers": {
            "tools": {
                "transport": "streamable_http",
                "url": "https://tools.example/mcp",
            }
        },
    }
    environment = {
        "HOME": str(tmp_path),
        "INSTALL_LOG": str(log),
        "PATH": f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }

    subprocess.run(
        ["sh", INSTALL_SCRIPT, "--config-stdin"],
        env=environment,
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
    )

    commands = log.read_text()
    assert (
        "uv tool install --python 3.13 --force --upgrade --compile-bytecode "
        "ursa-ai" in commands
    )
    assert "ursa self modify --installer-config" in commands


def test_install_script_warns_and_lists_all_ursa_paths(tmp_path):
    bin_dir = tmp_path / "bin"
    old_bin = tmp_path / "old-bin"
    tool_bin = tmp_path / "tool-bin"
    for directory in (bin_dir, old_bin, tool_bin):
        directory.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "tool dir --bin" ]; then\n'
        '  printf "%s\\n" "$TOOL_BIN"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    old_ursa = old_bin / "ursa"
    old_ursa.write_text('#!/bin/sh\nprintf "ursa 0.9.0\\n"\n')
    old_ursa.chmod(0o755)
    installed_ursa = tool_bin / "ursa"
    installed_ursa.write_text('#!/bin/sh\nprintf "ursa 1.0.0\\n"\n')
    installed_ursa.chmod(0o755)
    environment = {
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}:{old_bin}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }

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
    tool_bin = tmp_path / "tool-bin"
    bin_dir.mkdir()
    tool_bin.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/bin/sh\n"
        'if [ "$1 $2 $3" = "tool dir --bin" ]; then\n'
        '  printf "%s\\n" "$TOOL_BIN"\n'
        "fi\n"
    )
    uv.chmod(0o755)
    ursa = tool_bin / "ursa"
    ursa.write_text("#!/bin/sh\n")
    ursa.chmod(0o755)

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT],
        env={
            "HOME": str(tmp_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "TOOL_BIN": str(tool_bin),
        },
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "Adding uv's tool directory to PATH" in result.stdout
    assert "URSA is installed" in result.stdout


def test_docs_workflow_publishes_the_installer_at_the_site_root():
    workflow = SITE_WORKFLOW.read_text()

    assert "- install.sh" in workflow
    assert (
        'cp "$GITHUB_WORKSPACE/install.sh" "$docs_root/install.sh"' in workflow
    )
    assert "push origin HEAD:gh-pages" in workflow
