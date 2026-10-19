import subprocess
from pathlib import Path

INSTALL_SCRIPT = Path(__file__).parents[2] / "install.sh"
SITE_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "site.yml"


def test_install_script_is_posix_shell_and_runs_guided_setup():
    subprocess.run(["sh", "-n", INSTALL_SCRIPT], check=True)
    script = INSTALL_SCRIPT.read_text()

    assert "https://astral.sh/uv/install.sh" in script
    assert "--force --upgrade --compile-bytecode 'ursa-ai[dashboard]'" in script
    assert "tool dir --bin" in script
    assert "self setup </dev/tty" in script


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
        "PATH": f"{bin_dir}:/usr/bin:/bin",
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
    assert "uv tool install --force --upgrade --compile-bytecode" in commands
    assert "uv tool dir --bin" in commands
    assert "URSA is installed" in result.stdout


def test_docs_workflow_publishes_the_installer_at_the_site_root():
    workflow = SITE_WORKFLOW.read_text()

    assert "- install.sh" in workflow
    assert (
        'cp "$GITHUB_WORKSPACE/install.sh" "$docs_root/install.sh"' in workflow
    )
    assert "push origin HEAD:gh-pages" in workflow
