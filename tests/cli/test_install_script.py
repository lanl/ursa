import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
INSTALL_SCRIPT = ROOT / "install.sh"
POWERSHELL_INSTALL_SCRIPT = ROOT / "install.ps1"
RUN_INSTALLER_SCRIPT = ROOT / "scripts" / "run_installer.sh"
GETTING_STARTED = ROOT / "docs" / "getting-started" / "index.md"
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")
GIT_REQUIREMENT = "ursa-ai @ git+https://github.com/lanl/ursa.git@"


def _write_executable(path: Path, contents: str) -> Path:
    path.write_text(contents)
    path.chmod(0o755)
    return path


def _requires_posix(reason: str):
    return pytest.mark.skipif(os.name != "posix", reason=reason)


def _target(arguments, requirement, python="3.13"):
    return arguments.split(), python, requirement


def _fake_installation(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    bin_dir = tmp_path / "bin"
    tool_bin = tmp_path / "tool-bin"
    for directory in (bin_dir, tool_bin):
        directory.mkdir()
    log = tmp_path / "commands.log"
    _write_executable(
        bin_dir / "uv",
        "#!/bin/sh\n"
        'printf "uv %s\\n" "$*" >> "$INSTALL_LOG"\n'
        '[ "$1 $2 $3" = "tool dir --bin" ] && printf "%s\\n" "$TOOL_BIN" || :\n',
    )
    ursa = _write_executable(
        tool_bin / "ursa",
        '#!/bin/sh\nprintf "ursa %s\\n" "$*" >> "$INSTALL_LOG"\n',
    )
    environment = {
        "HOME": str(tmp_path),
        "INSTALL_LOG": str(log),
        "PATH": f"{bin_dir}:{tool_bin}:/usr/bin:/bin",
        "TOOL_BIN": str(tool_bin),
    }
    return environment, log, ursa


def _run_installer(environment, *, arguments=(), check=True, **options):
    options.setdefault("stdin", subprocess.DEVNULL)
    return subprocess.run(
        ["sh", INSTALL_SCRIPT, *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=check,
        **options,
    )


def test_installer_sources():
    powershell = POWERSHELL_INSTALL_SCRIPT.read_text()
    assert all(
        value in powershell
        for value in (
            '$ursaRequirement = "ursa-ai"',
            '$ursaRequirement = "ursa-ai==$Version"',
            "git+https://github.com/lanl/ursa.git@$Branch",
            '$setupArguments += @("--defaults-json", $Config)',
        )
    )

    getting_started = GETTING_STARTED.read_text()
    assert '=== "Linux/MacOS"' in getting_started
    assert '=== "Windows"' in getting_started
    assert "https://lanl.github.io/ursa/latest/install.ps1" in getting_started


@pytest.mark.parametrize("script", [INSTALL_SCRIPT, RUN_INSTALLER_SCRIPT])
def test_installer_scripts_are_posix_shell(script):
    subprocess.run(["sh", "-n", script], check=True)


def test_installer_harness_forwards_options_to_default_installer(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    _write_executable(
        bin_dir / "docker",
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n',
    )
    environment = {
        **os.environ,
        "DOCKER_LOG": str(docker_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    defaults = tmp_path / "defaults.json"
    defaults.write_text("{}")
    subprocess.run(
        [
            "sh",
            RUN_INSTALLER_SCRIPT,
            "--config",
            str(defaults),
            *"--branch feature/setup --python 3.12".split(),
        ],
        env=environment,
        check=True,
    )
    docker_run = docker_log.read_text().splitlines()[-1]
    assert docker_run.endswith(
        f"ursa-installer-test:ubuntu-24.04 --config {defaults} "
        "--branch feature/setup --python 3.12"
    )
    assert f"src={defaults},dst={defaults},readonly" in docker_run


@_requires_posix("pseudo-terminal setup invocation uses POSIX file descriptors")
def test_install_script_passes_json_defaults_to_guided_setup(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    payload = tmp_path / "defaults.json"
    payload.write_text("{}")
    master, slave = os.openpty()
    try:
        _run_installer(
            environment,
            arguments=("--config", payload),
            stdin=slave,
        )
    finally:
        os.close(master)
        os.close(slave)
    setup = next(
        command
        for command in log.read_text().splitlines()
        if command.startswith("ursa self setup --defaults-json ")
    )
    defaults_path = Path(setup.removeprefix("ursa self setup --defaults-json "))
    assert defaults_path == payload


@pytest.mark.parametrize(
    ("arguments", "python_version", "requirement"),
    [
        _target("--version 2.4.1", "ursa-ai==2.4.1"),
        _target("--branch feature/setup", GIT_REQUIREMENT + "feature/setup"),
        _target("--version 2.4.1 --python 3.11", "ursa-ai==2.4.1", "3.11"),
        _target(
            "--version 1.0 --version 2.0 --python 3.12 --python 3.13",
            "ursa-ai==2.0",
        ),
    ],
)
def test_install_script_selects_target_and_python(
    tmp_path, arguments, python_version, requirement
):
    environment, log, _ursa = _fake_installation(tmp_path)
    _run_installer(environment, arguments=arguments)
    expected = f"uv tool install --python {python_version} --force --upgrade "
    expected += f"--compile-bytecode {requirement}"
    assert expected in log.read_text().splitlines()


def test_install_script_rejects_branch_with_version(tmp_path):
    environment, _log, _ursa = _fake_installation(tmp_path)
    result = _run_installer(
        environment,
        arguments="--branch main --version 1.0".split(),
        check=False,
    )
    assert result.returncode != 0


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_powershell_installer_has_valid_syntax():
    path = str(POWERSHELL_INSTALL_SCRIPT).replace("'", "''")
    command = (
        "$e=$null;$t=$null;"
        f"[System.Management.Automation.Language.Parser]::ParseFile('{path}',"
        "[ref]$t,[ref]$e);if($e){exit 1}"
    )
    subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-Command", command],
        check=True,
    )


@_requires_posix("uv's shell installer targets macOS and Linux")
def test_install_script_bootstraps_uv_when_missing(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    fake_uv = tmp_path / "bin" / "uv"
    uv_payload = tmp_path / "uv-payload"
    uv_payload.write_text(fake_uv.read_text())
    uv_payload.chmod(0o755)
    fake_uv.unlink()
    _write_executable(
        tmp_path / "bin" / "curl",
        """#!/bin/sh
cat <<'EOF'
mkdir -p "$HOME/.local/bin"
cp "$UV_PAYLOAD" "$HOME/.local/bin/uv"
chmod +x "$HOME/.local/bin/uv"
EOF
""",
    )
    environment["UV_PAYLOAD"] = str(uv_payload)
    result = _run_installer(environment)
    assert "Installing uv" in result.stdout
    assert (tmp_path / ".local" / "bin" / "uv").is_file()
    assert "uv tool install --python 3.13" in log.read_text()
    assert "URSA is installed" in result.stdout


def test_install_script_reports_version_when_setup_is_unavailable(tmp_path):
    environment, _log, ursa = _fake_installation(tmp_path)
    _write_executable(
        ursa,
        '#!/bin/sh\n[ "$1" = "--version" ] && printf "ursa 2.4.1\\n"\n',
    )
    result = _run_installer(environment, check=False)
    assert result.returncode != 0
    assert (
        "installed URSA (ursa 2.4.1) does not support guided setup"
        in result.stderr
    )
    assert "https://lanl.github.io/ursa/latest/getting-started" in result.stderr
