import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

INSTALL_SCRIPT = Path(__file__).parents[2] / "install.sh"
POWERSHELL_INSTALL_SCRIPT = Path(__file__).parents[2] / "install.ps1"
RUN_INSTALLER_SCRIPT = (
    Path(__file__).parents[2] / "scripts" / "run_installer.sh"
)
SITE_WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "site.yml"
DOCS_HOOKS = Path(__file__).parents[2] / "docs" / "hooks.py"
GETTING_STARTED = (
    Path(__file__).parents[2] / "docs" / "getting-started" / "index.md"
)
POWERSHELL = shutil.which("pwsh") or shutil.which("powershell")


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


def _run_installer(environment, *, arguments=(), script=None, check=True):
    return subprocess.run(
        ["sh", INSTALL_SCRIPT, *arguments] if script is None else ["sh"],
        env=environment,
        stdin=subprocess.DEVNULL if script is None else None,
        input=script,
        capture_output=True,
        text=True,
        check=check,
    )


def test_install_script_is_posix_shell():
    subprocess.run(["sh", "-n", INSTALL_SCRIPT], check=True)

    script = INSTALL_SCRIPT.read_text(encoding="utf-8")
    assert "--config FILE" in script
    assert "--branch BRANCH | --version VERSION" in script
    assert "--python VERSION" in script
    assert "--config-json" not in script
    assert "--config-stdin" not in script
    assert 'set -- "$@" --defaults-json "$payload_file"' in script


def test_installer_harness_is_posix_shell():
    subprocess.run(["sh", "-n", RUN_INSTALLER_SCRIPT], check=True)

    result = subprocess.run(
        ["sh", RUN_INSTALLER_SCRIPT, "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Run an URSA installer in a clean Ubuntu container" in result.stdout

    script = RUN_INSTALLER_SCRIPT.read_text()
    assert '"$image" "$@"' in script


def test_installer_harness_forwards_options_to_default_installer(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$DOCKER_LOG"\n')
    docker.chmod(0o755)
    environment = {
        **os.environ,
        "DOCKER_LOG": str(docker_log),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }

    subprocess.run(
        [
            "sh",
            RUN_INSTALLER_SCRIPT,
            "--branch",
            "feature/setup",
            "--python",
            "3.12",
        ],
        env=environment,
        check=True,
    )

    docker_run = docker_log.read_text().splitlines()[-1]
    assert docker_run.endswith(
        "ursa-installer-test:ubuntu-24.04 --branch feature/setup --python 3.12"
    )


def test_powershell_installer_limits_package_sources():
    script = POWERSHELL_INSTALL_SCRIPT.read_text(encoding="utf-8")

    assert (
        "& $uvExecutable tool install --python $Python --force --upgrade "
        "--compile-bytecode $ursaRequirement" in script
    )
    assert '$ursaRequirement = "ursa-ai"' in script
    assert '$ursaRequirement = "ursa-ai==$Version"' in script
    assert (
        '$ursaRequirement = "ursa-ai @ '
        'git+https://github.com/lanl/ursa.git@$Branch"' in script
    )
    assert script.count("git+") == 1
    assert "--from" not in script
    assert (
        "installed URSA ($ursaVersion) does not support guided setup. See "
        "https://lanl.github.io/ursa/latest/getting-started." in script
    )
    assert "[string]$Config" in script
    assert "[string]$Version" in script
    assert "[string]$Branch" in script
    assert '[string]$Python = "3.13"' in script
    assert "ConfigJson" not in script
    assert '$setupArguments += @("--defaults-json", $payloadFile)' in script


@pytest.mark.skipif(
    os.name != "posix",
    reason="pseudo-terminal setup invocation uses POSIX file descriptors",
)
def test_install_script_passes_json_defaults_to_guided_setup(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    payload = tmp_path / "defaults.json"
    payload.write_text(
        json.dumps({
            "inference_providers": [
                {"name": "company", "base_url": "https://models.example/v1"}
            ]
        })
    )
    master, slave = os.openpty()
    try:
        subprocess.run(
            ["sh", INSTALL_SCRIPT, "--config", payload],
            env=environment,
            stdin=slave,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
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
    assert not defaults_path.exists()


@pytest.mark.skipif(
    os.name != "posix",
    reason="detaching the controlling terminal uses POSIX sessions",
)
def test_install_script_does_not_silently_drop_defaults_without_tty(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)
    payload = tmp_path / "defaults.json"
    payload.write_text("{}")

    result = subprocess.run(
        ["sh", INSTALL_SCRIPT, "--config", payload],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        start_new_session=True,
    )

    assert result.returncode != 0
    assert "setup defaults require an interactive terminal" in result.stderr
    assert not log.exists()


@pytest.mark.parametrize(
    ("arguments", "requirement"),
    [
        (("--version", "2.4.1"), "ursa-ai==2.4.1"),
        (("--version", "main"), "ursa-ai==main"),
        (
            ("--branch", "feature/setup"),
            "ursa-ai @ git+https://github.com/lanl/ursa.git@feature/setup",
        ),
        (
            ("--branch", "main"),
            "ursa-ai @ git+https://github.com/lanl/ursa.git@main",
        ),
    ],
)
def test_install_script_selects_pypi_version_or_branch(
    tmp_path, arguments, requirement
):
    environment, log, _ursa = _fake_installation(tmp_path)

    _run_installer(environment, arguments=arguments)

    assert (
        "uv tool install --python 3.13 --force --upgrade "
        f"--compile-bytecode {requirement}"
    ) in log.read_text().splitlines()


@pytest.mark.parametrize("python_version", ["3", "3.11", "3.13.1"])
def test_install_script_selects_python_version(tmp_path, python_version):
    environment, log, _ursa = _fake_installation(tmp_path)

    _run_installer(
        environment,
        arguments=(
            "--version",
            "2.4.1",
            "--python",
            python_version,
        ),
    )

    assert (
        f"uv tool install --python {python_version} --force --upgrade "
        "--compile-bytecode ursa-ai==2.4.1"
    ) in log.read_text().splitlines()


@pytest.mark.parametrize(
    "python_version",
    ["", "cpython", "3.x", "3..13", ".13", "3.", "3.13\n3.x"],
)
def test_install_script_rejects_invalid_python_version(
    tmp_path, python_version
):
    environment, log, _ursa = _fake_installation(tmp_path)

    result = _run_installer(
        environment,
        arguments=("--python", python_version),
        check=False,
    )

    assert result.returncode != 0
    assert "--python must be a Python version such as 3.13" in result.stderr
    assert not log.exists()


def test_install_script_uses_last_repeated_flag(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)

    _run_installer(
        environment,
        arguments=(
            "--version",
            "1.0",
            "--branch",
            "main",
            "--python",
            "3.12",
            "--python",
            "3.13",
        ),
    )

    assert (
        "uv tool install --python 3.13 --force --upgrade "
        "--compile-bytecode ursa-ai @ "
        "git+https://github.com/lanl/ursa.git@main"
    ) in log.read_text().splitlines()


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
def test_powershell_installer_has_valid_syntax():
    command = (
        "$tokens = $null; $parseErrors = $null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{str(POWERSHELL_INSTALL_SCRIPT).replace(chr(39), chr(39) * 2)}', "
        "[ref]$tokens, [ref]$parseErrors) | Out-Null; "
        "if ($parseErrors.Count) { "
        "$parseErrors | ForEach-Object { Write-Error $_ }; exit 1 }"
    )

    subprocess.run(
        [POWERSHELL, "-NoLogo", "-NoProfile", "-Command", command],
        check=True,
    )


@pytest.mark.skipif(POWERSHELL is None, reason="PowerShell is not installed")
@pytest.mark.parametrize(
    ("parameter", "value", "message"),
    [
        (
            "-Python",
            "3.x",
            "-Python must be a Python version such as 3.13",
        ),
        (
            "-Python",
            "3.13\n3.x",
            "-Python must be a Python version such as 3.13",
        ),
    ],
)
def test_powershell_installer_rejects_invalid_python_versions(
    parameter, value, message
):
    result = subprocess.run(
        [
            POWERSHELL,
            "-NoLogo",
            "-NoProfile",
            "-File",
            POWERSHELL_INSTALL_SCRIPT,
            parameter,
            value,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def test_install_script_uses_existing_uv_and_tool_bin(tmp_path):
    environment, log, _ursa = _fake_installation(tmp_path)

    result = _run_installer(environment)

    commands = log.read_text().splitlines()
    assert (
        "uv tool install --python 3.13 --force --upgrade "
        "--compile-bytecode ursa-ai" in commands
    )
    assert "uv tool dir --bin" in commands
    assert "URSA is installed" in result.stdout


@pytest.mark.skipif(
    os.name != "posix",
    reason="uv's shell installer targets macOS and Linux",
)
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


def test_install_script_reports_version_when_setup_is_unavailable(tmp_path):
    environment, _log, ursa = _fake_installation(tmp_path)
    ursa.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "--version" ]; then\n'
        '    printf "ursa 2.4.1\\n"\n'
        "    exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    ursa.chmod(0o755)

    result = _run_installer(environment, check=False)

    assert result.returncode != 0
    assert (
        "installed URSA (ursa 2.4.1) does not support guided setup"
        in result.stderr
    )
    assert "https://lanl.github.io/ursa/latest/getting-started" in result.stderr


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


@pytest.mark.skipif(
    os.name != "posix",
    reason="PATH shadowing follows macOS and Linux shell semantics",
)
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


def test_docs_workflow_publishes_versioned_installers():
    workflow = SITE_WORKFLOW.read_text()
    hooks = DOCS_HOOKS.read_text()
    getting_started = GETTING_STARTED.read_text()

    assert "mike deploy" in workflow
    assert '"$version" latest' in workflow
    assert 'mike deploy "${push_args[@]}" main' in workflow
    assert 'for installer in ("install.sh", "install.ps1")' in hooks
    assert "_installer_script" not in hooks
    assert '=== "Linux/MacOS"' in getting_started
    assert '=== "Windows"' in getting_started
    assert (
        'powershell -ExecutionPolicy Bypass -c "irm '
        'https://lanl.github.io/ursa/latest/install.ps1 | iex"'
        in getting_started
    )
    for installer in ("install.sh", "install.ps1"):
        assert f'cp "$GITHUB_WORKSPACE/{installer}"' not in workflow
