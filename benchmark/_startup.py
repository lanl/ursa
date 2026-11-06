"""Fresh-process readiness probe used by ASV and regression tests."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

MCP_SERVERS = {
    "stdio": {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-c", "raise RuntimeError('MCP started before a prompt')"],
    },
    "sse": {
        "transport": "sse",
        "url": "http://127.0.0.1:9/sse",
        "timeout": 5,
    },
    "http": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:9/mcp",
        "timeout": 30,
        "headers": {
            "Authorization": {
                "env": "URSA_BENCHMARK_MCP_TOKEN",
                "template": "Bearer %s",
            },
        },
    },
}
MCP_SCENARIOS = (
    "mcp-stdio-prompt",
    "mcp-sse-prompt",
    "mcp-http-prompt",
    "mcp-mixed-prompt",
)
SCENARIOS = (
    "prompt",
    "named-prompt",
    "provider-prompt",
    *MCP_SCENARIOS,
    "help",
    "version",
)


def probe(scenario: str, result_path: Path, *, headless: bool) -> None:
    """Exercise the real entrypoint, reporting readiness after its first frame."""
    from ursa.cli import main

    if scenario in {"help", "version"}:
        main([f"--{scenario}"])
        return

    from ursa.cli.tui.app import UrsaTextualApp
    from ursa.cli.tui.widgets import PromptArea

    original_mount = UrsaTextualApp.on_mount
    original_run = UrsaTextualApp.run

    def ready(app):
        prompt = app.query_one(PromptArea)
        assert app.focused is prompt and not prompt.disabled
        ready_ns = time.perf_counter_ns()
        if scenario in MCP_SCENARIOS:
            expected = (
                set(MCP_SERVERS)
                if scenario == "mcp-mixed-prompt"
                else {scenario.split("-")[1]}
            )
            assert set(app.hitl.config.mcp_servers) == expected
        prompt.insert("startup probe")
        assert prompt.text == "startup probe"
        result_path.write_text(
            json.dumps({
                "ready_ns": ready_ns,
                "modules": sorted(sys.modules),
            })
        )
        app.exit()

    def mounted(app):
        original_mount(app)
        app.call_after_refresh(ready, app)

    UrsaTextualApp.on_mount = mounted
    UrsaTextualApp.run = lambda app: original_run(
        app, headless=headless, size=(100, 35)
    )
    args = ["--workspace", str(result_path.parent / "workspace")]
    if scenario == "named-prompt":
        args += ["--name", "startup-benchmark"]
    elif scenario == "provider-prompt":
        args += ["--llm_model.model", "ollama:startup-benchmark"]
    elif scenario in MCP_SCENARIOS:
        args += ["--config", str(result_path.parent / "mcp.json")]
    main(args)
    if not result_path.exists():
        raise RuntimeError("The application exited before the prompt was ready")


def _drain_terminal(fd: int) -> None:
    try:
        while os.read(fd, 65536):
            pass
    except OSError as exc:
        if exc.errno != errno.EIO:  # Linux PTYs signal EOF with EIO.
            raise


def measure(
    scenario: str, timeout: float, *, headless: bool = False
) -> tuple[float, list[str]]:
    with tempfile.TemporaryDirectory(prefix="ursa-startup-") as temporary:
        root = Path(temporary)
        # Do not read user credentials, workspace files, or system configuration.
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("URSA_")
            and key not in {"OPENAI_API_KEY", "PYTHONPROFILEIMPORTTIME"}
        }
        for key in (
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "PROGRAMDATA",
            "XDG_CONFIG_HOME",
            "XDG_CONFIG_DIRS",
            "XDG_CACHE_HOME",
            "URSA_CACHE_HOME",
        ):
            env[key] = temporary
        env["URSA_SYSTEM_CONFIG"] = str(root / "system.yaml")
        env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
        env["TERM"] = "xterm-256color"
        if scenario in MCP_SCENARIOS:
            servers = (
                MCP_SERVERS
                if scenario == "mcp-mixed-prompt"
                else {
                    scenario.split("-")[1]: MCP_SERVERS[scenario.split("-")[1]]
                }
            )
            (root / "mcp.json").write_text(json.dumps({"mcp_servers": servers}))
            env["URSA_BENCHMARK_MCP_TOKEN"] = "startup-benchmark-token"
        result = root / "result.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--probe",
            scenario,
            "--result",
            str(result),
        ]
        terminal = scenario not in {"help", "version"} and not headless
        if headless:
            command.append("--headless")
        master = slave = None
        reader = None
        try:
            if terminal:
                import pty
                import termios

                master, slave = pty.openpty()
                termios.tcsetwinsize(slave, (35, 100))
                reader = threading.Thread(
                    target=_drain_terminal, args=(master,), daemon=True
                )
                reader.start()
            start = time.perf_counter_ns()
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                stdin=slave if terminal else subprocess.DEVNULL,
                stdout=slave if terminal else subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            end = time.perf_counter_ns()
        finally:
            if slave is not None:
                os.close(slave)
            if reader is not None:
                reader.join(timeout=1)
            if master is not None:
                os.close(master)
        if completed.returncode:
            raise RuntimeError(
                f"{scenario} failed ({completed.returncode}):\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        if scenario in {"help", "version"}:
            if not completed.stdout.strip():
                raise RuntimeError(f"{scenario} produced no output")
            return (end - start) / 1e9, []
        data = json.loads(result.read_text())
        return (data["ready_ns"] - start) / 1e9, data["modules"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, choices=SCENARIOS)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    probe(args.probe, args.result, headless=args.headless)
