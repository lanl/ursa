"""Fresh-process helpers for CLI import deferral regression tests."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

SCENARIOS = ("prompt", "mcp", "exec")
_PROJECT_ROOT = Path(__file__).parents[2]
_PROBE = Path(__file__).with_name("_startup_probe.py")
_MCP_SERVERS = {
    "stdio": {"command": "ursa-startup-test-must-not-connect"},
    "sse": {"transport": "sse", "url": "http://127.0.0.1:9/sse"},
    "http": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:9/mcp",
        "headers": {"Authorization": {"env": "UNSET_STARTUP_TEST_TOKEN"}},
    },
}


def measure(scenario: str) -> tuple[float, list[str]]:
    """Measure startup and record imported modules in an isolated process."""
    with TemporaryDirectory(prefix="ursa-startup-test-") as directory:
        root = Path(directory)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith((
                "URSA_",
                "OTEL_",
                "LANGSMITH_",
                "LANGCHAIN_",
            ))
        }
        config_home = root / "config"
        env.update({
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_CACHE_HOME": str(root / "cache"),
            "URSA_SYSTEM_CONFIG": str(root / "system.yaml"),
            "UV_PROJECT_ENVIRONMENT": sys.prefix,
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        })
        config_home.joinpath("ursa").mkdir(parents=True)
        config_home.joinpath("ursa", "config.yaml").write_text(
            json.dumps({
                "workspace": str(root / "workspace"),
                "mcp_servers": _MCP_SERVERS if scenario == "mcp" else {},
            })
        )
        result = root / "result.json"
        start = perf_counter()
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--offline",
                "--no-sync",
                "--project",
                str(_PROJECT_ROOT),
                "python",
                str(_PROBE),
                scenario,
                str(result),
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        if scenario == "exec":
            assert completed.stdout.strip() == "Hello!"
        ready, modules = json.loads(result.read_text())
        return ready - start, modules
