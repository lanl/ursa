"""Process creation through the first prompt or flushed exec response."""

import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

SCENARIOS = ("prompt", "mcp", "exec")
MCP_SERVERS = {
    "stdio": {"command": "ursa-benchmark-must-not-connect"},
    "sse": {"transport": "sse", "url": "http://127.0.0.1:9/sse"},
    "http": {
        "transport": "streamable-http",
        "url": "http://127.0.0.1:9/mcp",
        "headers": {"Authorization": {"env": "UNSET_BENCHMARK_TOKEN"}},
    },
}


def measure(scenario):
    with TemporaryDirectory(prefix="ursa-benchmark-") as directory:
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
        for key in (
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "PROGRAMDATA",
            "XDG_CONFIG_HOME",
            "XDG_CONFIG_DIRS",
            "XDG_CACHE_HOME",
        ):
            env[key] = directory
        env["URSA_SYSTEM_CONFIG"] = str(root / "system.yaml")
        env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
        env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
        (root / "ursa").mkdir()
        (root / "ursa/config.yaml").write_text(
            json.dumps({
                "workspace": str(root / "workspace"),
                "mcp_servers": MCP_SERVERS if scenario == "mcp" else {},
            })
        )
        result = root / "result.json"
        start = perf_counter()
        completed = subprocess.run(
            [
                "uv",
                "run",
                "--offline",
                "--project",
                str(Path(__file__).parent.parent),
                "python",
                str(Path(__file__).with_name("_cli.py")),
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


def track_latency(scenario):
    return max(measure(scenario)[0] for _ in range(5))


track_latency.params = SCENARIOS
track_latency.param_names = ["scenario"]
track_latency.unit = "seconds"
track_latency.timeout = 180
