"""Measure the real exec command against a local, deterministic model endpoint."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ._startup import isolated_env


def measure_exec(*, profile_imports: bool = False) -> dict:
    """Measure uv through the visible response, and separately through exit."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, payload):
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            requests.append((self.path, time.perf_counter()))
            self.reply({
                "object": "list",
                "data": [
                    {
                        "id": "startup-benchmark",
                        "object": "model",
                        "created": 0,
                        "owned_by": "benchmark",
                        "max_input_tokens": 32000,
                    }
                ],
            })

        def do_POST(self):
            requests.append((self.path, time.perf_counter()))
            body = json.loads(
                self.rfile.read(int(self.headers["Content-Length"]))
            )
            assert self.path == "/v1/chat/completions", self.path
            assert body["messages"][-1]["content"] == "say hello"
            assert body["tools"], "The real chat agent must retain its tools"
            self.reply({
                "id": "benchmark-response",
                "object": "chat.completion",
                "created": 0,
                "model": "startup-benchmark",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            })

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="ursa-exec-") as directory:
            root = Path(directory)
            config = root / "config.json"
            config.write_text(
                json.dumps({
                    "llm_model": {
                        "model": "startup-benchmark",
                        "model_provider": "openai",
                        "inference_provider": "benchmark",
                        "use_responses_api": False,
                    },
                    "inference_providers": {
                        "benchmark": {
                            "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                            "api_key": {
                                "env": "URSA_BENCHMARK_TOKEN",
                                "template": "%s",
                            },
                        }
                    },
                    "workspace": str(root / "workspace"),
                })
            )
            env = isolated_env(root)
            env["URSA_BENCHMARK_TOKEN"] = "benchmark-token"
            env["UV_PROJECT_ENVIRONMENT"] = sys.prefix
            env["UV_NO_PROGRESS"] = "1"
            env["NO_PROXY"] = "127.0.0.1,localhost"
            env.pop("PYTHONPROFILEIMPORTTIME", None)
            if profile_imports:
                env["PYTHONPROFILEIMPORTTIME"] = "1"
            uv = shutil.which("uv")
            if uv is None:
                raise RuntimeError(
                    "uv is required to benchmark the exec command"
                )
            command = [
                uv,
                "run",
                "--offline",
                "--project",
                str(Path(__file__).resolve().parents[2]),
                "ursa",
                "--config",
                str(config),
                "exec",
                "say hello",
            ]
            output = []
            response_times = []

            def read_response(stream):
                for line in stream:
                    output.append(line)
                    if line.strip() == "Hello!":
                        response_times.append(time.perf_counter())

            with tempfile.TemporaryFile(mode="w+t") as errors:
                start = time.perf_counter()
                with subprocess.Popen(
                    command,
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    start_new_session=os.name == "posix",
                ) as process:
                    reader = threading.Thread(
                        target=read_response, args=(process.stdout,)
                    )
                    reader.start()
                    try:
                        process.wait(timeout=30)
                    except BaseException:
                        if os.name == "posix":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                        raise
                    finally:
                        reader.join(timeout=2)
                    end = time.perf_counter()
                errors.seek(0)
                stderr = errors.read()
            if process.returncode or "".join(output).strip() != "Hello!":
                raise RuntimeError(f"exec failed:\n{''.join(output)}\n{stderr}")
            assert len(response_times) == 1
            sent = [
                stamp
                for path, stamp in requests
                if path == "/v1/chat/completions"
            ]
            assert len(sent) == 1
            return {
                "response_seconds": response_times[0] - start,
                "exit_seconds": end - start,
                "request_seconds": sent[0] - start,
                "requests": [path for path, _ in requests],
                "stderr": stderr,
            }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
