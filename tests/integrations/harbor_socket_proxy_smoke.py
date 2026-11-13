"""Exercise the socket proxy in a real, unprivileged Apptainer namespace."""

import argparse
import asyncio
import http.client
import json
import os
import shlex
import shutil
import socket
import ssl
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


async def inside(args):
    assert os.getuid() != 0, "Run the smoke test as an ordinary user"
    assert "NoNewPrivs:\t1" in Path("/proc/self/status").read_text()
    for name, identity in (("uid", os.getuid()), ("gid", os.getgid())):
        assert Path(f"/proc/self/{name}_map").read_text().split() == [
            str(identity),
            str(identity),
            "1",
        ]
    assert os.readlink("/proc/self/ns/net") != args.host_namespace
    with socket.socket() as direct:
        direct.settimeout(1)
        assert direct.connect_ex(("127.0.0.1", args.http_port)) != 0
    assert len(Path("/proc/net/route").read_text().splitlines()) == 1
    print(
        f"PASS: UID {os.getuid()}, no_new_privs, single UID/GID mappings, "
        "isolated namespace, host TCP unreachable",
        flush=True,
    )

    assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == ""
    assert os.environ["HTTP_PROXY"] == os.environ["http_proxy"]
    port = urlsplit(os.environ["HTTP_PROXY"]).port
    origin = f"http://127.0.0.1:{args.http_port}"

    def checks():
        def request(method, target, body=None):
            connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=10
            )
            try:
                connection.request(
                    method,
                    target,
                    body,
                    {"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                return response.status, response.read()
            finally:
                connection.close()

        status, body = request(
            "POST",
            origin + "/v1/chat/completions",
            json.dumps({"model": "smoke", "stream": False}),
        )
        assert (
            status == 200
            and json.loads(body)["choices"][0]["message"]["content"]
            == "socket works"
        )
        print("PASS: model base URL, JSON request/response", flush=True)

        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        connection.request(
            "POST",
            origin + "/v1/chat/completions",
            json.dumps({"stream": True}),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        try:
            # The upstream cannot finish until we receive its first event.
            assert response.readline().startswith(b"data: ")
            assert request("GET", origin + "/release-stream")[0] == 200
            assert b"data: [DONE]" in response.read()
        finally:
            connection.close()
        print("PASS: streaming SSE and concurrent requests", flush=True)

        status, body = request(
            "GET", f"http://127.0.0.1:{args.http_port}/allowed"
        )
        assert status == 200 and body == b"allowed"
        with urlopen(origin + "/allowed", timeout=10) as response:
            assert response.read() == b"allowed"
        print(
            "PASS: automatic proxy environment and HTTP forwarding", flush=True
        )

        context = ssl.create_default_context(cafile="/proxy-certs/cert.pem")
        connection = http.client.HTTPSConnection(
            "127.0.0.1", port, context=context, timeout=10
        )
        connection.set_tunnel("127.0.0.1", args.https_port)
        try:
            connection.request("GET", "/allowed")
            response = connection.getresponse()
            assert response.status == 200 and response.read() == b"allowed"
        finally:
            connection.close()
        print(
            "PASS: allowlisted HTTPS CONNECT with certificate verification",
            flush=True,
        )

        assert request("GET", "http://blocked.invalid/")[0] == 403
        assert request("GET", "http://127.0.0.2:1/")[0] == 403
        assert request("CONNECT", "blocked.invalid:443")[0] == 403
        print("PASS: denied HTTP/CONNECT hosts", flush=True)

    await asyncio.to_thread(checks)


async def outside(args):
    from aiohttp import web
    from harbor.models.task.config import (
        EnvironmentConfig,
        NetworkMode,
        NetworkPolicy,
    )
    from harbor.models.trial.paths import TrialPaths
    from harbor.trial.network_policy import (
        add_inference_hosts,
        inference_endpoint_host,
    )

    from ursa.integrations.harbor.singularity import (
        DockerfileSingularityEnvironment,
    )

    class ExistingImageEnvironment(DockerfileSingularityEnvironment):
        async def _build_main_sif(self, force_build):
            return args.image.resolve()

    assert "NoNewPrivs:\t1" in Path("/proc/self/status").read_text(), (
        "Run this smoke test through setpriv --no-new-privs"
    )
    assert os.getuid() != 0, "Run the smoke test without sudo"
    release = asyncio.Event()

    async def handler(request):
        if request.path == "/allowed":
            return web.Response(text="allowed")
        if request.path == "/release-stream":
            release.set()
            return web.Response(text="released")
        payload = await request.json()
        if not payload.get("stream"):
            return web.json_response({
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "socket works",
                        }
                    }
                ]
            })
        response = web.StreamResponse(
            headers={"Content-Type": "text/event-stream"}
        )
        await response.prepare(request)
        await response.write(
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
        )
        await release.wait()
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        with tempfile.TemporaryDirectory(
            prefix="ursa-uds-", dir="/tmp"
        ) as directory:
            runtime = Path(directory)
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-config",
                    "/dev/null",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-keyout",
                    str(runtime / "key.pem"),
                    "-out",
                    str(runtime / "cert.pem"),
                    "-days",
                    "1",
                    "-subj",
                    "/CN=localhost",
                    "-addext",
                    "subjectAltName=IP:127.0.0.1",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            tls.load_cert_chain(runtime / "cert.pem", runtime / "key.pem")
            http = web.TCPSite(runner, "127.0.0.1", 0)
            https = web.TCPSite(runner, "127.0.0.1", 0, ssl_context=tls)
            await http.start()
            await https.start()
            http_port = http._server.sockets[0].getsockname()[1]
            https_port = https._server.sockets[0].getsockname()[1]
            task = runtime / "environment"
            task.mkdir()
            (task / "Dockerfile").write_text(
                "FROM ubuntu:24.04\nWORKDIR /workspace\n"
            )
            trial = TrialPaths(runtime / "trial")
            trial.mkdir()
            offline = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)
            inferred = add_inference_hosts(
                offline,
                [inference_endpoint_host(f"http://127.0.0.1:{http_port}/v1")],
            )
            explicit = NetworkPolicy(
                network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["localhost"]
            )
            environment = ExistingImageEnvironment(
                environment_dir=task,
                environment_name="socket-proxy-smoke",
                session_id=runtime.name,
                trial_paths=trial,
                task_env_config=EnvironmentConfig(
                    workdir="/workspace", storage_mb=64
                ),
                network_policy=offline,
                phase_network_policies=[inferred, explicit],
                mounts=[
                    dict(
                        type="bind",
                        source=str(Path(__file__).resolve()),
                        target="/proxy-smoke.py",
                        read_only=True,
                    ),
                    dict(
                        type="bind",
                        source=str(runtime / "cert.pem"),
                        target="/proxy-certs/cert.pem",
                        read_only=True,
                    ),
                ],
                persistent_env={},
                singularity_fakeroot=False,
                singularity_image_cache_dir=str(runtime / "images"),
            )
            environment._runtime_path = shutil.which(args.runtime)
            assert environment._runtime_path, args.runtime

            async def checked(command):
                result = await environment.exec(command, timeout_sec=30)
                assert result.return_code == 0, (result.stdout, result.stderr)
                if result.stdout:
                    print(result.stdout, end="", flush=True)
                return result

            proxy_directory = None
            try:
                await environment.start(False)
                await checked(
                    'test -z "$HTTP_PROXY" && echo preserved > /workspace/marker'
                )
                assert environment._proxy_process is None
                await environment.set_network_policy(inferred)
                proxy_directory = Path(environment._proxy_temp.name)
                process = environment._proxy_process
                await checked(
                    shlex.join([
                        "python3",
                        "/proxy-smoke.py",
                        "--inside",
                        "--http-port",
                        str(http_port),
                        "--https-port",
                        str(https_port),
                        "--host-namespace",
                        os.readlink("/proc/self/ns/net"),
                    ])
                )
                await environment.set_network_policy(offline)
                assert process.returncode is not None
                assert not (proxy_directory / "host.sock").exists()
                await checked(
                    'test -z "$HTTP_PROXY" && test "$(cat /workspace/marker)" = preserved'
                )
                print(
                    "PASS: inferred model access revoked for offline phase; overlay preserved",
                    flush=True,
                )
                await environment.set_network_policy(explicit)
                await checked(f"curl -fsS http://localhost:{http_port}/allowed")
                result = await checked(
                    f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{http_port}/allowed"
                )
                assert result.stdout == "403"
                print(
                    "\nPASS: explicit allowlist replaces inferred model hosts",
                    flush=True,
                )
            finally:
                release.set()
                await environment.stop(True)
            assert proxy_directory is not None and not proxy_directory.exists()
            print("PASS: container and proxy cleanup", flush=True)
    finally:
        release.set()
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--runtime", default="apptainer")
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--http-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--https-port", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--host-namespace", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.inside and args.image is None:
        parser.error("--image is required")
    asyncio.run(inside(args) if args.inside else outside(args))


if __name__ == "__main__":
    main()
