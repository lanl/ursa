"""Small, reusable probes for configured external services."""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, suppress
from threading import Lock
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel


@tool
def _ursa_health_echo(value: str) -> str:
    """Return a value unchanged for an URSA health check."""
    return value


class _HealthResponse(BaseModel):
    status: str


PROBE_TIMEOUT_SECONDS = 30.0
_PROBE_START_LOCK = Lock()


def probe_failure_message(exc: BaseException) -> str:
    """Return useful leaf messages from failures raised by async task groups."""
    if isinstance(exc, BaseExceptionGroup):
        messages = [probe_failure_message(child) for child in exc.exceptions]
        unique = list(dict.fromkeys(filter(None, messages)))
        return "; ".join(unique) or type(exc).__name__
    try:
        return str(exc).strip() or type(exc).__name__
    except BaseException:
        return type(exc).__name__


def _probe_worker(connection: Any, probe: Any, args: tuple[Any, ...]) -> None:
    if os.name == "posix":
        with suppress(OSError):
            os.setsid()
    try:
        result = probe(*args)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        connection.send((True, result))
    except (Exception, SystemExit, asyncio.CancelledError) as exc:
        connection.send((False, probe_failure_message(exc)))
    finally:
        connection.close()


def _kill_probe(process: Any, *, force: bool = False) -> None:
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            (process.kill if force else process.terminate)()
    elif os.name == "nt" and not force:
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=2,
            )
    else:
        (process.kill if force else process.terminate)()


def _terminate_probe(process: Any) -> None:
    if not process.is_alive():
        process.join()
        return
    _kill_probe(process)
    process.join(timeout=1)
    if process.is_alive():
        _kill_probe(process, force=True)
        process.join(timeout=1)


def _start_probe(process: Any) -> None:
    with _PROBE_START_LOCK:
        try:
            stderr_fd = sys.stderr.fileno()
        except Exception:
            stderr_fd = -1
        if isinstance(stderr_fd, int) and stderr_fd >= 0:
            process.start()
            return
        with (
            open(os.devnull, "w", encoding="utf-8") as stderr,
            redirect_stderr(stderr),
        ):
            process.start()


def run_probe_with_deadline(
    probe: Any,
    *args: Any,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> str:
    """Run a health probe behind a hard, process-terminating deadline."""
    if timeout <= 0:
        raise ValueError("probe timeout must be greater than zero")
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_probe_worker, args=(send, probe, args))
    process.daemon = True
    _start_probe(process)
    send.close()
    try:
        if not receive.poll(timeout):
            raise TimeoutError(
                f"Health probe timed out after {timeout:g} seconds"
            )
        try:
            ok, payload = receive.recv()
        except EOFError as exc:
            raise RuntimeError(
                "Health probe exited without returning a result"
            ) from exc
        if not ok:
            raise RuntimeError(payload)
        if not isinstance(payload, str):
            raise TypeError(
                f"health probes must return str, got {type(payload).__name__}"
            )
        return payload
    finally:
        receive.close()
        _terminate_probe(process)
        with suppress(ValueError):
            process.close()


def probe_inference_provider(config: Any, require_models: bool = False) -> str:
    """Raise unless an inference provider authenticates and lists models."""
    from ursa.util.inference_providers import list_provider_models

    models = list_provider_models(config)
    if require_models and not models:
        raise ValueError("The provider returned no models.")
    return (
        "Connected; authentication accepted; provider advertised "
        f"{len(models)} model(s)"
    )


def probe_chat_model(model: Any) -> str:
    """Raise unless a chat model accepts a minimal request."""
    if model.invoke("URSA health check: reply briefly.") is None:
        raise ValueError("chat invocation returned no response")
    return "Model returned a response"


def probe_chat_config(config: Any, capability: str) -> str:
    """Initialize a chat config and run one named capability probe."""
    probes = {
        "chat": probe_chat_model,
        "tool": probe_chat_tool_use,
        "structured": probe_chat_structured_output,
    }
    try:
        probe = probes[capability]
    except KeyError:
        raise ValueError(f"Unknown chat capability '{capability}'") from None
    return probe(config.init_chat_model())


def probe_chat_tool_use(model: Any) -> str:
    """Raise unless a chat model honors an explicit tool choice."""
    response = model.bind_tools(
        [_ursa_health_echo], tool_choice=_ursa_health_echo.name
    ).invoke("Call ursa_health_echo with the value 'ursa'.")
    calls = getattr(response, "tool_calls", None) or ()
    if not any(
        call.get("name") == _ursa_health_echo.name
        and call.get("args", {}).get("value") == "ursa"
        for call in calls
        if isinstance(call, Mapping)
    ):
        raise ValueError("model did not return the requested tool call")
    return "Model returned the required tool call"


def probe_chat_structured_output(model: Any) -> str:
    """Raise unless a chat model returns the requested typed response."""
    from ursa.util.structured_output import invoke_structured

    response = invoke_structured(
        model,
        _HealthResponse,
        "Return a structured response whose status is exactly 'ok'.",
        context="URSA health check",
    )
    if not isinstance(response, _HealthResponse) or response.status != "ok":
        raise ValueError("model did not return the expected status='ok'")
    return "Model returned a valid typed response"


def _embedding_error(exc: Exception) -> str | None:
    detail = str(exc).lower()
    if (
        "does not support embedding" in detail
        or "not an embedding model" in detail
        or "not supported for this endpoint" in detail
        or "embedding" in detail
        and ("not supported" in detail or "operationnotsupported" in detail)
    ):
        return "The selected model does not support embeddings."
    if any(marker in detail for marker in ("401", "403", "unauthorized")):
        return (
            "The selected model may not support embeddings, or the API key may "
            "not have access to it."
        )
    return None


def probe_embedding_model(config: Any) -> str:
    """Raise unless an embedding configuration returns a non-empty vector."""
    message = None
    try:
        vector = config.init_embedding().embed_query("URSA health check")
    except Exception as exc:
        message = _embedding_error(exc)
        if message is None:
            raise
    if message is not None:
        raise ValueError(message) from None
    if not isinstance(vector, Sequence) or not vector:
        raise ValueError("embedding invocation returned no vector")
    return f"Embedding invocation returned {len(vector)} dimensions"


async def probe_mcp_server(name: str, server: Any) -> str:
    """Raise unless an MCP server connects and advertises its tools."""
    from ursa.util.mcp import start_mcp_client

    tools = await start_mcp_client({name: server}).get_tools(server_name=name)
    return f"Connected; server advertised {len(tools)} tool(s)"
