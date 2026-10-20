"""Small, reusable probes for configured external services."""

from __future__ import annotations

import asyncio
import inspect
import multiprocessing
import os
import signal
import subprocess
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel

from ursa.util.secrets import SecretReference


@tool
def _ursa_health_echo(value: str) -> str:
    """Return a value unchanged for an URSA health check."""
    return value


class _HealthResponse(BaseModel):
    status: str


PROBE_TIMEOUT_SECONDS = 30.0
_PROBE_FAILURES = (Exception, SystemExit, asyncio.CancelledError)


def _probe_worker(connection: Any, probe: Any, args: tuple[Any, ...]) -> None:
    """Run one probe in an isolated process and return a small outcome."""
    if os.name == "posix":
        with suppress(OSError):
            os.setsid()
    try:
        result = probe(*args)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        connection.send((True, result))
    except (Exception, SystemExit, asyncio.CancelledError) as exc:
        try:
            message = str(exc).strip()
        except BaseException:
            message = ""
        connection.send((False, message or type(exc).__name__))
    finally:
        connection.close()


def _terminate_probe(process: multiprocessing.Process) -> None:
    """Terminate a timed-out probe and any MCP subprocess it launched."""
    if not process.is_alive():
        process.join()
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            # The deadline can expire before the worker reaches setsid().
            process.terminate()
    elif os.name == "nt":
        with suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                timeout=2,
            )
    else:
        process.terminate()
    process.join(timeout=1)
    if process.is_alive():
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                process.kill()
        else:
            process.kill()
        process.join(timeout=1)


def run_probe_with_deadline(
    probe: Any,
    *args: Any,
    timeout: float = PROBE_TIMEOUT_SECONDS,
) -> str:
    """Run a health probe behind a hard, process-terminating deadline."""
    if timeout <= 0:
        raise ValueError("probe timeout must be greater than zero")
    # Setup runs final checks from a worker thread. ``spawn`` avoids the
    # deadlocks possible when forking a multithreaded Textual process.
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_probe_worker, args=(send, probe, args))
    process.daemon = True
    process.start()
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


def probe_embedding_model(config: Any) -> str:
    """Raise unless an embedding configuration returns a non-empty vector."""
    vector = config.init_embedding().embed_query("URSA health check")
    if not isinstance(vector, Sequence) or not vector:
        raise ValueError("embedding invocation returned no vector")
    return f"Embedding invocation returned {len(vector)} dimensions"


async def probe_mcp_server(name: str, server: Any) -> str:
    """Raise unless an MCP server connects and advertises its tools."""
    from ursa.util.mcp import start_mcp_client

    tools = await start_mcp_client({name: server}).get_tools(server_name=name)
    return f"Connected; server advertised {len(tools)} tool(s)"


def _with_keyring_default(config: Any, username: str | None) -> Any:
    """Make a relative keyring reference concrete before worker isolation."""
    reference = getattr(config, "api_key", None)
    if (
        isinstance(reference, SecretReference)
        and reference.keyring is True
        and username is not None
    ):
        return config.model_copy(
            update={
                "api_key": reference.model_copy(update={"keyring": username})
            }
        )
    return config


def check_config_health(
    config: Any, *, timeout: float = PROBE_TIMEOUT_SECONDS
) -> tuple[tuple[str, ...], ...]:
    """Run every external check for a candidate config and collect failures."""
    try:
        config = config.resolve()
    except _PROBE_FAILURES:
        return (("configuration",),)

    probes: list[tuple[tuple[str, ...], Any, tuple[Any, ...]]] = [
        (
            ("inference_providers", name),
            probe_inference_provider,
            (_with_keyring_default(provider, name),),
        )
        for name, provider in config.inference_providers.items()
    ]
    if config.llm_model is not None:
        llm = _with_keyring_default(
            config.llm_model,
            config.llm_model.inference_provider
            or config.llm_model.model_provider,
        )
        probes.extend(
            (
                ("llm_model", capability),
                probe_chat_config,
                (llm, capability),
            )
            for capability in ("chat", "tool", "structured")
        )
    if config.emb_model is not None:
        embedding = _with_keyring_default(
            config.emb_model,
            config.emb_model.inference_provider
            or config.emb_model.model_provider,
        )
        probes.append((("emb_model",), probe_embedding_model, (embedding,)))
    probes.extend(
        (("mcp_servers", name), probe_mcp_server, (name, server))
        for name, server in config.mcp_servers.items()
    )

    failures = []
    for path, probe, args in probes:
        try:
            run_probe_with_deadline(probe, *args, timeout=timeout)
        except _PROBE_FAILURES:
            failures.append(path)
    return tuple(failures)
