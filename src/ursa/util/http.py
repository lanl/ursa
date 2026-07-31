from __future__ import annotations

import ssl
from typing import Any

import httpx
import truststore

_truststore_injected = False


def truststore_ssl_context() -> ssl.SSLContext:
    """Return an SSL context backed by the system trust store."""
    return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)


def httpx_verify_value(*, verify: bool = True) -> bool | ssl.SSLContext:
    """Return the HTTPX verify value used across direct clients and kwargs."""
    if verify is False:
        return False
    return truststore_ssl_context()


def build_httpx_client(*, verify: bool = True, **kwargs: Any) -> httpx.Client:
    """Build an HTTPX client with truststore-based verification by default."""
    return httpx.Client(verify=httpx_verify_value(verify=verify), **kwargs)


def build_httpx_async_client(
    *, verify: bool = True, **kwargs: Any
) -> httpx.AsyncClient:
    """Build an async HTTPX client with truststore-based verification."""
    return httpx.AsyncClient(verify=httpx_verify_value(verify=verify), **kwargs)


def build_mcp_httpx_async_client(
    *,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Build an async HTTPX client for MCP HTTP transports using truststore."""
    kwargs: dict[str, Any] = {"follow_redirects": True}
    if headers is not None:
        kwargs["headers"] = headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if auth is not None:
        kwargs["auth"] = auth
    return build_httpx_async_client(**kwargs)


def inject_truststore_into_ssl() -> None:
    """Inject truststore into ssl for application entrypoints."""
    global _truststore_injected
    if not _truststore_injected:
        truststore.inject_into_ssl()
        _truststore_injected = True
