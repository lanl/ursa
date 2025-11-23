# scripts/mcp/localfs_mcp_server.py
import argparse
import base64
from pathlib import Path

from mcp.server.fastmcp import FastMCP

APP_NAME = "localfs"


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _ensure_path(root: str, key: str) -> Path:
    p = Path(root).expanduser().resolve() / key
    if not p.exists():
        raise FileNotFoundError(f"{key} not found under root {root}")
    return p


def make_app(root: str) -> FastMCP:
    app = FastMCP(APP_NAME)

    @app.tool()
    def fs_list_objects(prefix: str = "", max_keys: int = 1000) -> dict:
        """List files under the configured root (flat, not recursive)."""
        base = Path(root)
        out = []
        count = 0
        for p in sorted(base.glob(f"{prefix}*")):
            if p.is_file():
                out.append({
                    "key": p.relative_to(base).as_posix(),
                    "size": p.stat().st_size,
                })
                count += 1
                if count >= max_keys:
                    break
        return {"root": str(base), "objects": out}

    @app.tool()
    def fs_head_object(key: str) -> dict:
        """Return size/metadata for a file."""
        p = _ensure_path(root, key)
        return {
            "root": root,
            "key": key,
            "size": p.stat().st_size,
            "metadata": {},
        }

    @app.tool()
    def fs_get_object_bytes(key: str, allow_large: bool = False) -> dict:
        """Return whole file as base64."""
        p = _ensure_path(root, key)
        b = p.read_bytes()
        return {
            "root": root,
            "key": key,
            "size": len(b),
            "bytes_b64": _b64(b),
        }

    @app.tool()
    def fs_read_range_bytes(key: str, start: int, length: int) -> dict:
        """Return a byte range as base64 (clamped to file size)."""
        p = _ensure_path(root, key)
        b = p.read_bytes()
        end = min(len(b), max(0, start) + max(0, length))
        chunk = b[max(0, start) : end]
        return {
            "root": root,
            "key": key,
            "start": start,
            "size": len(chunk),
            "bytes_b64": _b64(chunk),
        }

    return app


def main():
    parser = argparse.ArgumentParser(
        description="LocalFS MCP server (STDIO by default)."
    )
    parser.add_argument(
        "--root", required=True, help="Directory to serve files from"
    )
    parser.add_argument(
        "--stdio", action="store_true", help="Serve over STDIO (default)"
    )
    parser.add_argument(
        "--sse", action="store_true", help="Serve over SSE (HTTP)"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host for SSE")
    parser.add_argument("--port", type=int, default=3333, help="Port for SSE")
    args = parser.parse_args()

    print(
        f"[{APP_NAME}] Serving root: {Path(args.root).expanduser().resolve()}"
    )

    app = make_app(args.root)

    # Default to stdio unless --sse is explicitly requested
    if args.sse and not args.stdio:
        # Try “modern” SSE helpers if they exist; otherwise fall back to uvicorn.
        try:
            # Some mcp releases export these:
            from mcp.server.sse import create_sse_app, serve  # type: ignore

            sse_app = create_sse_app(app)
            serve(sse_app, host=args.host, port=args.port)
            return
        except Exception:
            try:
                import uvicorn  # type: ignore

                # Many builds expose an ASGI app via FastMCP for SSE use.
                # Prefer an explicit factory if present; else common attribute names.
                sse_app = None
                # 1) Newer helper on FastMCP
                if hasattr(app, "create_asgi_app"):
                    sse_app = app.create_asgi_app()  # type: ignore[attr-defined]
                # 2) Older attribute commonly used
                elif hasattr(app, "asgi"):
                    sse_app = getattr(app, "asgi")
                if sse_app is None:
                    raise RuntimeError(
                        "Unable to locate ASGI app on FastMCP; upgrade 'mcp'"
                    )

                uvicorn.run(sse_app, host=args.host, port=args.port)
                return
            except ModuleNotFoundError as e:
                raise RuntimeError(
                    "SSE mode requested but helpers are not available.\n"
                    "Either install uvicorn (`pip install uvicorn`) or upgrade `mcp`."
                ) from e

    # STDIO mode (works on all known mcp versions)
    # Some releases expose run(), others run_stdio(); support both.
    if hasattr(app, "run_stdio"):
        app.run_stdio()  # type: ignore[attr-defined]
    else:
        app.run()  # falls back to stdio in older/newer versions


if __name__ == "__main__":
    main()
