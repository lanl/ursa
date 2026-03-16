from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ursa-dashboard")
    ap.add_argument(
        "--host", default=os.environ.get("URSA_DASHBOARD_HOST", "127.0.0.1")
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("URSA_DASHBOARD_PORT", "8080")),
    )
    ap.add_argument(
        "--reload", action="store_true", help="Enable auto-reload (dev only)"
    )
    ap.add_argument(
        "--workspace",
        default=os.environ.get("URSA_DASHBOARD_WORKSPACE_ROOT"),
        help="Workspace root directory (overrides URSA_DASHBOARD_WORKSPACE_ROOT)",
    )
    args = ap.parse_args(argv)

    if args.workspace:
        os.environ["URSA_DASHBOARD_WORKSPACE_ROOT"] = args.workspace

    try:
        import uvicorn  # type: ignore
    except Exception:
        print(
            "uvicorn is required to run the server. Install with: pip install uvicorn",
            file=sys.stderr,
        )
        return 2

    uvicorn.run(
        "ursa_dashboard.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
