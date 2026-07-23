from __future__ import annotations

import argparse
import os
import sys

from ursa.util.http import inject_truststore_into_ssl


def main(argv: list[str] | None = None) -> int:
    inject_truststore_into_ssl()
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
        "--group",
        default=os.environ.get("URSA_DASHBOARD_GROUP", "default"),
        help="Agent group to use from ~/.cache/ursa/<group>",
    )
    ap.add_argument(
        "--config",
        "-c",
        default=os.environ.get("URSA_DASHBOARD_CONFIG"),
        help="YAML/JSON URSA config whose llm_model settings initialize the dashboard LLM endpoint.",
    )
    ap.add_argument(
        "--use-web",
        action="store_true",
        default=str(os.environ.get("URSA_DASHBOARD_USE_WEB", ""))
        .strip()
        .lower()
        in {"1", "true", "yes", "on"},
        help="Enable web-search tools for dashboard-created chat/execution agents",
    )
    args = ap.parse_args(argv)

    os.environ["URSA_DASHBOARD_GROUP"] = str(args.group or "default")
    if args.config:
        config_path = os.path.abspath(os.path.expanduser(str(args.config)))
        if not os.path.isfile(config_path):
            print(  # noqa: T201
                f"Dashboard config file not found: {args.config}",
                file=sys.stderr,
            )
            return 2
        os.environ["URSA_DASHBOARD_CONFIG"] = config_path
    else:
        os.environ.pop("URSA_DASHBOARD_CONFIG", None)
    if args.use_web:
        os.environ["URSA_DASHBOARD_USE_WEB"] = "1"

    try:
        import uvicorn  # type: ignore
    except Exception:
        print(  # noqa: T201
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
