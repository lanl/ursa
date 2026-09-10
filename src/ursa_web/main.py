"""Launch the URSA Chainlit application."""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path


def _set_auth_secret() -> None:
    """Provide a stable local secret for Chainlit thread authentication."""
    if os.getenv("CHAINLIT_AUTH_SECRET"):
        return
    secret_path = Path.cwd() / ".ursa-data" / "chainlit-auth-secret"
    try:
        secret = secret_path.read_text().strip()
    except OSError:
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret = secrets.token_urlsafe(32)
        secret_path.write_text(secret + "\n")
    os.environ["CHAINLIT_AUTH_SECRET"] = secret


def app() -> None:
    """Run Chainlit with the URSA application and forwarded CLI arguments."""
    import chainlit.cli as chainlit_cli

    _set_auth_secret()
    app_path = Path(__file__).with_name("app.py").resolve()
    arguments = sys.argv[1:]
    if "--host" not in arguments:
        arguments.extend(["--host", "127.0.0.1"])
    if "--port" not in arguments:
        arguments.extend(["--port", "8000"])
    sys.argv = ["chainlit", "run", str(app_path), *arguments]
    chainlit_cli.init_markdown = lambda _root: None
    chainlit_cli.run_chainlit(str(app_path))


if __name__ == "__main__":
    app()
