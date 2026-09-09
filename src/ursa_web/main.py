"""Launch the URSA Chainlit application."""

from __future__ import annotations

import sys
from pathlib import Path


def app() -> None:
    """Run Chainlit with the URSA application and forwarded CLI arguments."""
    import chainlit.cli as chainlit_cli

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
