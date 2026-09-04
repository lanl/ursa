"""Launch a stdio MCP server with its stderr redirected to a file."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m ursa.util.mcp_stdio_proxy FILE COMMAND [ARG ...]"
        )
    path, command, *args = sys.argv[1:]
    stderr_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.dup2(stderr_fd, 2)
    finally:
        os.close(stderr_fd)
    if sys.platform == "win32":
        executable = shutil.which(command) or command
        result = subprocess.run([executable, *args], check=False)
        raise SystemExit(result.returncode)
    os.execvp(command, [command, *args])


if __name__ == "__main__":
    main()
