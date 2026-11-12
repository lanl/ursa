# TODO
# - [X] run_command tool
# - flags, env vars, config:
#     - [ ] `-s/--sandbox-mode`: workspace-write, full-access, read-only
#     - [ ] `--safety-level`: auto (disable if sandbox is working), high, low


import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

_disable_sbx_suggestion = """
Consider setting the environment variable \
URSA_SBX=0 to disable sandboxing.
""".strip()


def parse_extra_writable_paths(
    env_var: str = "URSA_EXTRA_WRITABLE_PATHS",
) -> list[Path]:
    extra_writable_paths = os.getenv(env_var)
    if extra_writable_paths is None or extra_writable_paths.strip() == "":
        return []
    else:
        delim = ";" if os.name == "nt" else ":"
        return [
            r
            for p in extra_writable_paths.strip().split(delim)
            if (r := Path(p).expanduser().resolve()).exists()
        ]


class SandboxError(Exception):
    """Raised when sandbox run is unsuccessful."""


class BaseSandbox(ABC):
    cwd: Path

    @abstractmethod
    def execute(self, cmd: str): ...

    @abstractmethod
    def is_working(self) -> bool: ...

    @property
    def backend(self) -> str:
        return type(self).__name__

    @property
    def workspace(self) -> str:
        return str(self.cwd.resolve())


class NoSandbox(BaseSandbox):
    def __init__(self, cwd: Path, timeout: int = 600000):
        self.cwd = cwd
        self.timeout = timeout

    def execute(self, cmd: str):
        result = subprocess.run(
            cmd,
            text=True,
            shell=True,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
            check=False,
        )
        return result

    def is_working(self) -> bool:
        return True
