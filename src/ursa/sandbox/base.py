# TODO
# - [X] run_command tool
# - flags, env vars, config:
#     - [ ] `-s/--sandbox-mode`: workspace-write, full-access, read-only
#     - [ ] `--safety-level`: auto (disable if sandbox is working), high, low


import os
import signal
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


def _run_process_group(cmd: str, **kwargs):
    """Run a shell command, killing its POSIX process group on interruption."""
    if os.name == "nt":
        return subprocess.run(cmd, check=False, **kwargs)

    timeout = kwargs.pop("timeout")
    kwargs.pop("capture_output", None)
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        **kwargs,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Restricted hosts may forbid group signaling; still terminate the
            # direct process rather than leaving everything running.
            process.kill()
        process.wait()
        raise
    return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)


class NoSandbox(BaseSandbox):
    def __init__(self, cwd: Path, timeout: int = 600000):
        self.cwd = cwd
        self.timeout = timeout

    def execute(self, cmd: str):
        # subprocess.run kills only its direct shell on timeout, leaving the
        # command's children alive. Give the shell its own process group and
        # kill the whole group before returning control to the agent.
        return _run_process_group(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
        )

    def is_working(self) -> bool:
        return True
