# TODO
# - [X] run_command tool
#   - flags, config, env vars:
#       - [ ] `-s/--sandbox-mode`: workspace-write, full-access, read-only
#       - [ ] `--safety-level`: auto (disable if sandbox is working), high, low
# - [ ] write_file tool
# - [ ] read_file tool
# - [ ] write_code tool


import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class BaseSandbox:
    cwd: Path
    readonly_paths: list[Path]
    write_paths: list[Path]
    extra_args: list[str]
    timeout: int = 600000

    def __init__(self, cwd: Path):
        self.cwd = cwd

    def _run(self, cmd: str):
        raise NotImplementedError("Sandbox not implemented")

    @property
    def backend(self) -> str:
        return type(self).__name__

    def _fallback_run(self, cmd: str):
        result = subprocess.run(
            cmd,
            text=True,
            shell=True,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.cwd,
            check=False,
        )
        return result

    def run(self, cmd: str):
        try:
            return self._run(cmd)
        except Exception:
            print(f"Sandbox {self.backend} failed")
            return self._fallback_run(cmd)


class WindowsSandbox(BaseSandbox): ...


class MacSandbox(BaseSandbox): ...


class LinuxBwrapSandbox(BaseSandbox):
    bwrap_path = str(shutil.which("bwrap"))

    def __init__(
        self, cwd: Path, access_network: bool = True, tmpdir: Path | None = None
    ):
        self.cwd = cwd
        self.shell = os.getenv("SHELL", "bash")
        self.tmpdir = tmpdir or Path(tempfile.gettempdir()).resolve()
        self.access_network = access_network
        self.workspace = str(Path.cwd().resolve())
        self.args = [
            "--new-session",
            "--die-with-parent",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-ipc",
            "--cap-drop",
            "ALL",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            self.tmpdir,
            "--bind",
            self.workspace,
            self.workspace,
            "--chdir",
            self.workspace,
        ]
        if not self.access_network:
            self.args += "--unshare-net"

    def _run(self, cmd: str):
        result = subprocess.run(
            [self.bwrap_path, *self.args, self.shell, "-c", cmd],
            text=True,
            shell=False,
            capture_output=True,
            timeout=600000,
            cwd=Path.cwd(),
            check=False,
        )
        return result
