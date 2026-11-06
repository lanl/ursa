import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ursa.sandbox.base import BaseSandbox, SandboxError, _disable_sbx_suggestion


class LinuxBwrapSandbox(BaseSandbox):
    """Bubblewrap Sandbox via bwrap on Linux."""

    sbx_path = shutil.which("bwrap")

    def __init__(
        self,
        cwd: Path,
        access_network: bool = True,
        tmpdir: Path | None = None,
        extra_writable_paths: list[Path] | None = None,
        timeout: int = 600000,
    ):
        self.cwd = cwd
        self.shell = os.getenv("SHELL", "bash")
        self.tmpdir = tmpdir or Path(tempfile.gettempdir()).resolve()
        self.access_network = access_network
        self.timeout = timeout
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
        self.register_extra_writable_paths(extra_writable_paths)

        if not self.access_network:
            self.args.append("--unshare-net")

        self.args.extend(self.extra_writable_paths)

    def register_extra_writable_paths(self, paths: list[Path] | None):
        self.extra_writable_paths = []
        if paths:
            for path in paths:
                if path.exists():
                    p = str(path.resolve())
                    self.extra_writable_paths.extend(["--bind", p, p])

    def is_working(self) -> bool:
        if self.sbx_path is None:
            return False

        # Test that sandboxing works.
        result = subprocess.run(
            [
                self.sbx_path,
                "--ro-bind",
                "/",
                "/",
                "touch",
                "foo",
            ],
            check=False,
            text=True,
            shell=False,
            capture_output=True,
            cwd=self.workspace,
        )

        return (
            result.stderr is not None
            and "Read-only file system" in result.stderr
        )

    def execute(self, cmd: str):
        if self.sbx_path is None:
            raise SandboxError(f"bwrap not on path. {_disable_sbx_suggestion}")

        result = subprocess.run(
            [self.sbx_path, *self.args, self.shell, "-c", cmd],
            text=True,
            shell=False,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
            check=False,
        )
        return result
