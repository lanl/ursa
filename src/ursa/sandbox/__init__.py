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
from abc import ABC, abstractmethod
from pathlib import Path

# class UnixSandboxMixin:
#     def parse_extra_writable_paths(
#         paths: list[Path] | None = None,
#     ) -> list[Path]:
#         if paths is None:
#             return []
#         else:
#             return [
#                 r
#                 for p in extra_writable_paths.strip().split(delim)
#                 if (r := Path(p).expanduser().resolve()).exists()
#             ]


def parse_extra_writable_paths(
    env_var: str = "URSA_EXTRA_WRITABLE_PATHS",
) -> list[Path] | None:
    extra_writable_paths = os.getenv(env_var)
    if extra_writable_paths is None or extra_writable_paths.strip() == "":
        return None
    else:
        delim = ";" if os.name == "nt" else ":"
        paths = [
            r
            for p in extra_writable_paths.strip().split(delim)
            if (r := Path(p).expanduser().resolve()).exists()
        ]
        return paths if paths else None


class SandboxError(Exception):
    """Raised when sandbox run is unsuccessful."""


class BaseSandbox(ABC):
    cwd: Path
    readonly_paths: list[Path]
    extra_writable_paths: list[str]
    extra_args: list[str]
    timeout: int

    @abstractmethod
    def execute(self, cmd: str): ...

    @abstractmethod
    def is_working(self) -> bool: ...

    # @abstractmethod
    # def extract_extra_writable_paths(self)

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


class WindowsSandbox(BaseSandbox): ...


_sandbox_exec_args_template = """
(version 1)
(deny default)
(allow sysctl-read)
(allow process*)
(allow file-read*)
(allow file-write*
  (subpath \"{cwd}\")
  (subpath \"{tmpdir}\"){extra_writable_paths})
{extra_args}
"""

_sandbox_exec_readonly_args = """
(version 1)
(deny default)
(allow process*)
(allow file-read*)
""".strip()


class MacSandbox(BaseSandbox):
    """Sandbox-exec Sandbox via sandbox-exec on MacOS."""

    sbx_path = shutil.which("sandbox-exec")

    def __init__(
        self,
        cwd: Path,
        access_network: bool = True,
        tmpdir: Path | None = None,
        extra_writable_paths: list[Path] | None = None,
        timeout: int = 600000,
    ):
        self.cwd = cwd
        self.shell = os.getenv("SHELL", "/bin/zsh")
        self.tmpdir = tmpdir or Path(tempfile.gettempdir()).resolve()
        self.access_network = access_network
        self.timeout = timeout

        self.extra_args = []
        self.extra_args.append(
            "(allow network*)" if self.access_network else "(deny network*)"
        )

        self.extra_writable_paths = []
        if extra_writable_paths:
            for p in extra_writable_paths:
                if p.exists():
                    self.extra_writable_paths.append(
                        f'\n  (subpath "{p.resolve()}")'
                    )

        self.args = [
            "-p",
            _sandbox_exec_args_template.format(
                cwd=self.workspace,
                tmpdir=self.tmpdir,
                extra_args="\n".join(self.extra_args),
                extra_writable_paths="".join(self.extra_writable_paths),
            ).strip(),
        ]

    def is_working(self) -> bool:
        if self.sbx_path is None:
            return False

        # Test that sandboxing works.
        result = subprocess.run(
            [
                self.sbx_path,
                "-p",
                _sandbox_exec_readonly_args,
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
            and "Operation not permitted" in result.stderr
        )

    def execute(self, cmd: str):
        if self.sbx_path is None:
            raise SandboxError("sandbox-exec not on path.")

        result = subprocess.run(
            [self.sbx_path, *self.args, "--", self.shell, "-c", cmd],
            text=True,
            shell=False,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
            check=False,
        )
        return result


class LinuxBwrapSandbox(BaseSandbox):
    """Bubblewrap Sandbox via bwrap on Linux."""

    bwrap_path = shutil.which("bwrap")

    def is_working(self) -> bool:
        if self.bwrap_path is None:
            return False

        # Test that sandboxing works.
        result = subprocess.run(
            [
                self.bwrap_path,
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

        self.extra_writable_paths = []
        if extra_writable_paths:
            for path in extra_writable_paths:
                if path.exists():
                    p = str(path.resolve())
                    self.extra_writable_paths.extend(["--bind", p, p])

        if not self.access_network:
            self.args.append("--unshare-net")

        self.args.extend(self.extra_writable_paths)

    def execute(self, cmd: str):
        if self.bwrap_path is None:
            raise SandboxError("bwrap not on path.")

        result = subprocess.run(
            [self.bwrap_path, *self.args, self.shell, "-c", cmd],
            text=True,
            shell=False,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
            check=False,
        )
        return result
