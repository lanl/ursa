import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from ursa.sandbox.base import (
    BaseSandbox,
    SandboxError,
    _disable_sbx_suggestion,
    _run_process_group,
)

_sandbox_exec_args_template = """
(version 1)
(deny default)
(allow sysctl-read)
(allow process*)
(allow file-read*)
(allow file-write*
  (subpath \"{cwd}\")
  (literal \"/dev/null\")
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
        extra_writable_paths: list[Path],
        access_network: bool = True,
        tmpdir: Path | None = None,
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
        self.register_extra_writable_paths(extra_writable_paths)

        self.args = [
            "-p",
            _sandbox_exec_args_template.format(
                cwd=self.workspace,
                tmpdir=self.tmpdir,
                extra_args="\n".join(self.extra_args),
                extra_writable_paths="".join(self.extra_writable_paths),
            ).strip(),
        ]

    def register_extra_writable_paths(self, paths: list[Path]):
        self.extra_writable_paths = []
        if paths:
            for p in paths:
                if p.exists():
                    self.extra_writable_paths.append(
                        f'\n  (subpath "{p.expanduser().resolve()}")'
                    )

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
            raise SandboxError(
                f"sandbox-exec not on path. {_disable_sbx_suggestion}"
            )

        return _run_process_group(
            [self.sbx_path, *self.args, "--", self.shell, "-c", cmd],
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=False,
            capture_output=True,
            timeout=self.timeout,
            cwd=self.workspace,
        )
