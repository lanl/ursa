# TODO
# - [ ] Sandbox all tools, starting with run_command

from pathlib import Path


class Sandbox:
    readonly_paths: list[Path]
    write_paths: list[Path]
    extra_args: list[str]

    def _fallback_run(self, *args, **kwargs): ...
    def run(self, *args, **kwargs): ...


class BwrapSandbox(Sandbox): ...
