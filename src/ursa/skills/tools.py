import yaml
from langchain_core.tools import tool

from .runtime import current_run


def build_skill_tools(owner):
    @tool
    def load_skill(name: str) -> str:
        """Activate a relevant skill by its catalog name for the current request."""
        run = current_run(owner)
        if run is None:
            return "No active skill request."
        try:
            return run.activate(name)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            return f"Cannot load skill: {exc}"

    @tool
    def read_skill_resource(name: str, path: str) -> str:
        """Read a supporting UTF-8 file relative to an active skill's directory."""
        run = current_run(owner)
        if run is None:
            return "No active skill request."
        try:
            return run.read_resource(name, path)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            return f"Cannot read skill resource: {exc}"

    return [load_skill, read_skill_resource]
