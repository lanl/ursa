import importlib
from typing import Iterable


def has_optional_dep_group(dep: str | list[str]) -> bool:
    """
    Check whether an optional dependency group is installed.

    Attempts to import each module.
    """
    if isinstance(dep, str):
        deps: Iterable[str] = [dep]
    else:
        deps = dep

    for d in deps:
        try:
            importlib.import_module(d)
        except Exception:
            return False
    return True
