"""Ratchet audit: unregistered agent states must match the allowlist.

Exact equality both directions: registering a listed state without
shrinking the allowlist fails, and any new unregistered state fails.
"""

import importlib
import inspect
import pkgutil
import warnings

from ursa.agents.base import BaseAgent


def test_state_registration_ratchet_allowlist():
    allowlist = {"AcquisitionState", "PaperState", "RAGState", "RecallState"}

    import ursa.agents as agents_pkg

    modules = []
    for module_info in pkgutil.iter_modules(agents_pkg.__path__):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                modules.append(
                    importlib.import_module(f"ursa.agents.{module_info.name}")
                )
            except ImportError:
                continue

    registered = set()
    for module in modules:
        for obj in vars(module).values():
            if inspect.isclass(obj) and issubclass(obj, BaseAgent):
                registered.add(getattr(obj, "state_type", dict))

    unregistered = set()
    for module in modules:
        for name, obj in vars(module).items():
            if (
                name.endswith("State")
                and inspect.isclass(obj)
                and obj.__module__ == module.__name__
                and issubclass(obj, dict)
                and obj not in registered
            ):
                unregistered.add(name)

    assert unregistered == allowlist
