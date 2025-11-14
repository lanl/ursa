from typing import Iterable
import inspect
import functools


def require_optional(mods: str | Iterable[str], *, extra: str | None = None):
    """
    Ensure all 'mods' are importable. Raise a clear, actionable error otherwise.
    """
    if isinstance(mods, str):
        mods = (mods,)
    missing = []
    for m in mods:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        hint = f" Install with: pip install yourpkg[{extra}]" if extra else ""
        plural = "ies" if len(missing) > 1 else "y"
        raise RuntimeError(
            f"Optional dependenc{plural} {', '.join(repr(m) for m in missing)} "
            f"is required for this API.{hint}"
        )


def needs(mods: str | Iterable[str], *, extra: str | None = None):
    """
    Decorate a function OR a class to require optional dependency/ies.

    - For functions/methods: check before calling.
    - For classes (incl. dataclasses / Pydantic models): check on instantiation.

    To Use:

    ```python
    from typing import TYPE_CHECKING

    # This is only needed to support type checking / LSP functionality
    if TYPE_CHECKING:
        import torch # Optional deps needed

    @needs(["torch"], extra="fm") # Declare optional packages needed and which extras provides them
    class Foo:
        pass
    ```
    """

    def deco(obj):
        # Function / method path
        if callable(obj) and not inspect.isclass(obj):

            @functools.wraps(obj)
            def wrapper(*a, **kw):
                require_optional(mods, extra=extra)
                return obj(*a, **kw)

            return wrapper

        # Class path
        if inspect.isclass(obj):
            orig_init = obj.__init__

            # Preserve signature & behavior; just gate before super-__init__
            @functools.wraps(orig_init)
            def __init__(self, *a, **kw):
                require_optional(mods, extra=extra)
                return orig_init(self, *a, **kw)

            # Create a lightweight subclass with overridden __init__ without
            # copying the entire class dict (which can include unpicklable ABC internals).
            namespace = {
                "__module__": obj.__module__,
                "__init__": __init__,
                "__wrapped__": obj,
            }
            if obj.__doc__:
                namespace["__doc__"] = obj.__doc__
            cls = type(obj.__name__, (obj,), namespace)
            cls.__qualname__ = obj.__qualname__
            return cls

        raise TypeError("@needs can only decorate callables or classes")

    return deco
