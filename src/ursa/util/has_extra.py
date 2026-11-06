"""Check installed extras from distribution metadata without importing them."""

from importlib import metadata
from importlib.util import find_spec

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def has_extra(extra: str | list[str], package: str = "ursa") -> bool:
    """Check declared extra requirements and import specs without importing.

    A list requires every extra; unknown extras raise ValueError. ``ursa``
    aliases ``ursa-ai``. This checks top-level import availability, not package health.
    """
    extras = [extra] if isinstance(extra, str) else extra
    return _has_extras(
        extras,
        "ursa-ai" if package == "ursa" else package,
        {},
        set(),
    )


def _has_extras(
    extras: list[str],
    package: str,
    modules: dict[str, list[str]],
    seen: set[tuple[str, str]],
) -> bool:
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return False
    provided = {
        canonicalize_name(name)
        for name in dist.metadata.get_all("Provides-Extra", [])
    }
    for extra in extras:
        extra = canonicalize_name(extra)
        if extra not in provided:
            raise ValueError(f"Unknown extra {extra!r} for {package}")
        key = (canonicalize_name(package), extra)
        if key in seen:
            continue
        seen.add(key)
        for raw in dist.requires or ():
            req = Requirement(raw)
            if req.marker is None or not req.marker.evaluate({"extra": extra}):
                continue
            if req.marker.evaluate({"extra": ""}):
                continue
            try:
                dependency = metadata.distribution(req.name)
            except metadata.PackageNotFoundError:
                return False
            if not req.specifier.contains(dependency.version, prereleases=True):
                return False
            if not modules:
                modules.update(metadata.packages_distributions())
            names = [
                name.split(".")[0]
                for name, distributions in modules.items()
                if canonicalize_name(req.name)
                in map(canonicalize_name, distributions)
            ]
            try:
                if not names or any(find_spec(name) is None for name in names):
                    return False
            except (ImportError, ValueError):
                return False
            if req.extras and not _has_extras(
                sorted(req.extras), req.name, modules, seen
            ):
                return False
    return True
