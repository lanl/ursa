"""Check installed extras from distribution metadata without importing them."""

from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
from importlib.util import find_spec

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def _import_names(dist: metadata.Distribution) -> set[str]:
    """Find import names, including packages inside shared namespaces."""
    declared = set((dist.read_text("top_level.txt") or "").split())
    names = set()
    for path in dist.files or ():
        parts = path.parts
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        elif len(parts) == 1 and path.suffix == ".py":
            parts = (path.stem,)
        elif len(parts) == 1:
            suffix = next(
                (
                    suffix
                    for suffix in EXTENSION_SUFFIXES
                    if path.name.endswith(suffix)
                ),
                None,
            )
            if suffix is None:
                continue
            parts = (path.name.removesuffix(suffix),)
        else:
            continue
        if parts and all(part.isidentifier() for part in parts):
            if not declared or parts[0] in declared:
                names.add(".".join(parts))
    if not names:
        names = declared
    # Check the shallowest package owned by this distribution, not merely a
    # shared namespace that could be supplied by an unrelated distribution.
    roots = set()
    for name in sorted(names, key=lambda name: (name.count("."), name)):
        if not any(name.startswith(parent + ".") for parent in roots):
            roots.add(name)
    return roots


def _has_module(name: str) -> bool:
    parts = name.split(".")
    try:
        spec = find_spec(parts[0])
        for index in range(1, len(parts)):
            if spec is None or spec.submodule_search_locations is None:
                return False
            # util.find_spec on a dotted name imports its parent. Resolve each
            # child against the parent's search path without executing it.
            spec = PathFinder.find_spec(
                ".".join(parts[: index + 1]), spec.submodule_search_locations
            )
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False


def has_extra(extra: str | list[str], package: str = "ursa") -> bool:
    """Return whether the declared extra's dependencies are discoverable.

    ``package`` is a distribution name; ``ursa`` aliases ``ursa-ai``. Check
    version constraints, environment markers, and dependency extras as well as
    import specs. A list requires every extra. Unknown extras raise ValueError;
    missing distributions or modules return False. Base requirements are assumed
    installed. This is an availability check, not an import/health check.
    """
    extras = [extra] if isinstance(extra, str) else extra
    return _has_extras(
        extras, "ursa-ai" if package == "ursa" else package, set()
    )


def _has_extras(
    extras: list[str], package: str, seen: set[tuple[str, str]]
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
            names = _import_names(dependency)
            if not names or not all(_has_module(name) for name in names):
                return False
            if req.extras and not _has_extras(
                sorted(req.extras), req.name, seen
            ):
                return False
    return True
