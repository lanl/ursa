from email.message import Message
from importlib import import_module, metadata
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

extras = import_module("ursa.util.has_extra")


def distribution(
    *, provides=(), requires=(), modules=(), files=(), version="1.0"
):
    headers = Message()
    for name in provides:
        headers["Provides-Extra"] = name
    return SimpleNamespace(
        metadata=headers,
        requires=requires,
        version=version,
        files=[PurePosixPath(path) for path in files],
        read_text=lambda name: "\n".join(modules)
        if name == "top_level.txt"
        else None,
    )


@pytest.fixture
def catalog(monkeypatch):
    packages = {}

    def get_distribution(name):
        try:
            return packages[name]
        except KeyError:
            raise metadata.PackageNotFoundError(name) from None

    monkeypatch.setattr(extras.metadata, "distribution", get_distribution)
    return packages


def test_default_package_uses_actual_extra_dependencies(catalog, monkeypatch):
    catalog["ursa-ai"] = distribution(
        provides=["lammps"],
        requires=[
            'atomman>=1; extra == "lammps"',
            'trafilatura>=1; extra == "lammps"',
        ],
    )
    for name in ("atomman", "trafilatura"):
        catalog[name] = distribution(modules=[name])
    checked = []
    monkeypatch.setattr(
        extras, "find_spec", lambda name: checked.append(name) or object()
    )
    assert extras.has_extra("lammps")
    assert checked == ["atomman", "trafilatura"]
    del catalog["trafilatura"]
    assert not extras.has_extra("lammps")


def test_distribution_and_import_names_can_differ(catalog, monkeypatch):
    catalog["other-package"] = distribution(
        provides=["office-readers"],
        requires=['python-docx>=1; extra == "office-readers"'],
    )
    catalog["python-docx"] = distribution(modules=["docx"])
    monkeypatch.setattr(
        extras, "find_spec", lambda name: object() if name == "docx" else None
    )
    assert extras.has_extra("office_readers", package="other-package")


def test_markers_ignore_base_and_inactive_dependencies(catalog):
    catalog["ursa-ai"] = distribution(
        provides=["example"],
        requires=[
            "missing-base-dependency>=1",
            'missing-platform-dependency; extra == "example" and python_version < "2"',
        ],
    )
    assert extras.has_extra("example")


def test_missing_distribution_and_unknown_extra(catalog):
    assert not extras.has_extra("missing")
    catalog["ursa-ai"] = distribution(provides=["known"])
    with pytest.raises(ValueError, match="Unknown extra"):
        extras.has_extra("unknown")


@pytest.mark.parametrize(
    "version, discoverable", [("0.9", True), ("1.0", False)]
)
def test_requires_compatible_version_and_import_spec(
    catalog, monkeypatch, version, discoverable
):
    catalog["ursa-ai"] = distribution(
        provides=["feature"], requires=['dependency>=1; extra == "feature"']
    )
    catalog["dependency"] = distribution(
        modules=["dependency"], version=version
    )
    monkeypatch.setattr(
        extras, "find_spec", lambda name: object() if discoverable else None
    )
    assert not extras.has_extra("feature")


def test_list_and_nested_extras_with_cycle(catalog, monkeypatch):
    catalog["ursa-ai"] = distribution(
        provides=["one", "two"],
        modules=["ursa"],
        requires=[
            'dependency[more]>=1; extra == "one"',
            'absent; extra == "two"',
        ],
    )
    catalog["dependency"] = distribution(
        provides=["more"],
        modules=["dependency"],
        requires=['ursa-ai[one]; extra == "more"'],
    )
    monkeypatch.setattr(extras, "find_spec", lambda name: object())
    assert extras.has_extra("one")
    assert not extras.has_extra(["one", "two"])


def test_namespace_discovery_does_not_execute_parent(
    catalog, monkeypatch, tmp_path
):
    root = tmp_path / "demo_namespace"
    root.mkdir()
    (root / "__init__.py").write_text("raise RuntimeError('must not import')")
    child = root / "feature"
    child.mkdir()
    (child / "__init__.py").write_text("raise RuntimeError('must not import')")
    monkeypatch.syspath_prepend(str(tmp_path))
    catalog["ursa-ai"] = distribution(
        provides=["feature"], requires=['namespace-feature; extra == "feature"']
    )
    catalog["namespace-feature"] = distribution(
        modules=["demo_namespace"],
        files=["demo_namespace/feature/__init__.py"],
    )
    assert extras.has_extra("feature")
    (child / "__init__.py").unlink()
    child.rmdir()
    assert not extras.has_extra("feature")


def test_import_names_from_record_and_missing_metadata():
    dist = distribution(
        files=[
            "PIL/__init__.py",
            "PIL/Image.py",
            "PIL/nested/__init__.py",
            "pillow.dist-info/METADATA",
        ]
    )
    assert extras._import_names(dist) == {"PIL"}
    assert extras._import_names(distribution()) == set()


def test_extraskip_checks_custom_package_and_rejects_unknown_extras(catalog):
    from tests.conftest import extraskip

    catalog["example"] = distribution(
        provides=["available", "missing"],
        requires=['missing-dependency; extra == "missing"'],
    )
    extraskip("available", package="example")
    with pytest.raises(
        pytest.skip.Exception, match=r"example\[missing\]"
    ) as exc:
        extraskip("missing", package="example")
    assert exc.value.allow_module_level
    with pytest.raises(ValueError, match="Unknown extra"):
        extraskip("typo", package="example")
