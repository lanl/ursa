from importlib import import_module

import pytest

extras = import_module("ursa.util.has_extra")


@pytest.fixture
def install(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(tmp_path))

    def write(
        name,
        *,
        provides=("feature",),
        requires=(),
        modules=(),
        files=(),
        version="1.0",
    ):
        info = tmp_path / f"{name.replace('-', '_')}-{version}.dist-info"
        info.mkdir()
        headers = [f"Name: {name}", f"Version: {version}"]
        headers += [f"Provides-Extra: {extra}" for extra in provides]
        headers += [f"Requires-Dist: {req}" for req in requires]
        (info / "METADATA").write_text("\n".join(headers) + "\n")
        (info / "top_level.txt").write_text("\n".join(modules))
        (info / "RECORD").write_text("\n".join(f"{path},," for path in files))
        for path in files:
            module = tmp_path / path
            module.parent.mkdir(parents=True, exist_ok=True)
            module.write_text(
                "raise RuntimeError('availability must not import')"
            )
        return info

    return write


@pytest.mark.parametrize(
    "requirement, installed, module, expected",
    [
        ('benchmark-dep>=1; extra == "feature"', True, "json", True),
        ('benchmark-dep>=1; extra == "feature"', False, "json", False),
        (
            'benchmark-dep>=1; extra == "feature"',
            True,
            "absent_benchmark_module",
            False,
        ),
        ('benchmark-dep>=2; extra == "feature"', True, "json", False),
        (
            'benchmark-dep; extra == "feature" and python_version < "2"',
            False,
            "json",
            True,
        ),
        ("benchmark-dep", False, "json", True),
    ],
)
def test_declared_requirements(
    install, requirement, installed, module, expected
):
    install("ursa-ai", requires=[requirement])
    if installed:
        install("benchmark-dep", modules=[module])
    assert extras.has_extra("feature") is expected


def test_all_dependencies_and_nested_extras(install):
    install(
        "ursa-ai",
        provides=["lammps", "missing"],
        modules=["json"],
        requires=[
            'benchmark-atomman[more]; extra == "lammps"',
            'benchmark-trafilatura; extra == "lammps"',
            'absent-benchmark-dep; extra == "missing"',
        ],
    )
    install(
        "benchmark-atomman",
        provides=["more"],
        modules=["json"],
        requires=[
            'ursa-ai[lammps]; extra == "more"',
        ],
    )
    assert not extras.has_extra("lammps")
    install("benchmark-trafilatura", modules=["json"])
    assert extras.has_extra("lammps")
    assert not extras.has_extra(["lammps", "missing"])


def test_namespace_discovery_without_imports(install, tmp_path):
    install(
        "benchmark-parent", requires=['benchmark-child; extra == "feature"']
    )
    install(
        "benchmark-child",
        modules=["benchmark_namespace"],
        files=["benchmark_namespace/feature/__init__.py"],
    )
    (tmp_path / "benchmark_namespace/__init__.py").write_text(
        "raise RuntimeError('availability must not import')"
    )
    assert extras.has_extra("feature", package="benchmark-parent")


def test_custom_package_extra_normalization_and_extraskip(install):
    from tests.conftest import extraskip

    install(
        "benchmark-package",
        provides=["office-readers", "missing"],
        requires=[
            'absent-benchmark-dep; extra == "missing"',
        ],
    )
    assert extras.has_extra("office_readers", package="benchmark-package")
    assert not extras.has_extra("feature", package="absent-benchmark-package")
    extraskip("office-readers", package="benchmark-package")
    with pytest.raises(
        pytest.skip.Exception, match=r"benchmark-package\[missing\]"
    ) as exc:
        extraskip("missing", package="benchmark-package")
    assert exc.value.allow_module_level
    with pytest.raises(ValueError, match="Unknown extra"):
        extraskip("typo", package="benchmark-package")
