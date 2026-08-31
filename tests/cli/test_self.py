from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ursa.cli import build_parser, main
from ursa.cli.self import (
    TOOL_PACKAGE,
    _select_extras,
    _select_with_packages,
    build_install_argv,
    build_upgrade_argv,
    build_upgrade_requirement,
    get_package_repository,
    get_uv_tool_dir,
    show_status,
    verify_uv_tool_install,
)


def _receipt(path: Path, requirement: str) -> Path:
    path.write_text(f"[tool]\nrequirements = [{requirement}]\n")
    return path


def test_get_uv_tool_dir_strips_output(monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.self.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=0, stdout="/tools\n")),
    )
    assert get_uv_tool_dir(Path("/bin/uv")) == Path("/tools")


def test_get_uv_tool_dir_reports_stderr(monkeypatch):
    monkeypatch.setattr(
        "ursa.cli.self.subprocess.run",
        MagicMock(return_value=MagicMock(returncode=1, stderr="bad uv\n")),
    )
    with pytest.raises(SystemExit, match="bad uv"):
        get_uv_tool_dir(Path("/bin/uv"))


def test_verify_install_accepts_samefile_and_receipt(tmp_path):
    environment = tmp_path / TOOL_PACKAGE
    environment.mkdir()
    receipt = _receipt(environment / "uv-receipt.toml", '{ name = "ursa-ai" }')
    assert verify_uv_tool_install(tmp_path, environment) == receipt


def test_verify_install_rejects_missing_receipt(tmp_path):
    environment = tmp_path / TOOL_PACKAGE
    environment.mkdir()
    with pytest.raises(SystemExit, match="not installed"):
        verify_uv_tool_install(tmp_path, environment)


def test_fixed_upgrade_argv():
    assert build_upgrade_argv(Path("/bin/uv")) == [
        "/bin/uv",
        "tool",
        "upgrade",
        "--reinstall",
        "--compile-bytecode",
        "ursa-ai",
    ]


def test_install_argv_with_additional_packages():
    assert build_install_argv(
        Path("/bin/uv"), "ursa-ai[dashboard]>=1", ["pytest>=8", "tox"]
    ) == [
        "/bin/uv",
        "tool",
        "install",
        "--force",
        "--reinstall",
        "--compile-bytecode",
        "--with",
        "pytest>=8",
        "--with",
        "tox",
        "ursa-ai[dashboard]>=1",
    ]


def test_with_packages_are_added(tmp_path):
    receipt = tmp_path / "receipt.toml"
    receipt.write_text(
        "[tool]\n"
        'requirements = [{ name = "ursa-ai" }, '
        '{ name = "pytest", specifier = ">=8" }, { name = "tox" }]\n'
    )

    assert _select_with_packages(receipt, ["ruff"], clean=False) == [
        "pytest>=8",
        "tox",
        "ruff",
    ]
    assert _select_with_packages(receipt, ["ruff"], clean=True) == ["ruff"]


def test_extras_are_added():
    assert _select_extras(["dashboard", "fm"], ["image"], clean=False) == [
        "dashboard",
        "fm",
        "image",
    ]
    assert _select_extras(["dashboard"], ["image"], clean=True) == ["image"]


def test_upgrade_requirement_uses_selected_extras(tmp_path):
    receipt = _receipt(
        tmp_path / "receipt.toml",
        '{ name = "ursa-ai", extras = ["dashboard", "fm"], specifier = ">=1" }',
    )
    assert build_upgrade_requirement(receipt, extras=["office_readers"]) == (
        "ursa-ai[office_readers]>=1"
    )
    assert build_upgrade_requirement(receipt, extras=["image"]) == (
        "ursa-ai[image]>=1"
    )


def test_upgrade_requirement_preserves_file_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    receipt = _receipt(
        tmp_path / "receipt.toml",
        f'{{ name = "ursa-ai", directory = "{source}" }}',
    )

    assert build_upgrade_requirement(receipt, extras=["dashboard"]) == (
        f"ursa-ai[dashboard] @ {source.as_uri()}"
    )


def test_version_and_ref_requirements(monkeypatch, tmp_path):
    registry = _receipt(
        tmp_path / "registry.toml",
        '{ name = "ursa-ai", extras = ["dashboard"] }',
    )
    assert build_upgrade_requirement(registry, version="1.2.3") == (
        "ursa-ai[dashboard]==1.2.3"
    )
    monkeypatch.setattr(
        "ursa.cli.self.get_package_repository",
        lambda: "https://example.test/default.git",
    )
    assert build_upgrade_requirement(registry, ref="main") == (
        "ursa-ai[dashboard] @ git+https://example.test/default.git@main"
    )

    git = _receipt(
        tmp_path / "git.toml",
        '{ name = "ursa-ai", git = "https://example.test/ursa.git" }',
    )
    assert build_upgrade_requirement(git, ref="release") == (
        "ursa-ai @ git+https://example.test/ursa.git@release"
    )


def test_package_repository_prefers_metadata(monkeypatch):
    package_metadata = MagicMock()
    package_metadata.get_all.return_value = [
        "Homepage, https://example.test/home",
        "Repository, https://example.test/repository.git",
    ]
    monkeypatch.setattr(
        "ursa.cli.self.metadata", lambda _name: package_metadata
    )

    assert get_package_repository() == "https://example.test/repository.git"


def test_package_repository_has_lanl_fallback(monkeypatch):
    package_metadata = MagicMock()
    package_metadata.get_all.return_value = []
    monkeypatch.setattr(
        "ursa.cli.self.metadata", lambda _name: package_metadata
    )

    assert get_package_repository() == "https://github.com/lanl/ursa"


def test_parser_supports_recipe_flags():
    cfg = build_parser().parse_args([
        "self",
        "modify",
        "--extra",
        "dashboard",
        "--extra",
        "fm",
        "--with",
        "pytest",
        "--with",
        "tox",
        "--ref",
        "main",
    ])
    assert cfg.self.modify.extra == ["dashboard", "fm"]
    assert cfg.self.modify.ref == "main"
    assert cfg.self.modify.with_packages == ["pytest", "tox"]
    assert callable(cfg.self.modify.action)


def test_parser_accepts_clean():
    cfg = build_parser().parse_args(["self", "modify", "--clean"])

    assert cfg.self.modify.clean is True
    assert cfg.self.modify.with_packages is None


def test_parser_ignores_self_management_environment_flags(monkeypatch):
    monkeypatch.setenv("URSA_SELF__MODIFY__EXTRA", "dashboard")
    monkeypatch.setenv("URSA_SELF__MODIFY__VERSION", "1.2.3")
    monkeypatch.setenv("URSA_SELF__MODIFY__REF", "main")
    monkeypatch.setenv("URSA_SELF__MODIFY__WITH_PACKAGES", "pytest")

    cfg = build_parser().parse_args(["self", "modify"])

    assert cfg.self.modify.extra is None
    assert cfg.self.modify.version is None
    assert cfg.self.modify.ref is None
    assert cfg.self.modify.with_packages is None
    assert cfg.self.modify.clean is False


def test_main_routes_upgrade_before_config(monkeypatch):
    called = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    parser = build_parser()
    monkeypatch.setattr("ursa.cli.build_parser", lambda: parser)
    monkeypatch.setattr("ursa.cli.self.upgrade", called)
    with pytest.raises(SystemExit, match="0"):
        main(["self", "modify", "--extra", "dashboard", "--version", "1.2.3"])
    called.assert_called_once_with(
        extras=["dashboard"],
        with_packages=None,
        version="1.2.3",
        ref=None,
        clean=False,
    )


def test_main_self_update_preserves_recipe(monkeypatch):
    called = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    parser = build_parser()
    monkeypatch.setattr("ursa.cli.build_parser", lambda: parser)
    monkeypatch.setattr("ursa.cli.self.upgrade", called)

    with pytest.raises(SystemExit, match="0"):
        main(["self", "update"])

    called.assert_called_once_with()


def test_main_clean_clears_extras_and_additional_packages(monkeypatch):
    called = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    parser = build_parser()
    monkeypatch.setattr("ursa.cli.build_parser", lambda: parser)
    monkeypatch.setattr("ursa.cli.self.upgrade", called)

    with pytest.raises(SystemExit, match="0"):
        main(["self", "modify", "--clean"])

    called.assert_called_once_with(
        extras=None,
        with_packages=None,
        version=None,
        ref=None,
        clean=True,
    )


def test_main_clean_applies_requested_extras_and_packages(monkeypatch):
    called = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    parser = build_parser()
    monkeypatch.setattr("ursa.cli.build_parser", lambda: parser)
    monkeypatch.setattr("ursa.cli.self.upgrade", called)

    with pytest.raises(SystemExit, match="0"):
        main([
            "self",
            "modify",
            "--clean",
            "--extra",
            "fm",
            "--with",
            "pytest",
        ])

    called.assert_called_once_with(
        extras=["fm"],
        with_packages=["pytest"],
        version=None,
        ref=None,
        clean=True,
    )


def test_status_reports_registry_recipe(monkeypatch, tmp_path, capsys):
    environment = tmp_path / TOOL_PACKAGE
    environment.mkdir()
    _receipt(
        environment / "uv-receipt.toml",
        '{ name = "ursa-ai", extras = ["dashboard"], specifier = ">=1" }',
    )
    monkeypatch.setattr("ursa.cli.self.find_uv", lambda: Path("/bin/uv"))
    monkeypatch.setattr("ursa.cli.self.get_uv_tool_dir", lambda _uv: tmp_path)
    monkeypatch.setattr("ursa.cli.self.sys.prefix", str(environment))
    monkeypatch.setattr("ursa.cli.self.package_version", lambda _name: "1.2.3")

    show_status()

    output = capsys.readouterr().out.splitlines()
    assert output[0] == "Version: 1.2.3"
    assert output[1].startswith("Python: ")
    assert output[2].startswith("Python path: ")
    assert output[3:] == ["Extras: dashboard", "Additional packages: none"]


def test_status_reports_additional_receipt_requirements(
    monkeypatch, tmp_path, capsys
):
    environment = tmp_path / TOOL_PACKAGE
    environment.mkdir()
    (environment / "uv-receipt.toml").write_text(
        "[tool]\n"
        'requirements = [{ name = "ursa-ai" }, '
        '{ name = "pytest", specifier = ">=8" }]\n'
    )
    monkeypatch.setattr("ursa.cli.self.find_uv", lambda: Path("/bin/uv"))
    monkeypatch.setattr("ursa.cli.self.get_uv_tool_dir", lambda _uv: tmp_path)
    monkeypatch.setattr("ursa.cli.self.sys.prefix", str(environment))
    monkeypatch.setattr("ursa.cli.self.package_version", lambda _name: "1.2.3")

    show_status()

    assert "Additional packages: pytest>=8" in capsys.readouterr().out


def test_status_works_without_uv_tool_install(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ursa.cli.self.sys.prefix", str(tmp_path))
    monkeypatch.setattr("ursa.cli.self.package_version", lambda _name: "1.2.3")

    show_status()

    output = capsys.readouterr().out
    assert "Version: 1.2.3" in output
    assert "Extras: unavailable (not a uv tool installation)" in output


def test_self_registration_does_not_inspect_installation(monkeypatch, capsys):
    inspected = MagicMock(side_effect=AssertionError("installation inspected"))
    monkeypatch.setattr("ursa.cli.self.is_uv_tool_install", inspected)
    monkeypatch.setattr("ursa.cli.self._running_uv_receipt", inspected)
    parser = build_parser()

    parser.parse_args(["self", "status"])
    parser.parse_args(["self", "update"])
    parser.parse_args(["self", "modify"])
    inspected.assert_not_called()

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["self", "--help"])
    assert help_exit.value.code == 0
    assert "uv tool install" in capsys.readouterr().out


def test_self_registers_update_and_modify():
    parser = build_parser()

    update = parser.parse_args(["self", "update"])
    modify = parser.parse_args(["self", "modify", "--ref", "main"])

    assert callable(update.self["update"].action)
    assert modify.self.modify.ref == "main"
