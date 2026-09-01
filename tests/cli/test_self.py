import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ursa.cli import build_parser, main
from ursa.cli.self import (
    TOOL_PACKAGE,
    build_upgrade_requirement,
    get_package_repository,
    show_status,
    upgrade,
)


def _receipt(path: Path, requirement: str) -> Path:
    path.write_text(f"[tool]\nrequirements = [{requirement}]\n")
    return path


@pytest.fixture
def isolated_uv_tool(monkeypatch, tmp_path):
    """Provide a complete fake uv tool environment without invoking uv."""
    tool_dir = tmp_path / "tools"
    environment = tool_dir / TOOL_PACKAGE
    environment.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    (environment / "uv-receipt.toml").write_text(
        "[tool]\n"
        f'requirements = [{{ name = "ursa-ai", extras = ["dashboard"], '
        f'directory = "{source.as_posix()}" }}, '
        '{ name = "pytest", specifier = ">=8" }]\n'
    )
    uv = tmp_path / "bin" / "uv"
    execv = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr(
        "ursa.cli.self._uv_install",
        lambda: (uv, environment / "uv-receipt.toml"),
    )
    monkeypatch.setattr("ursa.cli.self.sys.prefix", str(environment))
    monkeypatch.setattr("ursa.cli.self.package_version", lambda _name: "1.2.3")
    monkeypatch.setattr("ursa.cli.self.shutil.which", lambda _name: str(uv))
    monkeypatch.setattr(
        "ursa.cli.self.subprocess.run",
        MagicMock(
            return_value=MagicMock(
                stdout=(
                    "ursa-ai v1.2.3 [extras: dashboard] [with: pytest>=8] "
                    "[CPython 3.12.0]\n- ursa\nother v1.0.0\n"
                )
            )
        ),
    )
    monkeypatch.setattr(
        "ursa.cli.self.get_package_repository",
        lambda: "https://example.test/ursa.git",
    )
    monkeypatch.setattr("ursa.cli.self.os.execv", execv)
    return {"execv": execv, "source": source, "uv": uv}


def test_self_status_command_uses_isolated_receipt(isolated_uv_tool, capsys):
    main(["self", "status"])

    output = capsys.readouterr().out
    assert "Version: 1.2.3" in output
    assert "Extras: dashboard" in output
    assert "Additional packages: pytest>=8" in output
    isolated_uv_tool["execv"].assert_not_called()


@pytest.mark.parametrize(
    ("arguments", "expected_tail"),
    [
        (
            ["modify", "--ref", "main"],
            [
                "--with",
                "pytest>=8",
                "ursa-ai[dashboard] @ git+https://example.test/ursa.git@main",
            ],
        ),
        (
            ["modify", "--version", "2.0.0"],
            ["--with", "pytest>=8", "ursa-ai[dashboard]==2.0.0"],
        ),
        (
            ["modify", "--with", "tox"],
            [
                "--with",
                "pytest>=8",
                "--with",
                "tox",
                "ursa-ai[dashboard] @ {source}",
            ],
        ),
        (
            ["modify", "--extra", "fm"],
            [
                "--with",
                "pytest>=8",
                "ursa-ai[dashboard,fm] @ {source}",
            ],
        ),
        (
            ["modify", "--clean", "--extra", "fm", "--with", "tox"],
            ["--with", "tox", "ursa-ai[fm] @ {source}"],
        ),
    ],
)
def test_self_modify_commands_are_isolated(
    isolated_uv_tool, arguments, expected_tail
):
    source_uri = isolated_uv_tool["source"].as_uri()
    expected_tail = [value.format(source=source_uri) for value in expected_tail]

    with pytest.raises(SystemExit, match="0"):
        main(["self", *arguments])

    isolated_uv_tool["execv"].assert_called_once_with(
        str(isolated_uv_tool["uv"]),
        [
            str(isolated_uv_tool["uv"]),
            "tool",
            "install",
            "--force",
            "--reinstall",
            "--compile-bytecode",
            "--python",
            str(Path(sys.executable).resolve()),
            *expected_tail,
        ],
    )


def test_self_update_command_is_isolated(isolated_uv_tool):
    with pytest.raises(SystemExit, match="0"):
        main(["self", "update"])

    isolated_uv_tool["execv"].assert_called_once_with(
        str(isolated_uv_tool["uv"]),
        [
            str(isolated_uv_tool["uv"]),
            "tool",
            "upgrade",
            "--reinstall",
            "--compile-bytecode",
            TOOL_PACKAGE,
        ],
    )


@pytest.mark.parametrize("command", ["update", "modify"])
def test_self_management_rejects_non_uv_install_without_exec(
    monkeypatch, tmp_path, command
):
    execv = MagicMock(side_effect=AssertionError("process replaced"))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr(
        "ursa.cli.self._uv_install",
        MagicMock(side_effect=SystemExit("not installed with uv tool install")),
    )
    monkeypatch.setattr("ursa.cli.self.os.execv", execv)

    with pytest.raises(SystemExit, match="not installed with"):
        main(["self", command])

    execv.assert_not_called()


def test_self_modify_rejects_version_and_ref_together(isolated_uv_tool):
    with pytest.raises(SystemExit) as exc_info:
        main([
            "self",
            "modify",
            "--version",
            "2.0.0",
            "--ref",
            "main",
        ])

    assert exc_info.value.code == 2
    isolated_uv_tool["execv"].assert_not_called()


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
        f'{{ name = "ursa-ai", directory = "{source.as_posix()}" }}',
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


def test_local_install_can_switch_to_git_ref(monkeypatch, tmp_path):
    environment = tmp_path / TOOL_PACKAGE
    environment.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    (environment / "uv-receipt.toml").write_text(
        "[tool]\n"
        f'requirements = [{{ name = "ursa-ai", directory = "{source.as_posix()}" }}, '
        '{ name = "pytest", specifier = ">=8" }]\n'
    )
    monkeypatch.setattr(
        "ursa.cli.self._uv_install",
        lambda: (Path("/bin/uv"), environment / "uv-receipt.toml"),
    )
    monkeypatch.setattr(
        "ursa.cli.self.get_package_repository",
        lambda: "https://example.test/ursa.git",
    )
    execv = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.self.os.execv", execv)

    with pytest.raises(SystemExit, match="0"):
        upgrade(ref="main")

    uv = str(Path("/bin/uv"))
    execv.assert_called_once_with(
        uv,
        [
            uv,
            "tool",
            "install",
            "--force",
            "--reinstall",
            "--compile-bytecode",
            "--python",
            str(Path(sys.executable).resolve()),
            "--with",
            "pytest>=8",
            "ursa-ai @ git+https://example.test/ursa.git@main",
        ],
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


def test_status_works_without_uv_tool_install(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("ursa.cli.self.sys.prefix", str(tmp_path))
    monkeypatch.setattr("ursa.cli.self.package_version", lambda _name: "1.2.3")

    show_status()

    output = capsys.readouterr().out
    assert "Version: 1.2.3" in output
    assert "Extras: unavailable (not a uv tool installation)" in output


def test_self_registration_does_not_inspect_installation(monkeypatch, capsys):
    inspected = MagicMock(side_effect=AssertionError("installation inspected"))
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
