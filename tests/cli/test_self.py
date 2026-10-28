from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ursa.cli import main
from ursa.cli.self import (
    TOOL_PACKAGE,
    build_upgrade_requirement,
)


def _receipt(path: Path, requirement: str) -> Path:
    path.write_text(f"[tool]\nrequirements = [{requirement}]\n")
    return path


@pytest.fixture
def isolated_uv_tool(monkeypatch, tmp_path):
    environment = tmp_path / "tools" / TOOL_PACKAGE
    environment.mkdir(parents=True)
    source = tmp_path / "source"
    source.mkdir()
    receipt = _receipt(
        environment / "uv-receipt.toml",
        f'{{ name = "ursa-ai", extras = ["dashboard"], '
        f'directory = "{source.as_posix()}" }}, '
        '{ name = "pytest", specifier = ">=8" }',
    )
    uv = tmp_path / "bin" / "uv"
    execv = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr(
        "ursa.cli.self._uv_install",
        lambda: (uv, receipt),
    )
    monkeypatch.setattr(
        "ursa.cli.self.get_package_repository",
        lambda: "https://example.test/ursa.git",
    )
    monkeypatch.setattr("ursa.cli.self.os.execv", execv)
    return {"execv": execv, "receipt": receipt, "source": source, "uv": uv}


@pytest.mark.parametrize("option", ["--config", "--defaults-json"])
def test_self_setup_routes_paths(monkeypatch, tmp_path, option):
    called = MagicMock(return_value=True)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr("ursa.cli.setup.setup.run_setup", called)
    path = tmp_path / "input"
    main(["self", "setup", option, str(path)])
    called.assert_called_once_with(
        path if option == "--config" else None,
        path if option == "--defaults-json" else None,
    )


def test_self_setup_launches_selected_action(monkeypatch):
    from ursa.cli.setup.setup import SetupAction

    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr(
        "ursa.cli.setup.setup.run_setup",
        MagicMock(return_value=SetupAction.TERMINAL),
    )
    execvp = MagicMock(side_effect=SystemExit(0))
    monkeypatch.setattr("ursa.cli.self.os.execvp", execvp)
    with pytest.raises(SystemExit, match="0"):
        main(["self", "setup"])
    execvp.assert_called_once_with("ursa", ["ursa"])


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
            ["modify", "--extra", "fm", "--with", "tox"],
            [
                "--with",
                "pytest>=8",
                "--with",
                "tox",
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

    uv = str(isolated_uv_tool["uv"])
    isolated_uv_tool["execv"].assert_called_once()
    executable, argv = isolated_uv_tool["execv"].call_args.args
    assert executable == uv and argv[-len(expected_tail) :] == expected_tail


def test_self_update_command_is_isolated(isolated_uv_tool):
    with pytest.raises(SystemExit, match="0"):
        main(["self", "update"])

    uv = str(isolated_uv_tool["uv"])
    executable, argv = isolated_uv_tool["execv"].call_args.args
    assert executable == uv and argv[-2:] == [
        "--compile-bytecode",
        TOOL_PACKAGE,
    ]


def test_self_management_rejects_non_uv_install_without_exec(
    monkeypatch, tmp_path
):
    execv = MagicMock(side_effect=AssertionError("process replaced"))
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)
    monkeypatch.setattr(
        "ursa.cli.self._uv_install",
        MagicMock(side_effect=SystemExit("not installed with uv tool install")),
    )
    monkeypatch.setattr("ursa.cli.self.os.execv", execv)

    with pytest.raises(SystemExit, match="not installed with"):
        main(["self", "update"])

    execv.assert_not_called()


def test_upgrade_requirement_uses_selected_extras(tmp_path):
    receipt = _receipt(
        tmp_path / "receipt.toml",
        '{ name = "ursa-ai", extras = ["dashboard", "fm"], specifier = ">=1" }',
    )
    assert build_upgrade_requirement(receipt, extras=["office_readers"]) == (
        "ursa-ai[office_readers]>=1"
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
