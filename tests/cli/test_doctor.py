import asyncio
import json
import os
import subprocess
from pathlib import Path
from threading import Barrier, Event, current_thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import SecretStr

from ursa.cli import build_parser, main
from ursa.cli import doctor as doctor_mod
from ursa.cli.config import UrsaConfig
from ursa.cli.doctor import (
    CheckResult,
    CheckStatus,
    DoctorContext,
    check_additional_healthchecks,
    check_configuration,
    check_embedding_models,
    check_inference_providers,
    check_installation_integrity,
    check_llm_models,
    check_mcp_servers,
    check_ursa_on_path,
    render_report,
    run_checks,
)


def test_doctor_parser_registers_json_without_running_checks(monkeypatch):
    checked = MagicMock(side_effect=AssertionError("check ran while parsing"))
    monkeypatch.setattr(doctor_mod, "run_checks", checked)

    parsed = build_parser().parse_args(["self", "doctor", "--json"])

    assert parsed.self.doctor.json is True
    checked.assert_not_called()


def test_configuration_check_parses_every_file_and_reports_final_config(
    tmp_path,
):
    valid = tmp_path / "valid.yaml"
    invalid = tmp_path / "invalid.yaml"
    later = tmp_path / "later.json"
    valid.write_text("unknown_setting: accepted-by-parser\n", encoding="utf-8")
    invalid.write_text("broken: [\n", encoding="utf-8")
    later.write_text('{"group": "default"}\n', encoding="utf-8")

    result = check_configuration(
        DoctorContext(
            config=UrsaConfig(),
            config_paths=(valid, invalid, later),
        )
    )

    assert [check.name for check in result.checks] == [
        str(valid),
        str(invalid),
        str(later),
        "Final resolved configuration",
    ]
    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.ERROR,
        CheckStatus.OK,
        CheckStatus.OK,
    ]
    assert result.status is CheckStatus.ERROR


def test_configuration_check_reports_no_files_and_resolution_failure():
    result = check_configuration(
        DoctorContext(
            config=None,
            config_error=ValueError("merge failed"),
        )
    )

    assert [check.name for check in result.checks] == [
        "Configuration files",
        "Final resolved configuration",
    ]
    assert result.checks[0].status is CheckStatus.OK
    assert result.checks[1].status is CheckStatus.ERROR
    assert result.checks[1].message == "merge failed"


def test_installation_integrity_checks_receipt_dependencies_and_storage(
    monkeypatch, tmp_path
):
    prefix = tmp_path / "tool"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").write_text(
        '[tool]\nrequirements = [{ name = "ursa-ai", specifier = ">=1" }]\n',
        encoding="utf-8",
    )
    completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="Checked 42 packages", stderr=""
    )
    run = MagicMock(return_value=completed)
    monkeypatch.setattr(doctor_mod.sys, "prefix", str(prefix))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _name: "/bin/uv")
    monkeypatch.setattr(doctor_mod.subprocess, "run", run)
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "cache")

    result = check_installation_integrity(
        DoctorContext(config=SimpleNamespace(workspace=tmp_path / "workspace"))
    )

    assert result.status is CheckStatus.OK
    assert [check.name for check in result.checks] == [
        "uv tool receipt",
        "Python dependencies",
        "Workspace",
        "URSA cache",
    ]
    assert all(check.status is CheckStatus.OK for check in result.checks)
    assert "ursa-ai>=1" in result.checks[0].message
    run.assert_called_once_with(
        [
            "/bin/uv",
            "pip",
            "check",
            "--python",
            doctor_mod.sys.executable,
            "--no-progress",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=55,
    )


def test_installation_integrity_reports_independent_failures(
    monkeypatch, tmp_path
):
    prefix = tmp_path / "tool"
    prefix.mkdir()
    (prefix / "uv-receipt.toml").write_text(
        "[tool]\nrequirements = []\n", encoding="utf-8"
    )
    workspace = tmp_path / "not-a-directory"
    workspace.write_text("occupied", encoding="utf-8")
    monkeypatch.setattr(doctor_mod.sys, "prefix", str(prefix))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _name: "/bin/uv")
    monkeypatch.setattr(
        doctor_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="broken dependency"
        ),
    )
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "cache")

    result = check_installation_integrity(
        DoctorContext(config=SimpleNamespace(workspace=workspace))
    )

    assert [check.status for check in result.checks] == [
        CheckStatus.ERROR,
        CheckStatus.ERROR,
        CheckStatus.ERROR,
        CheckStatus.OK,
    ]
    assert "invalid" in result.checks[0].message.lower()
    assert result.checks[1].message == "broken dependency"


def test_installation_integrity_warns_when_uv_or_config_are_unavailable(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(doctor_mod.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(doctor_mod.shutil, "which", lambda _name: None)
    monkeypatch.setattr("ursa.security.URSA_CACHE_DIR", tmp_path / "cache")

    result = check_installation_integrity(DoctorContext(config=None))

    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.WARN,
        CheckStatus.WARN,
        CheckStatus.OK,
    ]


def test_result_tree_aggregates_status_and_drives_both_renderers():
    report = run_checks(
        DoctorContext(config=None),
        checks=(
            lambda _context: CheckResult("healthy", CheckStatus.OK),
            lambda _context: CheckResult(
                "optional", CheckStatus.WARN, "not configured"
            ),
        ),
    )

    assert report.status is CheckStatus.WARN
    optional = report.as_dict()["checks"][1]
    assert {key: optional[key] for key in ("name", "status", "message")} == {
        "name": "optional",
        "status": "warn",
        "message": "not configured",
    }
    assert isinstance(optional["duration_ms"], int)
    rendered = render_report(report)
    assert "─" in rendered
    assert "⚠️ URSA Doctor" in rendered
    assert "✅ healthy" in rendered
    assert "ms)" in rendered
    assert "1 ok  |  1 warn  |  0 errors" in rendered


def test_runner_contains_system_exit_and_malformed_results():
    called = []

    def exits(_context):
        called.append("exit")
        raise SystemExit("plugin tried to exit")

    def malformed(_context):
        called.append("malformed")
        return "not a result"

    def healthy(_context):
        called.append("healthy")
        return CheckResult("healthy", CheckStatus.OK)

    report = run_checks(
        DoctorContext(config=None), checks=(exits, malformed, healthy)
    )

    assert set(called) == {"exit", "malformed", "healthy"}
    assert [result.status for result in report.checks] == [
        CheckStatus.ERROR,
        CheckStatus.ERROR,
        CheckStatus.OK,
    ]


def test_runner_executes_headings_concurrently_and_preserves_order():
    rendezvous = Barrier(2, timeout=10)

    def check(name):
        def run(_context):
            rendezvous.wait()
            return CheckResult(name, CheckStatus.OK)

        return run

    report = run_checks(
        DoctorContext(config=None),
        checks=(check("first"), check("second")),
    )

    assert [result.name for result in report.checks] == ["first", "second"]
    assert all(result.status is CheckStatus.OK for result in report.checks)


def test_runner_times_out_a_check_without_waiting_for_its_thread(monkeypatch):
    release = Event()
    finished = Event()
    monkeypatch.setattr(doctor_mod, "CHECK_DEADLINE_SECONDS", 0)

    def stuck(_context):
        try:
            release.wait(timeout=10)
            return CheckResult("stuck", CheckStatus.OK)
        finally:
            finished.set()

    try:
        report = run_checks(
            DoctorContext(config=None),
            checks=(stuck,),
        )
    finally:
        release.set()

    assert report.checks[0].status is CheckStatus.ERROR
    assert report.checks[0].message == "Timed out after 0 seconds"
    assert finished.wait(timeout=10), "detached test worker did not exit"


@pytest.mark.asyncio
async def test_async_runner_contains_check_originated_cancellation():
    def cancelled(_context):
        raise asyncio.CancelledError

    report = await doctor_mod.run_checks_async(
        DoctorContext(config=None),
        checks=(
            cancelled,
            lambda _context: CheckResult("healthy", CheckStatus.OK),
        ),
    )

    assert [check.status for check in report.checks] == [
        CheckStatus.ERROR,
        CheckStatus.OK,
    ]


@pytest.mark.asyncio
async def test_async_runner_propagates_external_cancellation():
    started = Event()
    release = Event()
    finished = Event()

    def blocking(_context):
        started.set()
        try:
            release.wait(timeout=10)
            return CheckResult("blocking", CheckStatus.OK)
        finally:
            finished.set()

    task = asyncio.create_task(
        doctor_mod.run_checks_async(
            DoctorContext(config=None), checks=(blocking,)
        )
    )
    assert await asyncio.to_thread(started.wait, 10), "worker did not start"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    assert await asyncio.to_thread(finished.wait, 10), "worker did not exit"


def test_interactive_spinner_uses_requested_frames_and_updates_progress(
    monkeypatch,
):
    class FakeStatus:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.updates = []

        def start(self):
            self.started = True

        def stop(self):
            self.stopped = True

        def update(self, value):
            self.updates.append(value)

    class FakeConsole:
        is_terminal = True

        def __init__(self):
            self.status_instance = FakeStatus()
            self.initial = None
            self.options = None

        def status(self, initial, **options):
            self.initial = initial
            self.options = options
            return self.status_instance

    console = FakeConsole()
    results = (
        CheckResult("first", CheckStatus.OK),
        CheckResult("second", CheckStatus.WARN),
    )

    def fake_run_checks(_context, *, progress=None):
        for completed, result in enumerate(results, start=1):
            if progress is not None:
                progress(completed, len(results), result)
        return CheckResult("URSA Doctor", CheckStatus.WARN, checks=results)

    monkeypatch.setattr(doctor_mod, "run_checks", fake_run_checks)

    report = doctor_mod._run_human_checks(DoctorContext(config=None), console)

    status = console.status_instance
    assert report.name == "URSA Doctor"
    assert console.options["spinner"] == "dots"
    assert status.started and status.stopped
    assert len(status.updates) == len(results)
    assert f"{len(results)}/{len(results)}" in status.updates[-1].plain
    assert "complete" in status.updates[-1].plain


def test_spinner_rendering_failures_do_not_interrupt_checks(monkeypatch):
    class BrokenStatus:
        def start(self):
            pass

        def stop(self):
            raise RuntimeError("stop failed")

        def update(self, _value):
            raise RuntimeError("update failed")

    class FakeConsole:
        is_terminal = True

        def status(self, *_args, **_kwargs):
            return BrokenStatus()

    report = CheckResult(
        "URSA Doctor",
        CheckStatus.OK,
        checks=(CheckResult("healthy", CheckStatus.OK),),
    )

    def fake_run_checks(_context, *, progress=None):
        if progress is not None:
            progress(1, 1, report.checks[0])
        return report

    monkeypatch.setattr(doctor_mod, "run_checks", fake_run_checks)

    actual = doctor_mod._run_human_checks(
        DoctorContext(config=None), FakeConsole()
    )

    assert actual is report


def test_path_check_lists_every_other_ursa(monkeypatch, tmp_path):
    first = tmp_path / "one" / "ursa"
    second = tmp_path / "two" / "ursa"
    monkeypatch.setattr(
        doctor_mod,
        "_ursa_executables_on_path",
        lambda: [
            doctor_mod._PathUrsa(str(first.parent), first),
            doctor_mod._PathUrsa(str(second.parent), second),
        ],
    )
    monkeypatch.setattr(doctor_mod, "_running_ursa_path", lambda: first)

    result = check_ursa_on_path(DoctorContext(config=None))

    assert result.status is CheckStatus.WARN
    assert [check.message for check in result.checks] == [
        str(first),
        str(second),
    ]


def test_path_discovery_preserves_the_responsible_path_entry(
    monkeypatch, tmp_path
):
    first = str(tmp_path / "first-bin")
    second = str(tmp_path / "second-bin")
    monkeypatch.setenv("PATH", os.pathsep.join((first, second)))
    monkeypatch.setattr(
        doctor_mod.shutil,
        "which",
        lambda _name, *, path: str(Path(path) / "ursa"),
    )

    entries = doctor_mod._ursa_executables_on_path()

    assert [entry.path_entry for entry in entries] == [first, second]
    assert [entry.executable for entry in entries] == [
        Path(first, "ursa").absolute(),
        Path(second, "ursa").absolute(),
    ]


def test_path_check_marks_entries_that_shadow_current_ursa(
    monkeypatch, tmp_path
):
    shadowing = tmp_path / "one" / "ursa"
    current = tmp_path / "two" / "ursa"
    lower = tmp_path / "three" / "ursa"
    monkeypatch.setattr(
        doctor_mod,
        "_ursa_executables_on_path",
        lambda: [
            doctor_mod._PathUrsa(str(shadowing.parent), shadowing),
            doctor_mod._PathUrsa(str(current.parent), current),
            doctor_mod._PathUrsa(str(lower.parent), lower),
        ],
    )
    monkeypatch.setattr(doctor_mod, "_running_ursa_path", lambda: current)

    result = check_ursa_on_path(DoctorContext(config=None))

    assert result.status is CheckStatus.ERROR
    assert [check.name for check in result.checks] == [
        f"PATH {shadowing.parent} · SHADOWS CURRENT",
        f"PATH {current.parent} · current",
        f"PATH {lower.parent} · additional copy",
    ]
    assert [check.status for check in result.checks] == [
        CheckStatus.ERROR,
        CheckStatus.OK,
        CheckStatus.WARN,
    ]


def test_provider_checks_validate_every_provider(monkeypatch):
    providers = {"good": object(), "bad": object()}

    def list_models(provider):
        if provider is providers["bad"]:
            raise RuntimeError("credentials rejected")
        return [object(), object()]

    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models", list_models
    )
    context = DoctorContext(
        config=SimpleNamespace(inference_providers=providers)
    )

    result = check_inference_providers(context)

    assert result.status is CheckStatus.ERROR
    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.ERROR,
    ]
    assert "credentials rejected" in result.checks[1].message


def _resolved_keyring_config() -> UrsaConfig:
    return UrsaConfig.model_validate({
        "group": None,
        "inference_providers": {
            "shared": {"api_key": {"keyring": "shared-account"}}
        },
        "llm_model": {
            "model": "chat-test",
            "model_provider": "openai",
            "inference_provider": "shared",
        },
        "emb_model": {
            "model": "embedding-test",
            "model_provider": "openai",
            "inference_provider": "shared",
        },
        "mcp_servers": {
            "tools": {
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
                "headers": {
                    "Authorization": {
                        "keyring": "shared-account",
                        "template": "Bearer %s",
                    }
                },
            }
        },
    }).resolve()


def test_keyring_credentials_are_primed_once_on_the_calling_thread(
    monkeypatch,
):
    config = _resolved_keyring_config()
    calls = []

    def get_password(service, username):
        calls.append((service, username, current_thread()))
        return "token"

    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(get_password=get_password),
    )

    context = doctor_mod._prime_keyring_credentials(
        DoctorContext(config=config)
    )

    assert calls == [("ursa", "shared-account", current_thread())]
    assert isinstance(config.inference_providers["shared"].api_key, SecretStr)
    assert isinstance(config.llm_model.api_key, SecretStr)
    assert isinstance(config.emb_model.api_key, SecretStr)
    assert config.mcp_servers["tools"].headers["Authorization"] == (
        "Bearer token"
    )
    assert not context.credential_errors
    assert not context.mcp_credential_errors


def test_keyring_failure_is_reused_without_blocking_other_checks(monkeypatch):
    config = _resolved_keyring_config()
    calls = []

    def get_password(_service, _username):
        calls.append(current_thread())
        raise RuntimeError("keychain is locked")

    monkeypatch.setattr(
        "ursa.util.secrets.get_keyring",
        lambda: SimpleNamespace(get_password=get_password),
    )
    context = doctor_mod._prime_keyring_credentials(
        DoctorContext(config=config)
    )
    shared_provider = config.inference_providers["shared"]
    list_models = MagicMock(return_value=[])
    start_mcp = MagicMock(side_effect=AssertionError("MCP probe ran"))
    monkeypatch.setattr(
        "ursa.util.inference_providers.list_provider_models", list_models
    )
    monkeypatch.setattr("ursa.util.mcp.start_mcp_client", start_mcp)

    provider_result = check_inference_providers(context)
    llm_result = check_llm_models(context)
    embedding_result = check_embedding_models(context)
    mcp_result = check_mcp_servers(context)

    assert calls == [current_thread()]
    provider_checks = {check.name: check for check in provider_result.checks}
    assert provider_checks["shared"].status is CheckStatus.ERROR
    assert "locked" in provider_checks["shared"].message
    assert all(
        probe.status is CheckStatus.ERROR
        for probe in llm_result.checks[0].checks
    )
    assert embedding_result.status is CheckStatus.ERROR
    assert mcp_result.status is CheckStatus.ERROR
    assert "Authorization: keychain is locked" in mcp_result.checks[0].message
    assert all(
        call.args[0] is not shared_provider
        for call in list_models.call_args_list
    )
    start_mcp.assert_not_called()


class _FakeChatResponse:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls or []


class _FakeChatModel:
    def __init__(self):
        self.tool = False

    def invoke(self, _prompt):
        if self.tool:
            return _FakeChatResponse([
                {"name": "_ursa_doctor_echo", "args": {"value": "ursa"}}
            ])
        return _FakeChatResponse()

    def bind_tools(self, tools, *, tool_choice):
        assert tools[0].name == tool_choice == "_ursa_doctor_echo"
        self.tool = True
        return self

    def with_structured_output(self, schema, **_kwargs):
        return SimpleNamespace(invoke=lambda _prompt: schema(status="ok"))


def test_llm_check_calls_chat_forced_tool_use_and_structured_output():
    model = _FakeChatModel()
    config = SimpleNamespace(
        model="chat-test",
        inference_provider="test-provider",
        model_provider="test",
        base_url=None,
        api_key=None,
        init_chat_model=lambda: model,
    )
    context = DoctorContext(config=SimpleNamespace(llm_model=config))

    result = check_llm_models(context)

    assert result.name == "LLM models"
    assert result.status is CheckStatus.OK
    assert [probe.name for probe in result.checks[0].checks] == [
        "Chat invocation",
        "Forced tool use",
        "Structured output",
    ]


def test_llm_probe_failure_does_not_suppress_later_probes():
    class ChatFailureModel(_FakeChatModel):
        def invoke(self, prompt):
            if not self.tool:
                raise RuntimeError("chat failed")
            return super().invoke(prompt)

    model = ChatFailureModel()
    config = SimpleNamespace(
        model="chat-test",
        inference_provider="test-provider",
        model_provider="test",
        base_url=None,
        api_key=None,
        init_chat_model=lambda: model,
    )

    result = check_llm_models(
        DoctorContext(config=SimpleNamespace(llm_model=config))
    )

    assert [probe.status for probe in result.checks[0].checks] == [
        CheckStatus.ERROR,
        CheckStatus.OK,
        CheckStatus.OK,
    ]


@pytest.mark.parametrize(
    ("vector", "status"),
    [([0.1, 0.2], CheckStatus.OK), ([], CheckStatus.ERROR)],
)
def test_embedding_check_calls_model(vector, status):
    embedding = SimpleNamespace(embed_query=lambda _text: vector)
    config = SimpleNamespace(
        model="embedding-test",
        inference_provider="test-provider",
        model_provider="test",
        base_url=None,
        api_key=None,
        init_embedding=lambda: embedding,
    )
    context = DoctorContext(config=SimpleNamespace(emb_model=config))

    assert check_embedding_models(context).status is status


def test_mcp_check_connects_to_each_server(monkeypatch):
    class FakeClient:
        async def get_tools(self, *, server_name):
            if server_name == "bad":
                raise RuntimeError("cannot connect")
            return [object()]

    monkeypatch.setattr(
        "ursa.util.mcp.start_mcp_client", lambda _servers: FakeClient()
    )
    context = DoctorContext(
        config=SimpleNamespace(mcp_servers={"good": object(), "bad": object()})
    )

    result = check_mcp_servers(context)

    assert result.status is CheckStatus.ERROR
    assert [check.status for check in result.checks] == [
        CheckStatus.OK,
        CheckStatus.ERROR,
    ]


def test_mcp_check_accepts_an_empty_configuration():
    result = check_mcp_servers(
        DoctorContext(config=SimpleNamespace(mcp_servers={}))
    )

    assert result.status is CheckStatus.OK
    assert result.message == "No MCP servers are configured"


def test_additional_entry_points_are_sorted_and_isolated(monkeypatch):
    def warning_check(_context):
        return CheckResult("ignored", CheckStatus.WARN, "degraded")

    entries = [
        SimpleNamespace(name="broken", load=lambda: lambda _context: 1 / 0),
        SimpleNamespace(
            name="zeta",
            load=lambda: lambda _context: CheckResult(
                "ignored", CheckStatus.OK
            ),
        ),
        SimpleNamespace(name="alpha", load=lambda: warning_check),
    ]

    def discover(*, group):
        assert group == "ursa-healthcheck"
        return entries

    monkeypatch.setattr(doctor_mod, "entry_points", discover)

    result = check_additional_healthchecks(DoctorContext(config=None))

    assert result.status is CheckStatus.ERROR
    assert [check.name for check in result.checks] == [
        "alpha",
        "broken",
        "zeta",
    ]
    assert result.checks[0].message == "degraded"
    assert "division by zero" in result.checks[1].message


def test_main_doctor_json_writes_one_document(monkeypatch, capsys):
    report = CheckResult(
        "URSA doctor",
        CheckStatus.ERROR,
        checks=(CheckResult("broken", CheckStatus.ERROR, "boom"),),
    )

    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)
    monkeypatch.setattr("ursa.cli.inject_truststore_into_ssl", lambda: None)

    return_code = main(["self", "doctor", "--json"])

    assert return_code == 1
    assert json.loads(capsys.readouterr().out) == report.as_dict()


def test_human_output_falls_back_to_json_when_rich_rendering_fails(
    monkeypatch, capsys
):
    report = CheckResult(
        "URSA doctor",
        CheckStatus.WARN,
        checks=(CheckResult("optional", CheckStatus.WARN, "missing"),),
    )
    monkeypatch.setattr(doctor_mod, "run_checks", lambda _context: report)
    monkeypatch.setattr(
        doctor_mod,
        "_print_report",
        MagicMock(side_effect=RuntimeError("renderer failed")),
    )

    return_code = doctor_mod.run_doctor()

    assert return_code == 0
    assert json.loads(capsys.readouterr().out) == report.as_dict()
