"""Bounded dashboard previews without losing captured process output."""

from __future__ import annotations

import ast
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from ursa import security
from ursa.environments.visualization import get_environment_run_paths
from ursa_dashboard.app import create_app
from ursa_dashboard.credentials import MemoryCredentialStore
from ursa_dashboard.environment_run_manager import (
    EnvironmentRunManager,
    EnvironmentRunManagerConfig,
)
from ursa_dashboard.environment_run_ui import render_environment_run_detail_page
from ursa_dashboard.run_manager import RunConfig, RunManager


@pytest.fixture
def log_client(monkeypatch, tmp_path):
    root = tmp_path / "dashboard"
    monkeypatch.setattr(security, "URSA_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setenv("URSA_DASHBOARD_WORKSPACE_ROOT", str(root))
    monkeypatch.setenv("URSA_DASHBOARD_GROUP", "default")
    monkeypatch.setenv("URSA_DASHBOARD_MODE", "remote")
    monkeypatch.setenv("URSA_DASHBOARD_TOKEN", "test-log-token")
    monkeypatch.delenv("URSA_DASHBOARD_CONFIG", raising=False)
    run_dir = root / "runs" / "agent" / "run-1"
    (run_dir / "logs").mkdir(parents=True)
    meta = root / "_meta" / "runs"
    meta.mkdir(parents=True)
    (meta / "run-1.json").write_text(
        json.dumps({
            "run_id": "run-1",
            "agent_id": "agent",
            "status": "succeeded",
            "run_dir": "runs/agent/run-1",
        })
    )
    paths = get_environment_run_paths("default", "env-1")
    paths.logs_dir.mkdir(parents=True)
    paths.manifest_path.write_text(
        json.dumps({
            "run_id": "env-1",
            "status": "succeeded",
            "environment_name": "team",
        })
    )
    # Use actual source bytes as the captured byte stream, including all newlines.
    data = Path(__file__).read_bytes()
    for directory in (run_dir, paths.run_dir):
        for stream in ("stdout", "stderr"):
            (directory / "logs" / f"{stream}.log").write_bytes(data)
    client = TestClient(create_app(credential_store=MemoryCredentialStore()))
    client.headers["Authorization"] = "Bearer test-log-token"
    yield client, run_dir, paths.run_dir, data
    client.close()


@pytest.mark.parametrize("prefix", ["/runs/run-1", "/environment-runs/env-1"])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_log_download_and_bounded_tail(log_client, prefix, stream):
    client, _, _, data = log_client
    url = f"{prefix}/logs/{stream}"
    full = client.get(url)
    assert full.status_code == 200
    assert full.content == data
    assert "attachment" in full.headers["content-disposition"]
    assert full.headers["cache-control"] == "no-store"
    tail = client.get(url + "?tail_bytes=128")
    assert tail.status_code == 200
    assert "Recent output only" in tail.text
    assert tail.text.endswith(data[-128:].decode("utf-8", errors="replace"))
    assert len(tail.content) < 300
    assert client.get(url + "?tail_bytes=0").status_code == 422
    assert client.get(url + "?tail_bytes=262145").status_code == 422
    assert client.get(prefix + "/logs/other").status_code == 422
    client.headers.pop("Authorization")
    assert client.get(url).status_code == 401


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
def test_log_download_rejects_symlink_escape_and_missing_files(
    log_client, tmp_path, prefix, index
):
    client = log_client[0]
    path = log_client[index] / "logs" / "stdout.log"
    path.unlink()
    assert client.get(prefix + "/logs/stdout").status_code == 404
    outside = tmp_path / "outside.log"
    outside.write_text("not a run log")
    path.symlink_to(outside)
    assert client.get(prefix + "/logs/stdout").status_code == 403
    assert client.get("/runs/missing/logs/stdout").status_code == 404
    assert (
        client.get("/environment-runs/missing/logs/stdout").status_code == 404
    )


def test_initial_sse_tail_is_bounded_but_reconnect_resumes(log_client):
    client, run_dir, _, _ = log_client
    records = [
        {
            "seq": i,
            "type": "log",
            "payload": {"stream": "stdout", "text": str(i)},
        }
        for i in range(1, 21)
    ]
    path = run_dir / "logs" / "events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in records))

    def sequences(response):
        assert response.status_code == 200
        return [
            int(line[4:])
            for line in response.text.splitlines()
            if line.startswith("id: ")
        ]

    recent = sequences(client.get("/runs/run-1/stream?tail_bytes=200"))
    assert recent[-1] == 20
    assert len(recent) < 20
    assert sequences(client.get("/runs/run-1/stream")) == list(range(1, 21))
    assert sequences(
        client.get(
            "/runs/run-1/stream?tail_bytes=1", headers={"Last-Event-ID": "17"}
        )
    ) == [18, 19, 20]
    assert sequences(client.get("/api/runs/run-1/events?after_seq=19")) == [20]


@pytest.mark.asyncio
@pytest.mark.parametrize("manager_kind", ["agent", "environment"])
async def test_full_raw_capture_default_and_optional_cap(
    tmp_path, manager_kind
):
    data = Path(__file__).read_bytes()
    # Exercise multiple pipe reads, byte-for-byte, independently of UI previews.
    if manager_kind == "agent":
        manager = RunManager(dashboard_root=tmp_path / "dashboard")
        manager._emit = AsyncMock()
        cap = RunConfig().stdout_cap_bytes
    else:
        cap = EnvironmentRunManagerConfig().stdout_cap_bytes
    assert cap is None
    for cap in (None, 128):
        reader = asyncio.StreamReader()
        reader.feed_data(data)
        reader.feed_eof()
        path = tmp_path / f"{manager_kind}-{cap}.log"
        if manager_kind == "agent":
            await manager._drain_stream(
                run_id="run-1",
                agent_id="agent",
                stream_name="stdout",
                stream=reader,
                log_path=path,
                cap_bytes=cap,
            )
        else:
            await EnvironmentRunManager._drain(reader, path, cap)
        captured = path.read_bytes()
        if cap is None:
            assert captured == data
        else:
            assert captured.startswith(data[:cap])
            assert data[cap:] not in captured
        assert reader.at_eof()


def _dashboard_js():
    import ursa_dashboard.app as app_module

    tree = ast.parse(Path(app_module.__file__).read_text())
    return next(
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "DASHBOARD_JS"
            for target in node.targets
        )
    )


def _javascript_runner():
    runner = shutil.which("node") or shutil.which("jsc")
    jsc = Path(
        "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"
    )
    runner = runner or (str(jsc) if jsc.is_file() else None)
    if runner is None:
        pytest.skip("JavaScript runtime (node or jsc) not installed")
    return runner


def test_dashboard_javascript_bounds_and_batches(tmp_path):
    runner = _javascript_runner()
    js = _dashboard_js()
    detail = render_environment_run_detail_page(run_id="env-1", manifest={})
    detail_js = detail.split("<script>", 1)[1].split("</script>", 1)[0]
    log_functions = js[
        js.index("  function stripBackspaces(") : js.index(
            "  function closeRunStream("
        )
    ]
    add_event = detail_js[
        detail_js.index("function addEvent(") : detail_js.index(
            "function setupControls("
        )
    ]
    script = tmp_path / "dashboard-check.js"
    script.write_text(
        "new Function("
        + json.dumps(js)
        + ");\n"
        + "new Function("
        + json.dumps(detail_js)
        + ");\n"
        + "const state = {settings:{ui:{stdout_buffer_lines:100000000}},logs:{stdout:'',stderr:''}};\n"
        + "function scheduleLogRender() {}\n"
        + log_functions
        + "appendLog('stdout', 'x'.repeat(300000));\n"
        + "if (state.logs.stdout.length !== 131072) throw Error('unbounded log');\n"
        + "if (trimToLastLines('a\\nb\\nc', 2) !== 'b\\nc') throw Error('line cap');\n"
        + "let events=[], live=true, selected=-1, topology={}, cy=null, renderTimer=null, graphNeedsRebuild=false;\n"
        + "let timers=[], draws=0; function setTimeout(fn) {timers.push(fn);return 1;}\n"
        + "function payload(e) {return e.payload || {};} function renderAll() {draws++;}\n"
        + "function initializeGraph() {throw Error('unnecessary graph rebuild');}\n"
        + add_event
        + "for (let i=0;i<20;i++) addEvent({seq:i});\n"
        + "if (timers.length !== 1 || draws !== 0) throw Error('unbatched render');\n"
        + "timers[0](); if (draws !== 1 || selected !== 19) throw Error('lost event');\n"
    )
    subprocess.run(
        [runner, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _set_run_status(log_client, prefix, status):
    _, run_dir, environment_dir, _ = log_client
    if prefix.startswith("/environment-runs"):
        path = environment_dir / "manifest.json"
    else:
        path = run_dir.parents[2] / "_meta" / "runs" / "run-1.json"
    record = json.loads(path.read_text())
    record["status"] = status
    path.write_text(json.dumps(record))


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
@pytest.mark.parametrize("status", ["succeeded", "failed", "cancelled"])
def test_delete_stored_logs_preserves_other_run_data(
    log_client, prefix, index, status
):
    client, _, _, data = log_client
    _set_run_status(log_client, prefix, status)
    root = log_client[index]
    # Keep real source data in non-log files to verify the deletion's exact scope.
    keep = [
        root / "logs" / "events.jsonl",
        root / "output.json",
        root / "graph_store.sqlite",
        root / "artifact.txt",
    ]
    for path in keep:
        path.write_bytes(data)
    response = client.delete(prefix + "/logs")
    assert response.status_code == 200
    assert response.json() == {
        "deleted_files": ["stdout.log", "stderr.log"],
        "deleted_bytes": len(data) * 2,
    }
    assert all(path.read_bytes() == data for path in keep)
    assert client.get(prefix + "/logs/stdout").status_code == 404
    assert client.get(prefix + "/logs/stderr").status_code == 404
    assert client.delete(prefix + "/logs").json() == {
        "deleted_files": [],
        "deleted_bytes": 0,
    }


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
@pytest.mark.parametrize(
    "status", ["queued", "starting", "running", "cancelling", "unknown"]
)
def test_deleting_active_or_unknown_status_logs_is_rejected(
    log_client, prefix, index, status
):
    client, _, _, data = log_client
    _set_run_status(log_client, prefix, status)
    assert client.delete(prefix + "/logs").status_code == 409
    assert (log_client[index] / "logs" / "stdout.log").read_bytes() == data
    assert (log_client[index] / "logs" / "stderr.log").read_bytes() == data


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
def test_deleting_logs_requires_auth_and_known_run(log_client, prefix, index):
    client, _, _, data = log_client
    assert (
        client.delete(
            prefix.replace("run-1", "missing").replace("env-1", "missing")
            + "/logs"
        ).status_code
        == 404
    )
    client.headers.pop("Authorization")
    assert client.delete(prefix + "/logs").status_code == 401
    assert (log_client[index] / "logs" / "stdout.log").read_bytes() == data


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
@pytest.mark.parametrize("inside", [False, True])
def test_delete_rejects_log_symlinks_before_removing_either_file(
    log_client, tmp_path, prefix, index, inside
):
    client, _, _, data = log_client
    root = log_client[index]
    target = (root if inside else tmp_path) / "preserve.txt"
    target.write_bytes(data)
    path = root / "logs" / "stderr.log"
    path.unlink()
    path.symlink_to(target)
    assert client.delete(prefix + "/logs").status_code == 403
    assert target.read_bytes() == data
    assert (root / "logs" / "stdout.log").read_bytes() == data
    assert path.is_symlink()


@pytest.mark.parametrize(
    "prefix,index", [("/runs/run-1", 1), ("/environment-runs/env-1", 2)]
)
def test_delete_rejects_symlinked_log_directory(log_client, prefix, index):
    client = log_client[0]
    root = log_client[index]
    (root / "logs").rename(root / "preserve-logs")
    (root / "logs").symlink_to(root / "preserve-logs", target_is_directory=True)
    assert client.delete(prefix + "/logs").status_code == 403
    assert (root / "preserve-logs" / "stdout.log").is_file()
    assert (root / "preserve-logs" / "stderr.log").is_file()


@pytest.mark.parametrize(
    "prefix,manager,run_id",
    [
        ("/runs/run-1", RunManager, "run-1"),
        ("/environment-runs/env-1", EnvironmentRunManager, "env-1"),
    ],
)
def test_delete_rejects_terminal_run_while_worker_is_still_tracked(
    log_client, monkeypatch, prefix, manager, run_id
):
    # Terminal metadata can precede the end of process/pipe cleanup.
    monkeypatch.setattr(
        manager,
        "_inflight",
        property(lambda _: {run_id: object()}),
        raising=False,
    )
    assert log_client[0].delete(prefix + "/logs").status_code == 409


def test_log_delete_buttons_confirm_guard_and_handle_errors(tmp_path):
    runner = _javascript_runner()
    js = _dashboard_js()
    main_functions = js[
        js.index("  function setLogDownloads(") : js.index(
            "  function sessionLogMetaText("
        )
    ]
    detail = render_environment_run_detail_page(run_id="env-1", manifest={})
    environment_function = detail[
        detail.index(
            "async function deleteStoredEnvironmentLogs("
        ) : detail.index("function setupControls(")
    ]
    assert "$('#deleteStoredLogsBtn').onclick = deleteStoredRunLogs" in js
    assert (
        "$('deleteStoredLogs').onclick = deleteStoredEnvironmentLogs" in detail
    )
    script = tmp_path / "delete-buttons.js"
    script.write_text(
        r"""
let state={viewRunId:'run-1', viewRunKind:'static', _logToken:1, _deletingLogs:false};
let button={disabled:false}, envButton={disabled:false}, select={value:'run-1'};
let status={textContent:''}, links={stdout:{},stderr:{}};
let calls=[], alerts=[], allow=false, clears=0, mode='ok', confirmations=0;
let deletingStoredLogs=false;
const runId='env-1';
function escHtml(s) {return s;}
function $(id) {
  if (id === '#logDownloadRun') return select;
  if (id === '#deleteStoredLogsBtn') return button;
  if (id === '#downloadstdout') return links.stdout;
  if (id === '#downloadstderr') return links.stderr;
  if (id === 'deleteStoredLogs') return envButton;
  if (id === 'logDeleteStatus') return status;
  throw Error('Unexpected element '+id);
}
function confirm(message) {
  if (!message.includes('Permanently delete') || !message.includes('provenance')) throw Error('Missing warning');
  confirmations++;
  return allow;
}
function alert(message) {alerts.push(message);}
function clearLogs() {clears++;}
function updateSummary() {envButton.disabled=deletingStoredLogs;}
async function api(method, path) {
  if (method !== 'DELETE' || path !== '/runs/run-1/logs') throw Error('Wrong deletion target');
  calls.push(path);
  if (mode === 'fail') throw Error('Wait for the run to finish');
  if (mode === 'switched') state._logToken++;
  return {deleted_bytes:100};
}
async function fetch(path, options) {
  if (path !== '/environment-runs/env-1/logs' || options.method !== 'DELETE') throw Error('Wrong environment target');
  calls.push(path);
  return {ok:mode !== 'fail', json:async()=>mode === 'fail' ? {detail:'Wait for the run to finish'} : {deleted_bytes:100}};
}
function check(value, message) {if (!value) throw Error(message);}
"""
        + main_functions
        + "\n"
        + environment_function
        + r"""
(async () => {
  setLogDownloads(['run-1']);
  state.viewRunKind='stream'; select.onchange();
  check(button.disabled, 'Active selection should be disabled');
  select.value='old'; select.onchange();
  check(!button.disabled, 'An old run must remain selectable during live work');
  state.viewRunKind='static'; select.value='run-1'; select.onchange();
  await deleteStoredRunLogs();
  check(calls.length === 0 && clears === 0, 'Cancelled confirmation deleted data');
  allow=true;
  await deleteStoredRunLogs();
  check(calls.length === 1 && clears === 1, 'Confirmed deletion did not clear preview');
  check(!state._deletingLogs && !button.disabled, 'Button remained busy');
  button.disabled=true; const before=confirmations;
  await deleteStoredRunLogs();
  check(calls.length === 1 && confirmations === before, 'Disabled deletion ran');
  button.disabled=false; mode='switched';
  await deleteStoredRunLogs();
  check(clears === 1, 'Deletion cleared another view after a switch');
  mode='fail';
  await deleteStoredRunLogs();
  check(alerts[alerts.length-1].includes('Wait for the run to finish'), 'Missing error');
  check(clears === 1 && !state._deletingLogs, 'Failed deletion altered preview or stayed busy');
  allow=false; const beforeEnv=calls.length;
  await deleteStoredEnvironmentLogs();
  check(calls.length === beforeEnv, 'Cancelled environment confirmation deleted data');
  allow=true; mode='ok';
  await deleteStoredEnvironmentLogs();
  check(status.textContent.includes('100') && !deletingStoredLogs && !envButton.disabled, 'Environment deletion failed');
  mode='fail';
  await deleteStoredEnvironmentLogs();
  check(status.textContent.includes('Wait for the run to finish'), 'Missing environment error');
  check(!deletingStoredLogs && !envButton.disabled, 'Environment button remained busy');
})().then(() => {
  if (typeof print === 'function') print('delete checks passed');
  else console.log('delete checks passed');
}, error => {throw error;});
"""
    )
    result = subprocess.run(
        [runner, str(script)],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert "delete checks passed" in result.stdout, result.stderr
