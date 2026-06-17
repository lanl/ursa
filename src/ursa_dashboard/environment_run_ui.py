from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from typing import Any

CYTOSCAPE_CDN_URL = (
    "https://cdn.jsdelivr.net/npm/cytoscape/dist/cytoscape.min.js"
)


def _escape(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _status_class(status: Any) -> str:
    value = str(status or "unknown").lower()
    if value in {"succeeded", "failed", "cancelled", "running"}:
        return value
    return "unknown"


def _run_sort_key(run: Mapping[str, Any]) -> str:
    return str(
        run.get("updated_at")
        or run.get("created_at")
        or run.get("run_id")
        or ""
    )


def render_environment_runs_page(
    *,
    dashboard_group: str,
    runs: Sequence[Mapping[str, Any]],
) -> str:
    sorted_runs = sorted(runs, key=_run_sort_key, reverse=True)
    cards = []
    for run in sorted_runs:
        run_id_raw = str(run.get("run_id", ""))
        run_id = _escape(run_id_raw)
        name = _escape(run.get("environment_name", "Environment"))
        env_type = _escape(run.get("environment_type", ""))
        status = _escape(run.get("status", "unknown"))
        status_class = _status_class(run.get("status"))
        updated = _escape(run.get("updated_at", ""))
        preview = _escape(run.get("task_preview", ""))
        cards.append(
            "<article class='run-card'>"
            "<div class='run-card-top'>"
            f"<div><a class='run-title' href='/ui/environment-runs/{run_id}'>{name}</a>"
            f"<div class='muted mono'>{run_id}</div></div>"
            f"<span class='status {status_class}'>{status}</span>"
            "</div>"
            "<div class='run-meta'>"
            f"<span>{env_type}</span><span>Updated {updated}</span>"
            "</div>"
            f"<p class='task-preview'>{preview}</p>"
            f"<a class='open-link' href='/ui/environment-runs/{run_id}'>Open work replay →</a>"
            "</article>"
        )
    body = "".join(cards) or (
        "<div class='empty'>No environment runs recorded yet. Run an environment "
        "with <code>run_with_visualization</code> or "
        "<code>arun_with_visualization</code> to create one.</div>"
    )
    return f"""
<!doctype html>
<html>
<head>
  <title>URSA Environment Runs</title>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <style>
    :root {{ color-scheme: dark; --bg:#0b1020; --panel:#111a33; --panel2:#16213d; --line:#26324d; --text:#e7edf7; --muted:#9aa8bd; --accent:#8bd3ff; --good:#54d17a; --bad:#ff6b6b; --warn:#ffd166; }}
    * {{ box-sizing:border-box; }}
    body {{ font-family:system-ui,-apple-system,Segoe UI,sans-serif; margin:0; background:radial-gradient(circle at top left,#182442 0,#0b1020 38rem); color:var(--text); }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
    .page {{ max-width:1180px; margin:0 auto; padding:28px; }}
    .top {{ display:flex; justify-content:space-between; gap:16px; align-items:flex-start; margin-bottom:22px; }}
    h1 {{ margin:0 0 6px; font-size:2rem; }}
    .muted {{ color:var(--muted); }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:16px; }}
    .run-card {{ background:linear-gradient(180deg,var(--panel),#0f1830); border:1px solid var(--line); border-radius:16px; padding:16px; box-shadow:0 12px 32px rgba(0,0,0,.24); }}
    .run-card-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
    .run-title {{ font-size:1.1rem; font-weight:700; color:var(--text); }}
    .run-meta {{ display:flex; flex-wrap:wrap; gap:8px; color:var(--muted); font-size:.88rem; margin-top:12px; }}
    .run-meta span {{ background:#0c1429; border:1px solid var(--line); padding:4px 8px; border-radius:999px; }}
    .task-preview {{ color:#c9d6e8; min-height:2.6em; white-space:pre-wrap; overflow:hidden; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; }}
    .open-link {{ display:inline-block; margin-top:8px; font-weight:650; }}
    .status {{ border-radius:999px; padding:4px 9px; font-size:.78rem; text-transform:capitalize; border:1px solid var(--line); background:#26324d; }}
    .status.succeeded {{ background:rgba(84,209,122,.14); border-color:rgba(84,209,122,.55); color:#b8f4c7; }}
    .status.failed {{ background:rgba(255,107,107,.14); border-color:rgba(255,107,107,.55); color:#ffc0c0; }}
    .status.running {{ background:rgba(139,211,255,.14); border-color:rgba(139,211,255,.55); color:#ccefff; }}
    .status.cancelled {{ background:rgba(255,209,102,.14); border-color:rgba(255,209,102,.55); color:#ffe3a0; }}
    .empty {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:24px; color:var(--muted); }}
    code {{ background:#0c1429; padding:2px 5px; border-radius:5px; color:#ccefff; }}
  </style>
</head>
<body>
  <main class='page'>
    <div class='top'>
      <div>
        <h1>Environment Runs</h1>
        <div class='muted'>Replay agent-team and symposium work for group <span class='mono'>{_escape(dashboard_group)}</span>.</div>
      </div>
      <a href='/ui'>Dashboard</a>
    </div>
    <section class='grid'>{body}</section>
  </main>
</body>
</html>
"""


def render_environment_run_detail_page(
    *,
    run_id: str,
    manifest: Mapping[str, Any],
) -> str:
    title = _escape(manifest.get("environment_name") or run_id)
    safe_run_id = _escape(run_id)
    run_id_json = json.dumps(run_id)
    manifest_json = json.dumps(dict(manifest), ensure_ascii=False, default=str)
    cytoscape_url = _escape(CYTOSCAPE_CDN_URL)
    return (
        DETAIL_TEMPLATE.replace("__TITLE__", title)
        .replace("__RUN_ID__", safe_run_id)
        .replace("__RUN_ID_JSON__", run_id_json)
        .replace("__MANIFEST_JSON__", manifest_json)
        .replace("__CYTOSCAPE_URL__", cytoscape_url)
    )


DETAIL_TEMPLATE = r"""
<!doctype html>
<html>
<head>
  <title>__TITLE__ - URSA Environment Run</title>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <script src='__CYTOSCAPE_URL__'></script>
  <style>
    :root { color-scheme: dark; --bg:#0b1020; --panel:#111a33; --panel2:#16213d; --line:#26324d; --text:#e7edf7; --muted:#9aa8bd; --accent:#8bd3ff; --accent2:#b79cff; --good:#54d17a; --bad:#ff6b6b; --warn:#ffd166; --chip:#0c1429; }
    * { box-sizing:border-box; }
    body { font-family:system-ui,-apple-system,Segoe UI,sans-serif; margin:0; background:var(--bg); color:var(--text); }
    a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
    button,input { background:#182442; color:var(--text); border:1px solid #405071; border-radius:8px; padding:7px 9px; }
    button { cursor:pointer; } button:hover { border-color:var(--accent); }
    h1 { margin:4px 0 5px; font-size:1.45rem; } h2 { margin:0 0 12px; font-size:1.05rem; } h3 { margin:14px 0 8px; font-size:.92rem; color:#cfe2ff; }
    .page { min-height:100vh; display:flex; flex-direction:column; }
    .hero { padding:18px 22px; border-bottom:1px solid var(--line); background:linear-gradient(120deg,#111a33,#0b1020 70%); }
    .hero-row { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }
    .muted { color:var(--muted); } .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    .status { border-radius:999px; padding:4px 9px; font-size:.8rem; text-transform:capitalize; border:1px solid var(--line); background:#26324d; display:inline-flex; align-items:center; gap:6px; }
    .status.succeeded { background:rgba(84,209,122,.14); border-color:rgba(84,209,122,.55); color:#b8f4c7; }
    .status.failed { background:rgba(255,107,107,.14); border-color:rgba(255,107,107,.55); color:#ffc0c0; }
    .status.running { background:rgba(139,211,255,.14); border-color:rgba(139,211,255,.55); color:#ccefff; }
    .status.cancelled { background:rgba(255,209,102,.14); border-color:rgba(255,209,102,.55); color:#ffe3a0; }
    .summary { display:grid; grid-template-columns:minmax(340px,2fr) repeat(4,minmax(120px,1fr)); gap:10px; margin-top:14px; }
    .metric { background:rgba(17,26,51,.85); border:1px solid var(--line); border-radius:14px; padding:11px; min-width:0; }
    .metric .label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
    .metric .value { font-weight:750; margin-top:4px; overflow:hidden; text-overflow:ellipsis; }
    .metric.task .value { white-space:pre-wrap; max-height:4.2em; font-weight:600; color:#dce8f8; }
    .layout { display:grid; grid-template-columns:minmax(390px,34vw) minmax(480px,1fr) minmax(330px,24vw); gap:0; flex:1; min-height:0; }
    .panel { padding:16px; overflow:auto; border-right:1px solid var(--line); min-height:0; } .panel.right { border-right:0; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px; margin-bottom:14px; box-shadow:0 10px 28px rgba(0,0,0,.18); }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
    #graph { height:420px; border-radius:14px; border:1px solid var(--line); background:radial-gradient(circle at center,#172441,#0c1429); overflow:hidden; }
    .graph-note { color:var(--muted); font-size:.83rem; margin-top:8px; }
    .search { width:100%; margin-bottom:10px; }
    .timeline-card { background:linear-gradient(180deg,var(--panel),#101932); border:1px solid var(--line); border-left:4px solid #405071; border-radius:14px; padding:11px; margin-bottom:9px; cursor:pointer; }
    .timeline-card:hover,.timeline-card.selected { border-color:var(--accent); border-left-color:var(--accent); }
    .timeline-card.failed { border-left-color:var(--bad); } .timeline-card.completed { border-left-color:var(--good); } .timeline-card.active { border-left-color:var(--accent); }
    .timeline-top { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .timeline-title { font-weight:750; } .timeline-msg { color:#c9d6e8; margin-top:4px; white-space:pre-wrap; }
    .timeline-meta { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
    .chip { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 8px; background:var(--chip); color:#c9d6e8; font-size:.82rem; }
    .chip.good { border-color:rgba(84,209,122,.55); color:#b8f4c7; } .chip.bad { border-color:rgba(255,107,107,.6); color:#ffc0c0; } .chip.active { border-color:rgba(139,211,255,.6); color:#ccefff; } .chip.warn { border-color:rgba(255,209,102,.55); color:#ffe3a0; }
    .event-hero { background:linear-gradient(180deg,#16213d,#101932); border:1px solid var(--line); border-radius:18px; padding:18px; }
    .event-title-row { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; }
    .event-title { font-size:1.35rem; font-weight:800; margin:0 0 4px; }
    .event-message { color:#e7edf7; white-space:pre-wrap; margin:12px 0; font-size:1.02rem; line-height:1.45; }
    .content-block { background:#0c1429; border:1px solid var(--line); border-radius:12px; padding:12px; margin-top:10px; }
    .content-block pre { margin:0; white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88rem; line-height:1.45; }
    .content-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }
    .final-content,.task-content,.path-content { max-height:340px; overflow:auto; white-space:pre-wrap; line-height:1.45; }
    .path-content { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88rem; }
    .empty { color:var(--muted); padding:14px; border:1px dashed var(--line); border-radius:12px; background:rgba(17,26,51,.45); }
    .fallback-graph { display:grid; gap:8px; padding:10px; }
    .fallback-node { display:flex; justify-content:space-between; align-items:center; gap:8px; background:var(--chip); border:1px solid var(--line); border-radius:12px; padding:8px 10px; cursor:pointer; }
    .fallback-node.active { border-color:var(--accent); box-shadow:0 0 0 1px rgba(139,211,255,.35) inset; }
    .fallback-node.completed { border-color:rgba(84,209,122,.55); } .fallback-node.failed { border-color:rgba(255,107,107,.65); }
    .small-link { font-size:.86rem; color:var(--muted); }
    @media (max-width:1180px) { .layout { grid-template-columns:1fr; } .panel { border-right:0; border-bottom:1px solid var(--line); } .summary { grid-template-columns:1fr 1fr; } #graph { height:340px; } }
  </style>
</head>
<body>
<div class='page'>
  <header class='hero'>
    <div class='hero-row'>
      <div>
        <a href='/ui/environment-runs'>← Environment Runs</a>
        <h1>__TITLE__</h1>
        <div class='muted'>Run <span class='mono'>__RUN_ID__</span></div>
      </div>
      <div id='statusBadge'></div>
    </div>
    <section class='summary' id='summary'></section>
  </header>
  <main class='layout'>
    <aside class='panel left'>
      <section class='card'>
        <h2>Environment Graph</h2>
        <div id='graph'></div>
        <div id='graphNote' class='graph-note'>Loading Cytoscape.js graph…</div>
      </section>
      <section class='card'>
        <h2>Work Timeline</h2>
        <div class='toolbar'>
          <button id='live'>Pause live</button>
          <button id='prev'>Older</button>
          <button id='next'>Newer</button>
          <input id='scrub' type='range' min='0' max='0' value='0' />
        </div>
        <input id='timelineSearch' class='search' placeholder='Search visible timeline…' />
        <div id='timeline'></div>
      </section>
    </aside>
    <section class='panel'>
      <h2>Current Activity</h2>
      <section class='event-hero' id='currentEvent'>
        <div class='empty'>Waiting for events…</div>
      </section>
    </section>
    <aside class='panel right'>
      <section class='card'>
        <h2>Task</h2>
        <div id='task' class='task-content muted'></div>
      </section>
      <section class='card'>
        <h2>Final Result</h2>
        <div id='finalResult' class='final-content muted'>Waiting for a completion event…</div>
      </section>
      <section class='card'>
        <h2>Workspace</h2>
        <div id='workspace' class='path-content muted'>No workspace path recorded yet.</div>
      </section>
      <section class='card'>
        <h2>Raw Events</h2>
        <div class='small-link'>Raw event JSON is available separately at <a id='rawEventsLink' href='#'>the events API</a>.</div>
      </section>
    </aside>
  </main>
</div>
<script>
const runId = __RUN_ID_JSON__;
const manifest = __MANIFEST_JSON__;
let events = [];
let selected = -1;
let live = true;
let selectedParticipant = null;
let cy = null;
let topology = null;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const payload = (e) => e && e.payload && typeof e.payload === 'object' ? e.payload : {};
const text = (v) => typeof v === 'string' ? v : JSON.stringify(v, null, 2);
function statusClass(s) { const v = String(s || 'unknown').toLowerCase(); return ['succeeded','failed','running','cancelled'].includes(v) ? v : 'unknown'; }
function fmtDuration(seconds) { const n = Number(seconds); if (!Number.isFinite(n)) return '—'; if (n < 60) return `${n.toFixed(n < 10 ? 1 : 0)}s`; const m = Math.floor(n / 60); const s = Math.round(n % 60); return `${m}m ${s}s`; }
function readableType(type) {
  const labels = {
    topology_declared:'Environment structure', team_started:'Team started', team_completed:'Team completed', team_failed:'Team failed',
    delegation_started:'Delegation started', delegation_completed:'Delegation completed', delegation_failed:'Delegation failed',
    symposium_started:'Symposium started', symposium_completed:'Symposium completed', symposium_failed:'Symposium failed',
    symposium_phase_started:'Phase started', symposium_phase_completed:'Phase completed', initial_work_started:'Initial work started', initial_work_completed:'Initial work completed',
    review_round_started:'Review round started', review_round_completed:'Review round completed', revision_round_started:'Revision round started', revision_round_completed:'Revision round completed',
    synthesis_started:'Synthesis started', synthesis_completed:'Final synthesis completed', tool_search:'Tool search', tool_execute:'Tool execution', tool_write:'File write', tool_safety_check:'Tool safety check'
  };
  return labels[type] || String(type || 'event').replaceAll('_',' ').replace(/^./, c => c.toUpperCase());
}
function phaseLabel(e) { const p = payload(e); if (p.round_index) return `Round ${p.round_index} · ${p.phase || e.phase || ''}`; return p.phase || e.phase || p.stage || e.stage || ''; }
function eventLevel(e) {
  const type = String(e.event_type || ''); const p = payload(e);
  if ((e.level || p.level) === 'error' || type.endsWith('_failed') || p.phase === 'error' || p.error || p.error_type) return 'failed';
  if (type.endsWith('_started') || p.phase === 'start') return 'active';
  if (type.endsWith('_completed') || type.endsWith('_declared') || p.phase === 'end') return 'completed';
  return '';
}
function topologyNodeIds() { const top = topology || extractTopology(); return new Set(((top && top.nodes) || []).map(n => String(n.id || n.name))); }
function nodeById(id) { const top = topology || extractTopology(); return ((top && top.nodes) || []).find(n => String(n.id || n.name) === id); }
function participantNameById(id) { const n = nodeById(id); return n ? String(n.name || n.id).split('.').pop() : String(id || '').split('.').pop(); }
function participantName(obj) { return obj && (obj.name || obj.id) ? String(obj.name || obj.id).split('.').pop() : ''; }
function participantId(obj) { return obj && (obj.id || obj.name) ? String(obj.id || obj.name) : ''; }
function participantIdFromValue(v) {
  if (!v) return '';
  const ids = topologyNodeIds();
  if (typeof v === 'object') {
    const id = String(v.id || v.name || '');
    if (ids.has(id)) return id;
    const bySuffix = [...ids].find(nodeId => nodeId.endsWith('.' + id));
    return bySuffix || '';
  }
  const s = String(v); if (ids.has(s)) return s;
  return [...ids].find(nodeId => nodeId.endsWith('.' + s)) || '';
}
function explicitParticipantForEvent(e) {
  const p = payload(e);
  for (const obj of [e.source, e.target, p.source, p.target]) {
    if (obj && obj.kind !== 'tool') { const id = participantIdFromValue(obj); if (id) return id; }
  }
  for (const key of ['environment_member_id','environment_member','member','agent','agent_id','participant','participant_id','owner','owner_id']) {
    const id = participantIdFromValue(p[key]); if (id) return id;
  }
  return '';
}
function activeDelegationOwners(upto) {
  const ids = topologyNodeIds(); const active = new Set();
  for (let i = 0; i <= upto; i++) {
    const e = events[i]; if (!e) continue; const type = String(e.event_type || '');
    if (type === 'delegation_started') { const target = participantIdFromValue(e.target || payload(e).target); if (target) active.add(target); }
    if (type === 'delegation_completed' || type === 'delegation_failed') {
      const source = participantIdFromValue(e.source || payload(e).source); const target = participantIdFromValue(e.target || payload(e).target);
      if (source && ids.has(source)) active.delete(source); if (target && ids.has(target)) active.delete(target);
    }
  }
  return [...active];
}
function inferredOwner(e, index) {
  const explicit = explicitParticipantForEvent(e); if (explicit) return explicit;
  const isTool = (e.source && e.source.kind === 'tool') || payload(e).tool || String(e.event_type || '').startsWith('tool_');
  if (!isTool) return '';
  const active = activeDelegationOwners(index);
  return active.length === 1 ? active[0] : '';
}
function toolName(e) { return payload(e).tool || (e.source && e.source.kind === 'tool' ? e.source.name : ''); }
function sourceTargetText(e, index) {
  const s = participantIdFromValue(e.source || payload(e).source); const t = participantIdFromValue(e.target || payload(e).target);
  if (s && t) return `${participantNameById(s)} → ${participantNameById(t)}`;
  const owner = inferredOwner(e, index); const tool = toolName(e);
  if (tool) return owner ? `${participantNameById(owner)} used ${tool}` : `Tool: ${tool} · member not recorded`;
  return participantName(e.source || payload(e).source) || participantName(e.target || payload(e).target) || '';
}
function eventMatchesParticipant(e, id, index) {
  if (!id) return true;
  const s = participantIdFromValue(e.source || payload(e).source); const t = participantIdFromValue(e.target || payload(e).target); const owner = inferredOwner(e, index);
  return s === id || t === id || owner === id;
}
function eventSearchText(e, index) { return [e.event_type, e.message, e.stage, e.phase, sourceTargetText(e, index), text(payload(e))].join(' ').toLowerCase(); }
function extractTopology() { const ev = events.find(e => e.event_type === 'topology_declared' && payload(e).topology) || events.find(e => payload(e).topology); return ev ? payload(ev).topology : null; }
function extractTask() { for (const e of events) { const p = payload(e); if (p.task) return text(p.task); } return manifest.task_preview || ''; }
function extractFinal() { const preferred = ['team_completed','symposium_completed','synthesis_completed']; for (const type of preferred) { for (let i = events.length - 1; i >= 0; i--) { const p = payload(events[i]); if (events[i].event_type === type && (p.result || p.final)) return text(p.result || p.final); } } for (let i = events.length - 1; i >= 0; i--) { const p = payload(events[i]); if (p.result && String(events[i].event_type || '').endsWith('_completed')) return text(p.result); } return ''; }
function durationFromEvents() { for (let i = events.length - 1; i >= 0; i--) { const p = payload(events[i]); if (p.elapsed_seconds != null && ['team_completed','symposium_completed'].includes(events[i].event_type)) return Number(p.elapsed_seconds); } const ns = events.map(e => Number(e.monotonic_timestamp_ns)).filter(n => Number.isFinite(n)); if (ns.length > 1) return (Math.max(...ns) - Math.min(...ns)) / 1e9; return null; }
function workspacePaths() {
  const paths = [];
  const add = (p) => { if (p && typeof p === 'string' && p.startsWith('/')) paths.push(p); };
  if (manifest.paths) { add(manifest.paths.run_dir); add(manifest.paths.artifacts_dir); add(manifest.paths.logs_dir); }
  for (const e of events) { const p = payload(e); add(p.workspace); add(p.workspace_path); add(p.path); }
  return [...new Set(paths)];
}
function likelyWorkspacePath() {
  const dirs = workspacePaths().map(p => p.match(/\.[A-Za-z0-9_+-]+$/) ? p.split('/').slice(0,-1).join('/') : p);
  const workspace = dirs.find(p => /workspace/i.test(p));
  return workspace || dirs[0] || '';
}
function updateSummary() {
  const duration = durationFromEvents(); const status = manifest.status || 'unknown'; const task = extractTask(); const workspace = likelyWorkspacePath();
  $('statusBadge').innerHTML = `<span class='status ${statusClass(status)}'>${esc(status)}</span>`;
  $('summary').innerHTML = [
    ['Task', esc(task || 'No task payload recorded.'), 'task'],
    ['Status', `<span class='status ${statusClass(status)}'>${esc(status)}</span>`, ''],
    ['Events', events.length || '—', ''],
    ['Duration', fmtDuration(duration), ''],
    ['Workspace', workspace ? esc(workspace) : '—', '']
  ].map(([k,v,cls]) => `<div class='metric ${cls}'><div class='label'>${k}</div><div class='value'>${v}</div></div>`).join('');
  $('task').textContent = task || 'No task payload recorded.';
  const final = extractFinal(); $('finalResult').textContent = final || 'Waiting for a completion event…'; $('finalResult').classList.toggle('muted', !final);
  $('workspace').textContent = workspacePaths().join('\n') || 'No workspace path recorded yet.';
  $('rawEventsLink').href = '/environment-runs/' + encodeURIComponent(runId) + '/events';
}
function communicationKind(e) {
  if (e.event_type === 'delegation_completed') return 'response';
  if (e.event_type === 'delegation_started') return 'delegates_to';
  if (String(e.event_type || '').startsWith('tool_')) return toolName(e) || 'tool';
  return String(e.event_type || 'communication').replace(/_started$|_completed$/,'');
}
function topologyToElements(top) {
  if (!top) return [];
  const nodeSet = new Set((top.nodes || []).map(n => String(n.id || n.name)));
  const nodes = (top.nodes || []).map(n => ({ data:{ id:String(n.id || n.name), label:String(n.name || n.id), kind:n.kind || '', role:n.role || '' }, classes:`node-${n.kind || 'participant'}` }));
  const edgeMap = new Map();
  const addEdge = (source, target, kind) => { if (!source || !target || !nodeSet.has(source) || !nodeSet.has(target)) return; const key = `${source}->${target}:${kind || 'link'}`; if (!edgeMap.has(key)) edgeMap.set(key, { data:{ id:key, source, target, label:kind || '', kind:kind || '' }, classes:`edge-${kind || 'link'}` }); };
  (top.edges || []).forEach(e => addEdge(String(e.source), String(e.target), e.kind || 'link'));
  events.forEach((e, i) => {
    let s = participantIdFromValue(e.source || payload(e).source); let t = participantIdFromValue(e.target || payload(e).target);
    const owner = inferredOwner(e, i); if ((!s || !t) && owner && toolName(e)) { s = owner; t = owner; }
    addEdge(s, t, communicationKind(e));
  });
  return [...nodes, ...edgeMap.values()];
}
function edgeBetween(source, target) { if (!cy || !source || !target) return null; return cy.edges().filter(edge => edge.data('source') === source && edge.data('target') === target).first(); }
function initializeGraph() {
  topology = extractTopology();
  if (!topology) { $('graph').innerHTML = `<div class='empty'>Waiting for a topology event…</div>`; $('graphNote').textContent = 'No topology has been recorded yet.'; return; }
  if (!window.cytoscape) { renderFallbackGraph(); $('graphNote').textContent = 'Cytoscape.js did not load, so a simplified graph fallback is shown.'; return; }
  if (cy) cy.destroy();
  cy = cytoscape({ container:$('graph'), elements:topologyToElements(topology), style:[
    { selector:'node', style:{ 'label':'data(label)', 'font-size':12, 'color':'#e7edf7', 'text-outline-width':2, 'text-outline-color':'#111a33', 'background-color':'#405071', 'border-width':1, 'border-color':'#8ba0c7', 'width':50, 'height':50 } },
    { selector:'.node-environment', style:{ 'background-color':'#b79cff', 'shape':'round-rectangle', 'width':74, 'height':42 } },
    { selector:'edge', style:{ 'curve-style':'bezier', 'target-arrow-shape':'triangle', 'line-color':'#526487', 'target-arrow-color':'#526487', 'label':'data(label)', 'font-size':9, 'color':'#9aa8bd', 'text-rotation':'autorotate', 'width':2 } },
    { selector:'edge[source = target]', style:{ 'curve-style':'bezier', 'loop-direction':'45deg', 'loop-sweep':'70deg' } },
    { selector:'node.active', style:{ 'background-color':'#8bd3ff', 'border-color':'#d8f3ff', 'border-width':3 } },
    { selector:'edge.active', style:{ 'line-color':'#8bd3ff', 'target-arrow-color':'#8bd3ff', 'width':4 } },
    { selector:'node.completed', style:{ 'background-color':'#54d17a', 'border-color':'#b8f4c7' } },
    { selector:'edge.completed', style:{ 'line-color':'#54d17a', 'target-arrow-color':'#54d17a', 'width':3 } },
    { selector:'node.failed', style:{ 'background-color':'#ff6b6b', 'border-color':'#ffc0c0' } },
    { selector:'edge.failed', style:{ 'line-color':'#ff6b6b', 'target-arrow-color':'#ff6b6b', 'width':4 } },
    { selector:'.selected', style:{ 'border-width':4, 'border-color':'#ffd166' } }
  ], layout:{ name:'breadthfirst', directed:true, padding:24, spacingFactor:1.25 } });
  cy.on('tap', 'node', (evt) => { selectedParticipant = selectedParticipant === evt.target.id() ? null : evt.target.id(); renderAll(); });
  cy.on('tap', 'edge', (evt) => {
    const d = evt.target.data(); const upto = selected >= 0 ? selected : events.length - 1; let idx = -1;
    for (let i = upto; i >= 0; i--) {
      const e = events[i];
      if (participantIdFromValue(e.source || payload(e).source) === d.source && participantIdFromValue(e.target || payload(e).target) === d.target) { idx = i; break; }
    }
    if (idx >= 0) selectEvent(idx);
  });
  $('graphNote').textContent = 'Click graph nodes to filter the timeline. Completed return messages use reverse arrows when the event records a reverse source and target.';
  applyGraphState();
}
function renderFallbackGraph() {
  const top = topology || extractTopology(); const parts = (top && top.nodes) || [];
  $('graph').innerHTML = `<div class='fallback-graph'>${parts.map(p => `<div class='fallback-node' data-id='${esc(String(p.id || p.name))}'><span>${esc(p.name || p.id)}</span><span class='chip'>${esc(p.kind || '')}</span></div>`).join('')}</div>`;
  $('graph').querySelectorAll('.fallback-node').forEach(el => el.onclick = () => { selectedParticipant = selectedParticipant === el.dataset.id ? null : el.dataset.id; renderAll(); });
}
function applyGraphState() {
  if (!topology) return; if (!cy) { renderFallbackGraph(); return; }
  cy.elements().removeClass('active completed failed selected');
  const upto = selected >= 0 ? selected : events.length - 1;
  for (let i = 0; i <= upto; i++) {
    const e = events[i]; const cls = eventLevel(e); let s = participantIdFromValue(e.source || payload(e).source); let t = participantIdFromValue(e.target || payload(e).target); const owner = inferredOwner(e, i);
    if ((!s || !t) && owner && toolName(e)) { s = owner; t = owner; }
    if (!cls) continue;
    for (const id of [s,t]) if (id) cy.$id(id).removeClass('active').addClass(cls);
    if (cls === 'completed' || cls === 'failed') { const reverse = edgeBetween(t,s); if (reverse) reverse.removeClass('active'); }
    const edge = edgeBetween(s,t); if (edge) edge.removeClass('active completed failed').addClass(cls);
  }
  if (selectedParticipant) cy.$id(selectedParticipant).addClass('selected');
  if (selected >= 0 && events[selected]) { const e = events[selected]; const ids = [participantIdFromValue(e.source || payload(e).source), participantIdFromValue(e.target || payload(e).target), inferredOwner(e, selected)]; ids.filter(Boolean).forEach(id => cy.$id(id).addClass('selected')); }
}
function timelineEvents() {
  const q = $('timelineSearch').value.trim().toLowerCase(); const upto = selected >= 0 ? selected : events.length - 1;
  return events.map((e,i) => [e,i]).filter(([,i]) => i <= upto).filter(([e,i]) => eventMatchesParticipant(e, selectedParticipant, i)).filter(([e,i]) => !q || eventSearchText(e, i).includes(q)).reverse();
}
function renderTimeline() {
  const rows = timelineEvents(); $('scrub').max = Math.max(0, events.length - 1); $('scrub').value = selected < 0 ? Math.max(0, events.length - 1) : selected;
  $('timeline').innerHTML = rows.length ? rows.map(([e,i]) => {
    const cls = eventLevel(e); const st = sourceTargetText(e, i); const p = payload(e); const dur = p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : (p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : '');
    return `<article class='timeline-card ${cls} ${i === selected ? 'selected' : ''}' data-index='${i}'><div class='timeline-top'><div><div class='timeline-title'>${esc(readableType(e.event_type))}</div><div class='timeline-msg'>${esc(e.message || p.message || '')}</div></div><span class='chip'>#${esc(e.seq ?? i + 1)}</span></div><div class='timeline-meta'>${phaseLabel(e) ? `<span class='chip'>${esc(phaseLabel(e))}</span>` : ''}${st ? `<span class='chip'>${esc(st)}</span>` : ''}${dur ? `<span class='chip'>${esc(dur)}</span>` : ''}</div></article>`;
  }).join('') : `<div class='empty'>No visible events match the current playback state or filter.</div>`;
  $('timeline').querySelectorAll('.timeline-card').forEach(el => el.onclick = () => selectEvent(Number(el.dataset.index)));
}
function block(label, value) { if (value == null || value === '') return ''; return `<div class='content-block'><div class='content-label'>${esc(label)}</div><pre>${esc(text(value))}</pre></div>`; }
function renderCurrentEvent() {
  const e = selected >= 0 ? events[selected] : events[events.length - 1]; if (!e) { $('currentEvent').innerHTML = `<div class='empty'>Waiting for events…</div>`; return; }
  const idx = events.indexOf(e); const p = payload(e); const owner = inferredOwner(e, idx); const tool = toolName(e); const cls = eventLevel(e);
  const chips = [phaseLabel(e), sourceTargetText(e, idx), p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : '', p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : ''].filter(Boolean);
  let sections = '';
  if (tool && !owner) sections += block('Assignment', 'This tool event did not record the member that invoked it. The UI is not assigning it to a graph participant to avoid misleading attribution.');
  sections += block('Task / instruction', p.task || p.prompt);
  sections += block('Tool query / command', p.query || p.command || p.input);
  sections += block('File or path', p.path || p.filename);
  sections += block('Result', p.result || p.output || p.final);
  sections += block('Error', p.error || p.error_type);
  sections += block('Safety rationale', p.reason);
  if (!sections) sections = block('Details', p);
  $('currentEvent').innerHTML = `<div class='event-title-row'><div><div class='event-title'>${esc(readableType(e.event_type))}</div><div class='muted'>Event #${esc(e.seq ?? idx + 1)}${owner ? ` · ${esc(participantNameById(owner))}` : ''}${tool ? ` · ${esc(tool)}` : ''}</div></div><span class='chip ${cls === 'failed' ? 'bad' : cls === 'completed' ? 'good' : cls === 'active' ? 'active' : ''}'>${esc(cls || 'event')}</span></div><div class='event-message'>${esc(e.message || p.message || '')}</div><div class='timeline-meta'>${chips.map(c => `<span class='chip'>${esc(c)}</span>`).join('')}</div>${sections}`;
}
function selectEvent(i) { if (i == null || i < 0 || i >= events.length) return; selected = i; live = false; $('live').textContent = 'Resume live'; renderAll(); }
function renderAll() { updateSummary(); renderTimeline(); renderCurrentEvent(); applyGraphState(); }
function addEvent(e) { if (!e || typeof e !== 'object') return; events.push(e); if (!topology && payload(e).topology) initializeGraph(); else if (topology && window.cytoscape) initializeGraph(); if (live) selected = events.length - 1; renderAll(); }
function setupControls() {
  $('live').onclick = () => { live = !live; $('live').textContent = live ? 'Pause live' : 'Resume live'; if (live && events.length) { selected = events.length - 1; renderAll(); } };
  $('prev').onclick = () => selectEvent(Math.max(0, (selected < 0 ? events.length : selected) - 1));
  $('next').onclick = () => selectEvent(Math.min(events.length - 1, selected + 1));
  $('scrub').oninput = () => selectEvent(Number($('scrub').value));
  $('timelineSearch').oninput = renderTimeline;
}
async function init() {
  setupControls(); updateSummary();
  const detail = await fetch('/environment-runs/'+encodeURIComponent(runId)).then(r => r.json()); Object.assign(manifest, detail); updateSummary();
  const data = await fetch('/environment-runs/'+encodeURIComponent(runId)+'/events').then(r => r.json()); (data.events || []).forEach(e => events.push(e));
  selected = events.length ? events.length - 1 : -1; topology = extractTopology(); initializeGraph(); renderAll();
  const last = events[events.length - 1]; const src = new EventSource('/environment-runs/'+encodeURIComponent(runId)+'/stream?after_seq='+(last ? last.seq : 0));
  src.onmessage = (ev) => addEvent(JSON.parse(ev.data)); src.onerror = () => {};
}
init().catch(err => { console.error(err); $('currentEvent').innerHTML = `<div class='empty'>Failed to initialize run view: ${esc(err.message)}</div>`; });
</script>
</body>
</html>
"""
