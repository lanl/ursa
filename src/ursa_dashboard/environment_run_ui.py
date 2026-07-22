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
    if value in {
        "succeeded",
        "failed",
        "cancelled",
        "running",
        "queued",
        "starting",
        "cancelling",
    }:
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
    team_starter_yaml: str,
    symposium_starter_yaml: str,
) -> str:
    sorted_runs = sorted(runs, key=_run_sort_key, reverse=True)
    cards = []
    for run in sorted_runs:
        run_id_raw = str(run.get("run_id", ""))
        run_id = _escape(run_id_raw)
        name = _escape(run.get("environment_name", "Environment"))
        env_type_raw = str(run.get("environment_type", ""))
        env_type = _escape(env_type_raw)
        env_label = (
            "Team"
            if "team" in env_type_raw.lower()
            else "Symposium"
            if "symposium" in env_type_raw.lower()
            else "Environment"
        )
        status = _escape(run.get("status", "unknown"))
        status_class = _status_class(run.get("status"))
        updated = _escape(run.get("updated_at", ""))
        preview = _escape(run.get("task_preview", ""))
        cards.append(
            f"<article class='run-card' data-status='{status_class}' "
            f"data-type='{_escape(env_label.lower())}' data-run-id='{run_id}'>"
            "<div class='run-card-top'>"
            f"<div><a class='run-title' href='/ui/environment-runs/{run_id}'>{name}</a>"
            f"<div class='run-id mono'>{run_id}</div></div>"
            f"<span class='status {status_class}'>{status}</span>"
            "</div>"
            "<div class='run-meta'>"
            f"<span class='type-chip' title='{env_type}'>{env_label}</span>"
            f"<span class='time-chip'>Updated <time datetime='{updated}'>{updated}</time></span>"
            "</div>"
            f"<p class='task-preview'>{preview}</p>"
            "<div class='card-footer'>"
            f"<a class='open-link' href='/ui/environment-runs/{run_id}'>Open work replay"
            "<svg class='icon' viewBox='0 0 24 24' aria-hidden='true'>"
            "<path d='M5 12h14m-6-6 6 6-6 6'/></svg></a>"
            "</div>"
            "</article>"
        )
    body = "".join(cards) or (
        "<div class='empty'><div class='empty-icon'>✦</div>"
        "<h2>No environment runs yet</h2>"
        "<p>Create a team or symposium here, or continue using "
        "<code>run_with_visualization</code> from Python.</p>"
        "<div class='empty-actions'><button class='btn primary' data-create='agent_team'>"
        "Create a team</button><button class='btn' data-create='agent_symposium'>"
        "Create a symposium</button></div></div>"
    )
    team_json = json.dumps(team_starter_yaml)
    symposium_json = json.dumps(symposium_starter_yaml)
    run_count = len(sorted_runs)
    active_count = sum(
        1
        for run in sorted_runs
        if str(run.get("status", "")).lower()
        in {"queued", "starting", "running", "cancelling"}
    )
    return f"""
<!doctype html>
<html>
<head>
  <title>URSA Environment Runs</title>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <style>
    :root {{ color-scheme:light; --bg:#fff; --panel:#fafafa; --panel-strong:#fff; --line:#e2e2e2; --text:#111; --muted:#666; --accent:#0b57d0; --accent-hover:#0842a0; --good:#188038; --bad:#b3261e; --warn:#9a5b00; --chip:#f5f7fb; --shadow:0 1px 2px rgba(0,0,0,.04); }}
    :root[data-theme='dark'] {{ color-scheme:dark; --bg:#111418; --panel:#1c2026; --panel-strong:#171b20; --line:#3a404a; --text:#eceff4; --muted:#aab3bf; --accent:#8ab4ff; --accent-hover:#aecbfa; --good:#81c995; --bad:#f28b82; --warn:#fdd663; --chip:#252b33; --shadow:0 1px 3px rgba(0,0,0,.3); }}
    * {{ box-sizing:border-box; }}
    body {{ font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:var(--bg); color:var(--text); }}
    a {{ color:var(--accent); text-decoration:none; }}
    a:hover {{ color:var(--accent-hover); }}
    button,input,select,textarea {{ font:inherit; }}
    button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible {{ outline:3px solid color-mix(in srgb,var(--accent) 28%,transparent); outline-offset:2px; }}
    .page {{ max-width:1180px; margin:0 auto; padding:30px 24px 48px; }}
    .top {{ display:flex; justify-content:space-between; gap:24px; align-items:flex-start; margin-bottom:22px; }}
    .eyebrow {{ display:flex; align-items:center; gap:7px; color:var(--muted); font-size:.78rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase; margin-bottom:8px; }}
    h1 {{ margin:0 0 6px; font-size:clamp(1.65rem,3vw,2.1rem); letter-spacing:-.025em; }}
    h2 {{ margin:0; font-size:1.1rem; }}
    .muted {{ color:var(--muted); }}
    .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }}
    .header-actions {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px; }}
    .header-actions [data-create] {{ background:#f1f3f4; border-color:#d7dbe0; color:#111; }}
    .header-actions [data-create]:hover {{ background:#e6e9ed; border-color:#c4c9d0; color:#111; }}
    :root[data-theme='dark'] .header-actions [data-create] {{ background:#233247; border-color:#355070; color:#eef4ff; }}
    :root[data-theme='dark'] .header-actions [data-create]:hover {{ background:#2b3d57; border-color:#49698f; color:#fff; }}
    .btn {{ display:inline-flex; align-items:center; justify-content:center; gap:7px; min-height:38px; border:1px solid var(--line); background:var(--panel-strong); color:var(--text); padding:8px 12px; border-radius:10px; cursor:pointer; font-weight:650; text-decoration:none; }}
    .btn:hover {{ border-color:color-mix(in srgb,var(--accent) 45%,var(--line)); background:var(--chip); }}
    .btn.primary {{ background:#0b57d0; border-color:#0b57d0; color:#fff; }}
    .btn.primary:hover {{ background:#0842a0; border-color:#0842a0; }}
    .btn:disabled {{ opacity:.55; cursor:not-allowed; }}
    .icon {{ width:16px; height:16px; fill:none; stroke:currentColor; stroke-linecap:round; stroke-linejoin:round; stroke-width:1.9; flex:0 0 auto; }}
    .toolbar {{ display:flex; align-items:center; justify-content:space-between; gap:12px; padding:11px; background:var(--panel); border:1px solid var(--line); border-radius:14px; margin-bottom:16px; }}
    .filters {{ display:flex; gap:8px; flex:1; }}
    .search-wrap {{ position:relative; flex:1; max-width:420px; }}
    .search-wrap .icon {{ position:absolute; left:10px; top:50%; transform:translateY(-50%); color:var(--muted); pointer-events:none; }}
    .input {{ width:100%; color:var(--text); background:var(--panel-strong); border:1px solid var(--line); border-radius:9px; padding:9px 10px; }}
    .search-wrap .input {{ padding-left:34px; }}
    select.input {{ width:auto; min-width:135px; }}
    .summary-pills {{ display:flex; gap:7px; flex-wrap:wrap; justify-content:flex-end; }}
    .summary-pill {{ border:1px solid var(--line); background:var(--chip); color:var(--muted); border-radius:999px; padding:5px 9px; font-size:.8rem; font-weight:650; }}
    .summary-pill.active {{ color:var(--accent); }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); gap:14px; }}
    .run-card {{ display:flex; flex-direction:column; min-height:250px; background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:15px; box-shadow:var(--shadow); transition:border-color .15s,transform .15s,box-shadow .15s; }}
    .run-card:hover {{ border-color:color-mix(in srgb,var(--accent) 30%,var(--line)); transform:translateY(-1px); box-shadow:0 5px 18px rgba(0,0,0,.07); }}
    .run-card.highlight {{ border-color:var(--accent); box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 16%,transparent); }}
    .run-card-top {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; }}
    .run-title {{ font-size:1.06rem; font-weight:750; color:var(--text); line-height:1.25; }}
    .run-title:hover {{ color:var(--accent); }}
    .run-id {{ margin-top:3px; color:var(--muted); font-size:.79rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:220px; }}
    .run-meta {{ display:flex; flex-wrap:wrap; gap:8px; color:var(--muted); font-size:.88rem; margin-top:12px; }}
    .run-meta span {{ background:var(--chip); border:1px solid var(--line); padding:4px 8px; border-radius:999px; }}
    .task-preview {{ color:var(--text); margin:18px 0 12px; white-space:pre-wrap; overflow:hidden; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; line-height:1.4; }}
    .card-footer {{ display:flex; margin-top:auto; padding-top:8px; }}
    .open-link {{ display:inline-flex; align-items:center; gap:5px; font-weight:700; }}
    .status {{ flex:0 0 auto; border-radius:999px; padding:4px 9px; font-size:.78rem; text-transform:capitalize; border:1px solid var(--line); background:var(--chip); }}
    .status.succeeded {{ background:#e6f4ea; border-color:#b7dfc2; color:var(--good); }}
    .status.failed {{ background:#fce8e6; border-color:#f3b6b0; color:var(--bad); }}
    .status.running {{ background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }}
    .status.cancelled {{ background:#fef7e0; border-color:#f6d58f; color:var(--warn); }}
    .status.queued,.status.starting {{ background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }}
    .status.cancelling {{ background:#fef7e0; border-color:#f6d58f; color:var(--warn); }}
    :root[data-theme='dark'] .status.succeeded {{ background:#173a25; border-color:#315c3d; }}
    :root[data-theme='dark'] .status.failed {{ background:#44201f; border-color:#6e3733; }}
    :root[data-theme='dark'] .status.running,:root[data-theme='dark'] .status.queued,:root[data-theme='dark'] .status.starting {{ background:#1d3557; border-color:#355070; }}
    :root[data-theme='dark'] .status.cancelled,:root[data-theme='dark'] .status.cancelling {{ background:#3d3218; border-color:#6b5626; }}
    .empty {{ grid-column:1/-1; text-align:center; background:var(--panel); border:1px dashed var(--line); border-radius:16px; padding:44px 24px; color:var(--muted); }}
    .empty h2 {{ color:var(--text); margin:8px 0; }}
    .empty p {{ max-width:540px; margin:0 auto 18px; line-height:1.5; }}
    .empty-icon {{ color:var(--accent); font-size:1.7rem; }}
    .empty-actions {{ display:flex; justify-content:center; flex-wrap:wrap; gap:8px; }}
    code {{ background:var(--chip); padding:2px 5px; border-radius:5px; color:var(--accent); }}
    .modal {{ position:fixed; inset:0; z-index:20; display:flex; align-items:center; justify-content:center; padding:20px; background:rgba(0,0,0,.48); }}
    .modal[hidden] {{ display:none; }}
    .modal-card {{ width:min(820px,100%); max-height:min(90vh,900px); display:flex; flex-direction:column; background:var(--panel-strong); border:1px solid var(--line); border-radius:18px; box-shadow:0 20px 60px rgba(0,0,0,.25); }}
    .modal-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:16px; padding:18px 20px 14px; border-bottom:1px solid var(--line); }}
    .modal-head p {{ margin:5px 0 0; color:var(--muted); font-size:.9rem; }}
    .icon-btn {{ border:0; background:transparent; color:var(--muted); border-radius:8px; padding:6px; cursor:pointer; }}
    .icon-btn:hover {{ background:var(--chip); color:var(--text); }}
    .modal-body {{ padding:18px 20px; overflow:auto; }}
    .field {{ display:grid; gap:6px; margin-bottom:16px; }}
    .field label {{ font-size:.86rem; font-weight:700; }}
    .field-help {{ color:var(--muted); font-size:.8rem; line-height:1.4; }}
    textarea.input {{ resize:vertical; line-height:1.45; }}
    #environmentYaml {{ min-height:250px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.86rem; tab-size:2; }}
    #environmentPrompt {{ min-height:110px; }}
    .replace-row {{ display:flex; align-items:flex-start; gap:8px; color:var(--muted); font-size:.85rem; line-height:1.4; }}
    .replace-row input {{ margin-top:3px; }}
    .form-message {{ min-height:22px; margin-top:10px; font-size:.88rem; }}
    .form-message.error {{ color:var(--bad); }} .form-message.success {{ color:var(--good); }}
    .modal-foot {{ display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px 20px; border-top:1px solid var(--line); }}
    .modal-foot-actions {{ display:flex; gap:8px; }}
    @media (max-width:820px) {{ .top {{ flex-direction:column; }} .header-actions {{ width:100%; justify-content:flex-start; }} .header-actions .btn {{ flex:1; }} }}
    @media (max-width:720px) {{
      .page {{ padding:22px 16px 36px; }} .toolbar {{ align-items:stretch; flex-direction:column; }} .filters {{ flex-direction:column; }}
      .search-wrap {{ max-width:none; }} select.input {{ width:100%; }} .summary-pills {{ justify-content:flex-start; }}
      .grid {{ grid-template-columns:minmax(0,1fr); }} .run-card {{ min-height:225px; }} .run-id {{ max-width:190px; }}
      .modal {{ padding:0; align-items:stretch; }} .modal-card {{ width:100%; max-height:100vh; border-radius:0; }}
      .modal-foot {{ align-items:stretch; flex-direction:column; }} .modal-foot-actions {{ width:100%; }} .modal-foot-actions .btn {{ flex:1; }}
    }}
  </style>
</head>
<body>
  <main class='page'>
    <div class='top'>
      <div>
        <div class='eyebrow'><span>URSA Dashboard</span><span aria-hidden='true'>/</span><span>Environments</span></div>
        <h1>Environment Runs</h1>
        <div class='muted'>Create and replay agent-team and symposium work for group <span class='mono'>{_escape(dashboard_group)}</span>.</div>
      </div>
      <div class='header-actions'>
        <a class='btn' href='/ui'><svg class='icon' viewBox='0 0 24 24' aria-hidden='true'><path d='m15 18-6-6 6-6'/></svg>Dashboard</a>
        <button class='btn' data-create='agent_symposium'>New symposium</button>
        <button class='btn' data-create='agent_team'>New team</button>
      </div>
    </div>
    <section class='toolbar' aria-label='Run filters'>
      <div class='filters'>
        <div class='search-wrap'><svg class='icon' viewBox='0 0 24 24' aria-hidden='true'><circle cx='11' cy='11' r='7'/><path d='m20 20-4-4'/></svg><input class='input' id='runSearch' type='search' placeholder='Search runs or tasks…' aria-label='Search environment runs' /></div>
        <select class='input' id='statusFilter' aria-label='Filter by status'><option value=''>All statuses</option><option value='queued'>Queued</option><option value='starting'>Starting</option><option value='running'>Running</option><option value='succeeded'>Succeeded</option><option value='failed'>Failed</option><option value='cancelled'>Cancelled</option></select>
        <select class='input' id='typeFilter' aria-label='Filter by environment type'><option value=''>All types</option><option value='team'>Teams</option><option value='symposium'>Symposia</option></select>
      </div>
      <div class='summary-pills'><span class='summary-pill'>{run_count} run{"s" if run_count != 1 else ""}</span><span class='summary-pill active' id='activeCount'>{active_count} active</span></div>
    </section>
    <section class='grid'>{body}</section>
  </main>
  <div class='modal' id='environmentModal' hidden>
    <section class='modal-card' role='dialog' aria-modal='true' aria-labelledby='environmentModalTitle'>
      <header class='modal-head'><div><h2 id='environmentModalTitle'>New team</h2><p id='environmentModalCopy'>Configure the team and the task it should complete.</p></div><button class='icon-btn' id='closeEnvironmentModal' type='button' aria-label='Close dialog'><svg class='icon' viewBox='0 0 24 24' aria-hidden='true'><path d='m6 6 12 12M18 6 6 18'/></svg></button></header>
      <div class='modal-body'>
        <div class='field'><label for='environmentYaml'>Environment YAML</label><div class='field-help'>Use built-in URSA agent classes. The dashboard group is applied automatically.</div><textarea class='input' id='environmentYaml' spellcheck='false'></textarea></div>
        <div class='field'><label for='environmentPrompt'>Task prompt</label><textarea class='input' id='environmentPrompt' placeholder='Describe the problem, desired output, constraints, and success criteria.'></textarea></div>
        <label class='replace-row'><input id='replaceExisting' type='checkbox' />Replace an existing saved definition with the same name. Existing run history is not changed.</label>
        <div class='form-message' id='environmentFormMessage' role='status' aria-live='polite'></div>
      </div>
      <footer class='modal-foot'><span class='field-help'>The run will appear here immediately after launch.</span><div class='modal-foot-actions'><button class='btn' id='validateEnvironment' type='button'>Validate</button><button class='btn primary' id='launchEnvironment' type='button'>Launch team</button></div></footer>
    </section>
  </div>
  <script>
  (() => {{
    const TEAM_YAML = {team_json};
    const SYMPOSIUM_YAML = {symposium_json};
    const modal = document.getElementById('environmentModal');
    const yamlInput = document.getElementById('environmentYaml');
    const promptInput = document.getElementById('environmentPrompt');
    const message = document.getElementById('environmentFormMessage');
    const validateBtn = document.getElementById('validateEnvironment');
    const launchBtn = document.getElementById('launchEnvironment');
    let environmentType = 'agent_team';
    let priorFocus = null;

    function esc(value) {{ return String(value == null ? '' : value); }}
    function setMessage(text, kind='') {{ message.textContent = text || ''; message.className = 'form-message' + (kind ? ' ' + kind : ''); }}
    async function api(path, payload) {{
      const response = await fetch(path, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(payload)}});
      let data = {{}}; try {{ data = await response.json(); }} catch (_) {{}}
      if (!response.ok) {{ const detail = Array.isArray(data.detail) ? data.detail.map(item => item.msg || String(item)).join('; ') : data.detail; throw new Error(detail || `Request failed (${{response.status}})`); }}
      return data;
    }}
    function payload() {{ return {{environment_type:environmentType, config_yaml:yamlInput.value, prompt:promptInput.value, replace_existing:document.getElementById('replaceExisting').checked}}; }}
    function openModal(type) {{
      environmentType = type; priorFocus = document.activeElement;
      const symposium = type === 'agent_symposium';
      document.getElementById('environmentModalTitle').textContent = symposium ? 'New symposium' : 'New team';
      document.getElementById('environmentModalCopy').textContent = symposium ? 'Configure independent participants, peer review, and the task.' : 'Configure the team, its members, and the task it should complete.';
      launchBtn.textContent = symposium ? 'Launch symposium' : 'Launch team';
      yamlInput.value = symposium ? SYMPOSIUM_YAML : TEAM_YAML;
      promptInput.value = ''; document.getElementById('replaceExisting').checked = false; setMessage('');
      modal.hidden = false; document.body.style.overflow = 'hidden'; yamlInput.focus();
    }}
    function closeModal() {{ modal.hidden = true; document.body.style.overflow = ''; setMessage(''); if (priorFocus) priorFocus.focus(); }}
    document.querySelectorAll('[data-create]').forEach(button => button.addEventListener('click', () => openModal(button.dataset.create)));
    document.getElementById('closeEnvironmentModal').addEventListener('click', closeModal);
    modal.addEventListener('click', event => {{ if (event.target === modal) closeModal(); }});
    document.addEventListener('keydown', event => {{ if (event.key === 'Escape' && !modal.hidden) closeModal(); }});
    validateBtn.addEventListener('click', async () => {{
      validateBtn.disabled = true; setMessage('Validating…');
      try {{ const result = await api('/environment-runs/validate', payload()); setMessage(`${{result.environment_name}} is valid and ready to launch.`, 'success'); }}
      catch (error) {{ setMessage(error.message, 'error'); }} finally {{ validateBtn.disabled = false; }}
    }});
    launchBtn.addEventListener('click', async () => {{
      if (!promptInput.value.trim()) {{ setMessage('Enter a task prompt before launching.', 'error'); promptInput.focus(); return; }}
      validateBtn.disabled = true; launchBtn.disabled = true; setMessage('Creating and queueing the environment…');
      try {{ const result = await api('/environment-runs', payload()); window.location.href = `/ui/environment-runs?launched=${{encodeURIComponent(result.run_id)}}`; }}
      catch (error) {{ setMessage(error.message, 'error'); validateBtn.disabled = false; launchBtn.disabled = false; }}
    }});
    function filterRuns() {{
      const query = document.getElementById('runSearch').value.trim().toLowerCase();
      const status = document.getElementById('statusFilter').value;
      const type = document.getElementById('typeFilter').value;
      document.querySelectorAll('.run-card').forEach(card => {{ const matches = (!query || card.textContent.toLowerCase().includes(query)) && (!status || card.dataset.status === status) && (!type || card.dataset.type === type); card.hidden = !matches; }});
    }}
    ['runSearch','statusFilter','typeFilter'].forEach(id => document.getElementById(id).addEventListener('input', filterRuns));
    document.querySelectorAll('time[datetime]').forEach(time => {{ const date = new Date(time.dateTime); if (!Number.isNaN(date.valueOf())) {{ time.textContent = new Intl.DateTimeFormat(undefined, {{dateStyle:'medium',timeStyle:'short'}}).format(date); time.title = date.toISOString(); }} }});
    const launched = new URLSearchParams(window.location.search).get('launched');
    if (launched) {{ const card = document.querySelector(`[data-run-id="${{CSS.escape(launched)}}"]`); if (card) card.classList.add('highlight'); history.replaceState(null,'','/ui/environment-runs'); }}
    function applyTheme(theme) {{ let value = theme; if (theme === 'system') value = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'; document.documentElement.dataset.theme = value || 'light'; }}
    fetch('/settings').then(response => response.ok ? response.json() : null).then(data => applyTheme(data?.settings?.ui?.theme || 'system')).catch(() => applyTheme('system'));
    async function refreshStatuses() {{
      try {{ const response = await fetch('/environment-runs'); if (!response.ok) return; const data = await response.json(); let active = 0; for (const run of data.runs || []) {{ const card = document.querySelector(`[data-run-id="${{CSS.escape(run.run_id)}}"]`); if (!card) continue; const value = String(run.status || 'unknown').toLowerCase(); card.dataset.status = value; const badge = card.querySelector('.status'); badge.className = 'status ' + value; badge.textContent = value; if (['queued','starting','running','cancelling'].includes(value)) active++; }} document.getElementById('activeCount').textContent = active + ' active'; filterRuns(); }} catch (_) {{}}
    }}
    if ({str(active_count > 0).lower()}) setInterval(refreshStatuses, 3000);
  }})();
  </script>
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
    :root { color-scheme: light; --bg:#ffffff; --panel:rgba(250,250,250,.94); --panelSolid:#fafafa; --line:#e2e2e2; --text:#111; --muted:#666; --accent:#0b57d0; --accent2:#5f4b8b; --good:#188038; --bad:#b3261e; --warn:#b06000; --chip:#f5f7fb; --code:#f7f7f7; }
    :root[data-theme='dark'] { color-scheme:dark; --bg:#111418; --panel:rgba(28,32,38,.94); --panelSolid:#1c2026; --line:#3a404a; --text:#eceff4; --muted:#aab3bf; --accent:#8ab4ff; --accent2:#c4b5fd; --good:#81c995; --bad:#f28b82; --warn:#fdd663; --chip:#252b33; --code:#171b20; }
    * { box-sizing:border-box; }
    body { font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; margin:0; background:var(--bg); color:var(--text); }
    a { color:var(--accent); text-decoration:none; } a:hover { text-decoration:underline; }
    button,input { background:var(--panelSolid); color:var(--text); border:1px solid var(--line); border-radius:9px; padding:8px 10px; font:inherit; }
    button { cursor:pointer; font-weight:650; } button:hover { border-color:var(--accent); background:var(--chip); }
    button.danger { color:var(--bad); border-color:color-mix(in srgb,var(--bad) 55%,var(--line)); }
    h1 { margin:4px 0 5px; font-size:1.45rem; } h2 { margin:0 0 12px; font-size:1.05rem; } h3 { margin:14px 0 8px; font-size:.92rem; color:#333; }
    .page { min-height:100vh; display:flex; flex-direction:column; }
    .hero { padding:18px 22px; border-bottom:1px solid var(--line); background:var(--panel); }
    .hero-row { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }
    .hero-actions { display:flex; align-items:center; gap:8px; }
    .muted { color:var(--muted); } .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    .status { border-radius:999px; padding:4px 9px; font-size:.8rem; text-transform:capitalize; border:1px solid var(--line); background:#f1f3f4; display:inline-flex; align-items:center; gap:6px; }
    .status.succeeded { background:#e6f4ea; border-color:#b7dfc2; color:var(--good); }
    .status.failed { background:#fce8e6; border-color:#f3b6b0; color:var(--bad); }
    .status.running { background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }
    .status.cancelled { background:#fef7e0; border-color:#f6d58f; color:var(--warn); }
    .status.queued,.status.starting { background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }
    .status.cancelling { background:#fef7e0; border-color:#f6d58f; color:var(--warn); }
    .summary { display:grid; grid-template-columns:minmax(340px,2fr) repeat(4,minmax(120px,1fr)); gap:10px; margin-top:14px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:14px; padding:11px; min-width:0; }
    .metric .label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
    .metric .value { font-weight:750; margin-top:4px; overflow:hidden; text-overflow:ellipsis; }
    .metric.task .value { white-space:pre-wrap; max-height:4.2em; font-weight:600; color:#333; }
    .layout { display:grid; grid-template-columns:minmax(390px,34vw) minmax(480px,1fr) minmax(330px,24vw); gap:0; flex:1; min-height:0; }
    .panel { padding:16px; overflow:auto; border-right:1px solid var(--line); min-height:0; } .panel.right { border-right:0; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
    #graph { height:420px; border-radius:14px; border:1px solid var(--line); background:#fff; overflow:hidden; }
    .graph-note { color:var(--muted); font-size:.83rem; margin-top:8px; }
    .search { width:100%; margin-bottom:10px; }
    .timeline-card { background:#fff; border:1px solid var(--line); border-left:4px solid #d0d7de; border-radius:14px; padding:11px; margin-bottom:9px; cursor:pointer; }
    .timeline-card:hover,.timeline-card.selected { border-color:var(--accent); border-left-color:var(--accent); }
    .timeline-card.failed { border-left-color:var(--bad); } .timeline-card.completed { border-left-color:var(--good); } .timeline-card.active { border-left-color:var(--accent); }
    .timeline-top { display:flex; justify-content:space-between; gap:10px; align-items:flex-start; }
    .timeline-title { font-weight:750; } .timeline-msg { color:#444; margin-top:4px; white-space:pre-wrap; }
    .timeline-meta { display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
    .chip { display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 8px; background:var(--chip); color:#444; font-size:.82rem; }
    .chip.good { border-color:#b7dfc2; color:var(--good); } .chip.bad { border-color:#f3b6b0; color:var(--bad); } .chip.active { border-color:#b8cdf7; color:var(--accent); } .chip.warn { border-color:#f6d58f; color:var(--warn); }
    .event-hero { background:#fff; border:1px solid var(--line); border-radius:18px; padding:18px; }
    .event-title-row { display:flex; justify-content:space-between; gap:14px; align-items:flex-start; }
    .event-title { font-size:1.35rem; font-weight:800; margin:0 0 4px; }
    .event-message { color:var(--text); margin:12px 0; font-size:1.02rem; line-height:1.45; }
    .content-block { background:var(--code); border:1px solid var(--line); border-radius:12px; padding:12px; margin-top:10px; }
    .content-block pre { margin:0; white-space:pre-wrap; word-break:break-word; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88rem; line-height:1.45; }
    .content-block .md { line-height:1.5; }
    .content-block .md p { margin:.45rem 0; }
    .content-block .md h1,.content-block .md h2,.content-block .md h3 { margin:.8rem 0 .35rem; color:var(--text); }
    .content-block .md ul,.content-block .md ol { margin:.45rem 0 .45rem 1.35rem; padding:0; }
    .content-block .md li { margin:.2rem 0; }
    .content-block .md code { background:#eef2f7; border-radius:4px; padding:1px 4px; }
    .content-block .md pre.codeblock { background:#f6f8fa; border:1px solid var(--line); border-radius:8px; padding:10px; overflow:auto; }
    .content-label { color:var(--muted); font-size:.75rem; text-transform:uppercase; letter-spacing:.05em; margin-bottom:6px; }
    .final-content,.task-content,.path-content { max-height:340px; overflow:auto; white-space:pre-wrap; line-height:1.45; }
    .path-content { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.88rem; }
    .empty { color:var(--muted); padding:14px; border:1px dashed var(--line); border-radius:12px; background:#fff; }
    .fallback-graph { display:grid; gap:8px; padding:10px; }
    .fallback-node { display:flex; justify-content:space-between; align-items:center; gap:8px; background:var(--chip); border:1px solid var(--line); border-radius:12px; padding:8px 10px; cursor:pointer; }
    .fallback-node.active { border-color:var(--accent); box-shadow:0 0 0 1px rgba(11,87,208,.22) inset; }
    .fallback-node.completed { border-color:#b7dfc2; } .fallback-node.failed { border-color:#f3b6b0; }
    .small-link { font-size:.86rem; color:var(--muted); }
    :root[data-theme='dark'] .metric,:root[data-theme='dark'] .timeline-card,:root[data-theme='dark'] .event-hero,:root[data-theme='dark'] .empty { background:var(--panelSolid); }
    :root[data-theme='dark'] .timeline-msg,:root[data-theme='dark'] h3,:root[data-theme='dark'] .metric.task .value { color:var(--muted); }
    :root[data-theme='dark'] .content-block .md code,:root[data-theme='dark'] .content-block .md pre.codeblock { background:var(--chip); color:var(--text); }
    :root[data-theme='dark'] .fallback-node { background:var(--chip); }
    :root[data-theme='dark'] .status.succeeded { background:#173a25; border-color:#315c3d; }
    :root[data-theme='dark'] .status.failed { background:#44201f; border-color:#6e3733; }
    :root[data-theme='dark'] .status.running,:root[data-theme='dark'] .status.queued,:root[data-theme='dark'] .status.starting { background:#1d3557; border-color:#355070; }
    :root[data-theme='dark'] .status.cancelled,:root[data-theme='dark'] .status.cancelling { background:#3d3218; border-color:#6b5626; }
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
      <div class='hero-actions'><div id='statusBadge'></div><button class='danger' id='cancelEnvironmentRun' type='button' hidden>Cancel run</button></div>
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
function sanitizeUrl(u) {
  u = String(u || '').trim();
  if (u.startsWith('http://') || u.startsWith('https://') || u.startsWith('/')) return u;
  return '#';
}
function mdToHtml(md) {
  md = String(md ?? '').replace(/\r\n/g, '\n');
  const src = esc(md);
  const out = [];
  const re = /```([a-zA-Z0-9_+-]+)?\n([\s\S]*?)```/g;
  let last = 0;
  let m;
  while ((m = re.exec(src)) !== null) {
    out.push({type:'text', text: src.slice(last, m.index)});
    out.push({type:'code', lang: (m[1] || '').trim(), code: m[2] || ''});
    last = re.lastIndex;
  }
  out.push({type:'text', text: src.slice(last)});
  function renderText(t) {
    t = t.replace(/`([^`]+)`/g, '<code>$1</code>');
    t = t.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    t = t.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    t = t.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (all, label, url) => `<a href="${esc(sanitizeUrl(url))}" target="_blank" rel="noreferrer">${label}</a>`);
    t = t.replace(/^###\s+(.+)$/gm, '<h3>$1</h3>');
    t = t.replace(/^##\s+(.+)$/gm, '<h2>$1</h2>');
    t = t.replace(/^#\s+(.+)$/gm, '<h1>$1</h1>');
    const blocks = t.split(/\n{2,}/).map(b => b.trim()).filter(Boolean);
    return blocks.map(b => {
      b = b.replace(/\n/g, '<br>');
      if (/^<h[1-3]>/.test(b)) return b;
      return `<p>${b}</p>`;
    }).join('\n');
  }
  const html = out.map(part => {
    if (part.type === 'code') {
      const lang = part.lang ? ` data-lang="${esc(part.lang)}"` : '';
      return `<pre class="codeblock"><code${lang}>${part.code}</code></pre>`;
    }
    return renderText(part.text);
  }).join('\n');
  return `<div class="md">${html || '<p class="muted">(empty)</p>'}</div>`;
}
const payload = (e) => e && e.payload && typeof e.payload === 'object' ? e.payload : {};
const text = (v) => typeof v === 'string' ? v : JSON.stringify(v, null, 2);
function statusClass(s) { const v = String(s || 'unknown').toLowerCase(); return ['succeeded','failed','running','cancelled','queued','starting','cancelling'].includes(v) ? v : 'unknown'; }
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
  const cancel = $('cancelEnvironmentRun'); cancel.hidden = manifest.launch_source !== 'dashboard' || !['queued','starting','running','cancelling'].includes(String(status).toLowerCase()); cancel.disabled = String(status).toLowerCase() === 'cancelling';
  $('summary').innerHTML = [
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
    { selector:'node', style:{ 'label':'data(label)', 'font-size':12, 'color':'#111', 'text-outline-width':2, 'text-outline-color':'#fff', 'background-color':'#d2e3fc', 'border-width':1, 'border-color':'#8ab4f8', 'width':50, 'height':50 } },
    { selector:'.node-environment', style:{ 'background-color':'#d7c7ff', 'shape':'round-rectangle', 'width':74, 'height':42 } },
    { selector:'edge', style:{ 'curve-style':'bezier', 'target-arrow-shape':'triangle', 'line-color':'#9aa0a6', 'target-arrow-color':'#9aa0a6', 'label':'data(label)', 'font-size':9, 'color':'#666', 'text-rotation':'autorotate', 'width':2 } },
    { selector:'edge[source = target]', style:{ 'curve-style':'bezier', 'loop-direction':'45deg', 'loop-sweep':'70deg' } },
    { selector:'node.active', style:{ 'background-color':'#8ab4f8', 'border-color':'#0b57d0', 'border-width':3 } },
    { selector:'edge.active', style:{ 'line-color':'#0b57d0', 'target-arrow-color':'#0b57d0', 'width':4 } },
    { selector:'node.completed', style:{ 'background-color':'#81c995', 'border-color':'#188038' } },
    { selector:'edge.completed', style:{ 'line-color':'#188038', 'target-arrow-color':'#188038', 'width':3 } },
    { selector:'node.failed', style:{ 'background-color':'#f28b82', 'border-color':'#b3261e' } },
    { selector:'edge.failed', style:{ 'line-color':'#b3261e', 'target-arrow-color':'#b3261e', 'width':4 } },
    { selector:'.selected', style:{ 'border-width':4, 'border-color':'#f9ab00' } }
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
  const q = $('timelineSearch').value.trim().toLowerCase();
  return events.map((e,i) => [e,i]).filter(([e,i]) => eventMatchesParticipant(e, selectedParticipant, i)).filter(([e,i]) => !q || eventSearchText(e, i).includes(q)).reverse();
}
function renderTimeline() {
  const rows = timelineEvents(); $('scrub').max = Math.max(0, events.length - 1); $('scrub').value = selected < 0 ? Math.max(0, events.length - 1) : selected;
  $('timeline').innerHTML = rows.length ? rows.map(([e,i]) => {
    const cls = eventLevel(e); const st = sourceTargetText(e, i); const p = payload(e); const dur = p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : (p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : '');
    return `<article class='timeline-card ${cls} ${i === selected ? 'selected' : ''}' data-index='${i}'><div class='timeline-top'><div><div class='timeline-title'>${esc(readableType(e.event_type))}</div><div class='timeline-msg'>${esc(e.message || p.message || '')}</div></div><span class='chip'>#${esc(e.seq ?? i + 1)}</span></div><div class='timeline-meta'>${phaseLabel(e) ? `<span class='chip'>${esc(phaseLabel(e))}</span>` : ''}${st ? `<span class='chip'>${esc(st)}</span>` : ''}${dur ? `<span class='chip'>${esc(dur)}</span>` : ''}</div></article>`;
  }).join('') : `<div class='empty'>No visible events match the current playback state or filter.</div>`;
  $('timeline').querySelectorAll('.timeline-card').forEach(el => el.onclick = () => selectEvent(Number(el.dataset.index)));
}
function block(label, value, options = {}) {
  if (value == null || value === '') return '';
  const raw = text(value);
  const renderMarkdown = options.markdown !== false && typeof value === 'string';
  const body = renderMarkdown ? mdToHtml(raw) : `<pre>${esc(raw)}</pre>`;
  return `<div class='content-block'><div class='content-label'>${esc(label)}</div>${body}</div>`;
}
function renderCurrentEvent() {
  const e = selected >= 0 ? events[selected] : events[events.length - 1]; if (!e) { $('currentEvent').innerHTML = `<div class='empty'>Waiting for events…</div>`; return; }
  const idx = events.indexOf(e); const p = payload(e); const owner = inferredOwner(e, idx); const tool = toolName(e); const cls = eventLevel(e);
  const chips = [phaseLabel(e), sourceTargetText(e, idx), p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : '', p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : ''].filter(Boolean);
  let sections = '';
  if (tool && !owner) sections += block('Assignment', 'This tool event did not record the member that invoked it. The UI is not assigning it to a graph participant to avoid misleading attribution.');
  sections += block('Task / instruction', p.task || p.prompt);
  sections += block('Tool query / command', p.query || p.command || p.input, {markdown:false});
  sections += block('Result', p.result || p.output || p.final);
  sections += block('File or path', p.path || p.filename, {markdown:false});
  sections += block('Error', p.error || p.error_type, {markdown:false});
  sections += block('Safety rationale', p.reason);
  if (!sections) sections = block('Details', p, {markdown:false});
  const message = e.message || p.message || '';
  $('currentEvent').innerHTML = `<div class='event-title-row'><div><div class='event-title'>${esc(readableType(e.event_type))}</div><div class='muted'>Event #${esc(e.seq ?? idx + 1)}${owner ? ` &middot; ${esc(participantNameById(owner))}` : ''}${tool ? ` &middot; ${esc(tool)}` : ''}</div></div><span class='chip ${cls === 'failed' ? 'bad' : cls === 'completed' ? 'good' : cls === 'active' ? 'active' : ''}'>${esc(cls || 'event')}</span></div>${message ? `<div class='event-message'>${mdToHtml(message)}</div>` : ''}<div class='timeline-meta'>${chips.map(c => `<span class='chip'>${esc(c)}</span>`).join('')}</div>${sections}`;
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
  $('cancelEnvironmentRun').onclick = async () => { const button = $('cancelEnvironmentRun'); button.disabled = true; button.textContent = 'Cancelling…'; try { const response = await fetch('/environment-runs/'+encodeURIComponent(runId)+'/cancel', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'user_request'})}); if (!response.ok) throw new Error('Cancellation request failed'); Object.assign(manifest, await response.json()); updateSummary(); } catch (error) { button.disabled = false; button.textContent = 'Cancel run'; console.error(error); } };
}
async function init() {
  fetch('/settings').then(r => r.ok ? r.json() : null).then(data => { let theme = data?.settings?.ui?.theme || 'system'; if (theme === 'system') theme = matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light'; document.documentElement.dataset.theme = theme; }).catch(() => {});
  setupControls(); updateSummary();
  const detail = await fetch('/environment-runs/'+encodeURIComponent(runId)).then(r => r.json()); Object.assign(manifest, detail); updateSummary();
  const data = await fetch('/environment-runs/'+encodeURIComponent(runId)+'/events').then(r => r.json()); (data.events || []).forEach(e => events.push(e));
  selected = events.length ? events.length - 1 : -1; topology = extractTopology(); initializeGraph(); renderAll();
  const last = events[events.length - 1]; let src = null;
  if (!['succeeded','failed','cancelled'].includes(String(manifest.status).toLowerCase())) {
    src = new EventSource('/environment-runs/'+encodeURIComponent(runId)+'/stream?after_seq='+(last ? last.seq : 0));
    src.onmessage = (ev) => addEvent(JSON.parse(ev.data));
    src.onerror = async () => { try { const latest = await fetch('/environment-runs/'+encodeURIComponent(runId)).then(r => r.json()); Object.assign(manifest, latest); updateSummary(); if (['succeeded','failed','cancelled'].includes(String(manifest.status).toLowerCase())) src.close(); } catch (_) {} };
  }
  const refresh = setInterval(async () => { try { const latest = await fetch('/environment-runs/'+encodeURIComponent(runId)).then(r => r.json()); Object.assign(manifest, latest); updateSummary(); if (['succeeded','failed','cancelled'].includes(String(manifest.status).toLowerCase())) clearInterval(refresh); } catch (_) {} }, 2500);
}
init().catch(err => { console.error(err); $('currentEvent').innerHTML = `<div class='empty'>Failed to initialize run view: ${esc(err.message)}</div>`; });
</script>
</body>
</html>
"""
