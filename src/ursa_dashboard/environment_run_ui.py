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
    elo_starter_yaml: str,
) -> str:
    sorted_runs = sorted(runs, key=_run_sort_key, reverse=True)
    cards = []
    for run in sorted_runs:
        run_id_raw = str(run.get("run_id", ""))
        run_id = _escape(run_id_raw)
        name = _escape(run.get("environment_name", "Environment"))
        env_type_raw = str(run.get("environment_type", ""))
        env_type = _escape(env_type_raw)
        if "team" in env_type_raw.lower():
            env_label = "Team"

        elif "symposium" in env_type_raw.lower():
            env_label = "Symposium"

        elif "elo" in env_type_raw.lower():
            env_label = "Elo"

        else:
            env_label = "Environment"
        status = _escape(run.get("status", "unknown"))
        status_class = _status_class(run.get("status"))
        updated = _escape(run.get("updated_at", ""))
        preview = _escape(run.get("task_preview", ""))
        rerun_action = (
            f"<button class='rerun-link' type='button' data-rerun='{run_id}' "
            f"title='Start a new run using {name} as a template'>"
            "<svg class='icon' viewBox='0 0 24 24' aria-hidden='true'>"
            "<rect x='8' y='8' width='11' height='11' rx='2'/><path d='M16 8V7a2 2 0 0 0-2-2H7a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h1'/></svg>"
            "Run again</button>"
            if run.get("can_rerun")
            else ""
        )
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
            f"{rerun_action}"
            "</div>"
            "</article>"
        )
    body = "".join(cards) or (
        "<div class='empty'><div class='empty-icon'>✦</div>"
        "<h2>No environment runs yet</h2>"
        "<p>Create a team, symposium, or Elo environment, or continue using "
        "<code>run_with_visualization</code> from Python.</p>"
        "<div class='empty-actions'><button class='btn primary' data-create='agent_team'>"
        "Create a team</button><button class='btn' data-create='agent_symposium'>"
        "Create a symposium</button>"
        "<button class='btn' data-create='agent_elo'>"
        "Create an Elo environment</button></div></div>"
    )
    team_json = json.dumps(team_starter_yaml)
    symposium_json = json.dumps(symposium_starter_yaml)
    elo_json = json.dumps(elo_starter_yaml)
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
    .card-footer {{ display:flex; align-items:center; justify-content:space-between; gap:10px; margin-top:auto; padding-top:8px; }}
    .open-link {{ display:inline-flex; align-items:center; gap:5px; font-weight:700; }}
    .rerun-link {{ display:inline-flex; align-items:center; gap:5px; border:0; background:transparent; color:var(--muted); padding:5px; border-radius:7px; cursor:pointer; font:inherit; font-size:.84rem; font-weight:700; }}
    .rerun-link:hover {{ color:var(--accent); background:var(--chip); }}
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
        <button class='btn' data-create='agent_elo'>New Elo</button>
      </div>
    </div>
    <section class='toolbar' aria-label='Run filters'>
      <div class='filters'>
        <div class='search-wrap'><svg class='icon' viewBox='0 0 24 24' aria-hidden='true'><circle cx='11' cy='11' r='7'/><path d='m20 20-4-4'/></svg><input class='input' id='runSearch' type='search' placeholder='Search runs or tasks…' aria-label='Search environment runs' /></div>
        <select class='input' id='statusFilter' aria-label='Filter by status'><option value=''>All statuses</option><option value='queued'>Queued</option><option value='starting'>Starting</option><option value='running'>Running</option><option value='succeeded'>Succeeded</option><option value='failed'>Failed</option><option value='cancelled'>Cancelled</option></select>
        <select class='input' id='typeFilter' aria-label='Filter by environment type'><option value=''>All types</option><option value='team'>Teams</option><option value='symposium'>Symposia</option><option value='elo'>Elo</option></select>
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
        <div class='field'><label for='environmentRunId'>Run ID <span class='muted'>(optional)</span></label><div class='field-help'>Give follow-on runs a descriptive unique ID. Leave blank to generate one automatically; the environment name continues to select the shared workspace.</div><input class='input' id='environmentRunId' maxlength='64' placeholder='For example: research-team-follow-up-2' /></div>
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
    const ELO_YAML = {elo_json};
    const modal = document.getElementById('environmentModal');
    const yamlInput = document.getElementById('environmentYaml');
    const runIdInput = document.getElementById('environmentRunId');
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
    function payload() {{ return {{environment_type:environmentType, config_yaml:yamlInput.value, prompt:promptInput.value, run_id:runIdInput.value.trim() || null, replace_existing:document.getElementById('replaceExisting').checked}}; }}
    function openModal(type) {{
      environmentType = type;
      priorFocus = document.activeElement;
      if (modal.hidden) priorFocus = document.activeElement;
    
      const symposium = type === 'agent_symposium';
      const elo = type === 'agent_elo';
      const repeated = !!template?.source_run_id;

      if (repeated) {{
        document.getElementById('environmentModalTitle').textContent = elo 
          ? 'Run Elo again' 
          : (symposium ? 'Run symposium again' : 'Run team again');
      }} else {{
        document.getElementById('environmentModalTitle').textContent = elo 
          ? 'New Elo environment' 
          : (symposium ? 'New symposium' : 'New team');
      }}

      if (repeated) {{
        document.getElementById('environmentModalCopy').textContent = 
          'Review the copied configuration and task before launching an independent new run.';
      }} else if (elo) {{
        document.getElementById('environmentModalCopy').textContent = 
          'Configure evolutionary competitors and the task.';
      }} else if (symposium) {{
        document.getElementById('environmentModalCopy').textContent = 
          'Configure independent participants, peer review, and the task.';
      }} else {{
        document.getElementById('environmentModalCopy').textContent = 
          'Configure the team, its members, and the task it should complete.';
      }}

      launchBtn.textContent = elo 
        ? 'Launch Elo' 
        : (symposium ? 'Launch symposium' : 'Launch team');

      // Set YAML Configuration Input
      yamlInput.value = template && 'config_yaml' in template 
        ? template.config_yaml 
        : (elo ? ELO_YAML : (symposium ? SYMPOSIUM_YAML : TEAM_YAML));

      // Reset Form Fields
      runIdInput.value = '';
      promptInput.value = template?.prompt || '';
      document.getElementById('replaceExisting').checked = false;

      setMessage(repeated && !template.loading 
        ? `Copied from ${{template.source_run_id}}. A fresh Run ID will be generated unless you provide one.` 
        : ''
      );

      modal.hidden = false;
      document.body.style.overflow = 'hidden';
      (repeated ? promptInput : yamlInput).focus();
    }}
    async function openRerun(runId, trigger=null) {{
      const cardRunType = trigger?.closest('.run-card')?.dataset.type;
      const cardType = cardRunType === 'symposium' 
        ? 'agent_symposium' 
        : (cardRunType === 'elo' ? 'agent_elo' : 'agent_team');

      openModal(cardType, {{source_run_id:runId, config_yaml:'', prompt:'', loading:true}});
      validateBtn.disabled = true; 
      launchBtn.disabled = true; 
      setMessage('Loading the previous configuration and full task…');

      try {{
        const response = await fetch(`/environment-runs/${{encodeURIComponent(runId)}}/rerun-template`);
        let data = {{}}; 
        try {{ data = await response.json(); }} catch (_) {{}}
        if (!response.ok) throw new Error(data.detail || `Unable to load run (${{response.status}})`);
        openModal(data.environment_type, data);
      }} catch (error) {{
        setMessage(error.message, 'error');
      }} finally {{
        validateBtn.disabled = false; 
        launchBtn.disabled = false;
      }}
    }}
    function closeModal() {{ modal.hidden = true; document.body.style.overflow = ''; setMessage(''); if (priorFocus) priorFocus.focus(); }}
    document.querySelectorAll('[data-create]').forEach(button => button.addEventListener('click', () => openModal(button.dataset.create)));
    document.querySelectorAll('[data-rerun]').forEach(button => button.addEventListener('click', () => openRerun(button.dataset.rerun, button)));
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
    const params = new URLSearchParams(window.location.search);
    const launched = params.get('launched'); const rerun = params.get('rerun');
    if (launched) {{ const card = document.querySelector(`[data-run-id="${{CSS.escape(launched)}}"]`); if (card) card.classList.add('highlight'); history.replaceState(null,'','/ui/environment-runs'); }}
    if (rerun) {{ openRerun(rerun); history.replaceState(null,'','/ui/environment-runs'); }}
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
    rerun_action = (
        f"<a class='small-link' href='/ui/environment-runs?rerun={safe_run_id}'>"
        "Run again</a>"
        if manifest.get("can_rerun")
        else ""
    )
    return (
        DETAIL_TEMPLATE.replace("__TITLE__", title)
        .replace("__RUN_ID__", safe_run_id)
        .replace("__RUN_ID_JSON__", run_id_json)
        .replace("__MANIFEST_JSON__", manifest_json)
        .replace("__CYTOSCAPE_URL__", cytoscape_url)
        .replace("__RERUN_ACTION__", rerun_action)
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
    .hero { padding:18px 22px 16px; border-bottom:1px solid var(--line); background:var(--panel); }
    .hero-row { display:flex; justify-content:space-between; align-items:flex-start; gap:18px; }
    .hero-actions { display:flex; align-items:center; flex-wrap:wrap; justify-content:flex-end; gap:8px; }
    .muted { color:var(--muted); } .mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
    .status { border-radius:999px; padding:4px 9px; font-size:.8rem; text-transform:capitalize; border:1px solid var(--line); background:#f1f3f4; display:inline-flex; align-items:center; gap:6px; }
    .status.succeeded { background:#e6f4ea; border-color:#b7dfc2; color:var(--good); }
    .status.failed { background:#fce8e6; border-color:#f3b6b0; color:var(--bad); }
    .status.running { background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }
    .status.cancelled { background:#fef7e0; border-color:#f6d58f; color:var(--warn); }
    .status.queued,.status.starting { background:#e8f0fe; border-color:#b8cdf7; color:var(--accent); }
    .status.cancelling { background:#fef7e0; border-color:#f6d58f; color:var(--warn); }
    .summary { display:grid; grid-template-columns:repeat(3,minmax(105px,.55fr)) minmax(360px,2.4fr); gap:10px; margin-top:14px; }
    .metric { background:#fff; border:1px solid var(--line); border-radius:14px; padding:11px; min-width:0; }
    .metric .label { color:var(--muted); font-size:.76rem; text-transform:uppercase; letter-spacing:.05em; }
    .metric .value { font-weight:750; margin-top:4px; min-width:0; }
    .metric.workspace .value { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .workspace-path { min-width:0; white-space:normal; overflow-wrap:anywhere; user-select:text; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; line-height:1.35; }
    .copy-btn { flex:0 0 auto; min-height:30px; padding:5px 8px; color:var(--muted); font-size:.78rem; }
    .task-banner { margin-top:10px; padding:13px 14px 14px; border:1px solid var(--line); border-radius:14px; background:color-mix(in srgb,var(--panelSolid) 88%,transparent); }
    .task-banner-head,.section-title-row,.activity-head { display:flex; align-items:flex-start; justify-content:space-between; gap:14px; }
    .task-kicker { color:var(--accent); font-size:.75rem; font-weight:800; letter-spacing:.065em; text-transform:uppercase; }
    .task-content { max-height:180px; margin-top:7px; overflow:auto; white-space:pre-wrap; line-height:1.48; user-select:text; }
    .layout { display:grid; grid-template-columns:minmax(430px,.95fr) minmax(500px,1.18fr); gap:0; flex:1; min-height:0; }
    .panel { padding:16px; overflow:auto; border-right:1px solid var(--line); min-height:0; }
    .panel.activity { border-right:0; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; margin-bottom:14px; box-shadow:0 1px 2px rgba(0,0,0,.04); }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
    #graph { height:440px; border-radius:14px; border:1px solid var(--line); background:radial-gradient(circle at 50% 38%,#fff 0,#f6f9ff 58%,#eef3fb 100%); overflow:hidden; }
    .graph-note { color:var(--muted); font-size:.83rem; margin-top:8px; }
    .section-copy { color:var(--muted); font-size:.83rem; line-height:1.4; margin:3px 0 10px; }
    .participant-filter { color:var(--accent); font-weight:720; }
    .search { width:100%; margin-bottom:10px; }
    .timeline-card { background:#fff; border:1px solid var(--line); border-left:4px solid #d0d7de; border-radius:14px; padding:11px; margin-bottom:9px; cursor:pointer; }
    .timeline-card:hover,.timeline-card.selected { border-color:var(--accent); border-left-color:var(--accent); }
    .timeline-card.failed { border-left-color:var(--bad); } .timeline-card.completed { border-left-color:var(--good); } .timeline-card.active { border-left-color:var(--accent); }
    .timeline-card.tool { border-left-color:var(--accent2); }
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
    .event-artifacts { display:grid; gap:10px; margin-top:12px; }
    .event-artifact { border:1px solid var(--line); border-radius:13px; overflow:hidden; background:var(--code); }
    .artifact-head { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:9px 11px; border-bottom:1px solid var(--line); background:var(--panelSolid); }
    .artifact-title { font-size:.82rem; font-weight:780; text-transform:capitalize; }
    .artifact-meta { display:flex; flex-wrap:wrap; align-items:center; gap:6px; min-width:0; color:var(--muted); font-size:.76rem; }
    .artifact-path { padding:8px 11px; border-bottom:1px solid var(--line); color:var(--muted); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.76rem; overflow-wrap:anywhere; user-select:text; }
    .artifact-output { margin:0; padding:12px; max-height:430px; overflow:auto; white-space:pre; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; line-height:1.45; background:var(--code); color:var(--text); tab-size:2; }
    .artifact-output.wrap { white-space:pre-wrap; word-break:break-word; }
    .artifact-output.diff .diff-line { display:block; min-height:1.45em; padding:0 5px; margin:0 -5px; }
    .artifact-output.diff .diff-add { color:#176b36; background:rgba(52,168,83,.12); }
    .artifact-output.diff .diff-remove { color:#a52714; background:rgba(234,67,53,.11); }
    .artifact-output.diff .diff-header { color:var(--accent); font-weight:700; background:rgba(11,87,208,.07); }
    .artifact-reference-note { padding:9px 11px; color:var(--muted); font-size:.8rem; line-height:1.4; }
    .artifact-copy { flex:0 0 auto; padding:5px 8px; font-size:.76rem; }
    .empty { color:var(--muted); padding:14px; border:1px dashed var(--line); border-radius:12px; background:#fff; }
    .fallback-graph { display:grid; gap:8px; padding:10px; }
    .fallback-node { display:flex; justify-content:space-between; align-items:center; gap:8px; background:var(--chip); border:2px solid var(--line); border-radius:14px; padding:8px 10px; cursor:pointer; }
    .fallback-node-main { display:flex; align-items:center; gap:9px; min-width:0; }
    .fallback-bear { width:36px; height:36px; flex:0 0 auto; padding:3px; border-radius:50%; background:#fff; }
    .fallback-bear img { width:100%; height:100%; display:block; }
    .fallback-node.active { border-color:var(--accent); box-shadow:0 0 0 1px rgba(11,87,208,.22) inset; }
    .fallback-node.completed { border-color:#b7dfc2; } .fallback-node.failed { border-color:#f3b6b0; }
    .small-link { font-size:.86rem; color:var(--muted); }
    :root[data-theme='dark'] .metric,:root[data-theme='dark'] .timeline-card,:root[data-theme='dark'] .event-hero,:root[data-theme='dark'] .empty { background:var(--panelSolid); }
    :root[data-theme='dark'] #graph { background:radial-gradient(circle at 50% 38%,#252c36 0,#1c222a 58%,#171b20 100%); }
    :root[data-theme='dark'] .timeline-msg,:root[data-theme='dark'] h3,:root[data-theme='dark'] .chip { color:var(--muted); }
    :root[data-theme='dark'] .content-block .md code,:root[data-theme='dark'] .content-block .md pre.codeblock { background:var(--chip); color:var(--text); }
    :root[data-theme='dark'] .artifact-output.diff .diff-add { color:#81c995; background:rgba(129,201,149,.11); }
    :root[data-theme='dark'] .artifact-output.diff .diff-remove { color:#f28b82; background:rgba(242,139,130,.11); }
    :root[data-theme='dark'] .fallback-node { background:var(--chip); }
    :root[data-theme='dark'] .status.succeeded { background:#173a25; border-color:#315c3d; }
    :root[data-theme='dark'] .status.failed { background:#44201f; border-color:#6e3733; }
    :root[data-theme='dark'] .status.running,:root[data-theme='dark'] .status.queued,:root[data-theme='dark'] .status.starting { background:#1d3557; border-color:#355070; }
    :root[data-theme='dark'] .status.cancelled,:root[data-theme='dark'] .status.cancelling { background:#3d3218; border-color:#6b5626; }
    @media (max-width:1180px) { .layout { grid-template-columns:1fr; } .panel { border-right:0; border-bottom:1px solid var(--line); } .summary { grid-template-columns:repeat(3,minmax(100px,1fr)); } .metric.workspace { grid-column:1/-1; } #graph { height:380px; } }
    @media (max-width:720px) { .hero { padding:16px; } .hero-row,.task-banner-head,.section-title-row,.activity-head { align-items:stretch; flex-direction:column; } .hero-actions { justify-content:flex-start; } .summary { grid-template-columns:1fr 1fr; } .metric.workspace { grid-column:1/-1; } .layout { display:block; } .panel { padding:12px; } #graph { height:330px; } }
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
      <div class='hero-actions'>__RERUN_ACTION__<a class='small-link' id='rawEventsLink' href='#'>Raw events</a><div id='statusBadge'></div><button class='danger' id='cancelEnvironmentRun' type='button' hidden>Cancel run</button></div>
    </div>
    <section class='summary' id='summary'></section>
    <section class='task-banner'>
      <div class='task-banner-head'>
        <div><div class='task-kicker'>Original task</div><div class='muted small-link'>The shared objective for this environment run</div></div>
        <button class='copy-btn' id='copyTask' type='button'>Copy task</button>
      </div>
      <div id='task' class='task-content muted'></div>
    </section>
  </header>
  <main class='layout'>
    <section class='panel work'>
      <section class='card'>
        <h2>Environment Graph</h2>
        <div id='graph'></div>
        <div id='graphNote' class='graph-note'>Loading Cytoscape.js graph…</div>
      </section>
      <section class='card'>
        <div class='section-title-row'>
          <div><h2>Work Timeline</h2><div class='section-copy' id='timelineContext'>Milestones and agent-to-agent messages</div></div>
          <button class='copy-btn' id='clearParticipant' type='button' hidden>Show all agents</button>
        </div>
        <div class='toolbar'>
          <button id='live'>Pause live</button>
          <button id='prev'>Older</button>
          <button id='next'>Newer</button>
          <input id='scrub' type='range' min='0' max='0' value='0' />
        </div>
        <input id='timelineSearch' class='search' placeholder='Search milestones and messages…' />
        <div id='timeline'></div>
      </section>
    </section>
    <section class='panel activity'>
      <div class='activity-head'><div><h2>Current Activity</h2><div class='section-copy' id='activityContext'>Latest high-signal event</div></div></div>
      <section class='event-hero' id='currentEvent'>
        <div class='empty'>Waiting for events…</div>
      </section>
    </section>
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
const SCIENTIST_BEAR_IMAGE = 'data:image/png;base64,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';

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
async function copyText(value, button, successLabel) {
  const original = button.textContent;
  try {
    await navigator.clipboard.writeText(String(value || ''));
    button.textContent = successLabel;
  } catch (_) {
    const field = document.createElement('textarea');
    field.value = String(value || ''); field.style.position = 'fixed'; field.style.opacity = '0';
    document.body.appendChild(field); field.select(); document.execCommand('copy'); field.remove();
    button.textContent = successLabel;
  }
  setTimeout(() => { button.textContent = original; }, 1200);
}
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
function eventArtifacts(p) {
  const items = [];
  if (p && p.artifact && typeof p.artifact === 'object') items.push(p.artifact);
  if (p && Array.isArray(p.artifacts)) p.artifacts.forEach(item => { if (item && typeof item === 'object') items.push(item); });
  return items;
}
function artifactTitle(artifact) {
  const metadata = artifact.metadata && typeof artifact.metadata === 'object' ? artifact.metadata : {};
  if (metadata.title) return String(metadata.title);
  const mime = String(artifact.mime_type || 'text/plain').split(';', 1)[0];
  if (mime === 'text/x-diff') return 'Code changes';
  if (mime === 'application/json') return 'JSON output';
  if (mime.startsWith('text/x-')) return 'Generated code';
  return 'Tool output';
}
function renderDiff(value) {
  return String(value || '').split('\n').map(line => {
    let cls = '';
    if (line.startsWith('@@') || line.startsWith('diff ') || line.startsWith('index ')) cls = 'diff-header';
    else if (line.startsWith('+') && !line.startsWith('+++')) cls = 'diff-add';
    else if (line.startsWith('-') && !line.startsWith('---')) cls = 'diff-remove';
    return `<span class='diff-line ${cls}'>${line ? esc(line) : '&nbsp;'}</span>`;
  }).join('');
}
function renderEventArtifact(artifact, index) {
  const metadata = artifact.metadata && typeof artifact.metadata === 'object' ? artifact.metadata : {};
  const mime = String(artifact.mime_type || 'text/plain').split(';', 1)[0];
  const title = artifactTitle(artifact); const path = String(metadata.path || '');
  const isReference = mime === 'application/vnd.ursa.file-reference';
  const raw = isReference ? String(path || artifact.content || '') : text(artifact.content ?? '');
  let body = '';
  if (isReference) {
    body = `<pre class='artifact-output wrap'>${esc(raw)}</pre><div class='artifact-reference-note'>This event contains a file reference only. New write-code events preserve a content snapshot for replay.</div>`;
  } else if (mime === 'text/x-diff') {
    body = `<pre class='artifact-output diff'>${renderDiff(raw)}</pre>`;
  } else if (mime === 'text/markdown') {
    body = `<div class='artifact-output wrap'>${mdToHtml(raw)}</div>`;
  } else if (mime === 'application/json') {
    let formatted = raw; try { formatted = JSON.stringify(typeof artifact.content === 'string' ? JSON.parse(artifact.content) : artifact.content, null, 2); } catch (_) {}
    body = `<pre class='artifact-output'>${esc(formatted)}</pre>`;
  } else {
    body = `<pre class='artifact-output ${mime === 'text/plain' ? 'wrap' : ''}'>${esc(raw || '(no output)')}</pre>`;
  }
  return `<section class='event-artifact'><div class='artifact-head'><div class='artifact-meta'><span class='artifact-title'>${esc(title)}</span><span>${esc(mime)}</span></div><button class='artifact-copy' type='button' data-artifact-copy='${index}'>Copy</button></div>${path && !isReference ? `<div class='artifact-path'>${esc(path)}</div>` : ''}${body}</section>`;
}
function renderEventArtifacts(p) {
  const artifacts = eventArtifacts(p);
  return artifacts.length ? `<div class='event-artifacts'>${artifacts.map(renderEventArtifact).join('')}</div>` : '';
}
function statusClass(s) { const v = String(s || 'unknown').toLowerCase(); return ['succeeded','failed','running','cancelled','queued','starting','cancelling'].includes(v) ? v : 'unknown'; }
function fmtDuration(seconds) { const n = Number(seconds); if (!Number.isFinite(n)) return '—'; if (n < 60) return `${n.toFixed(n < 10 ? 1 : 0)}s`; const m = Math.floor(n / 60); const s = Math.round(n % 60); return `${m}m ${s}s`; }
function readableType(type) {
  const labels = {
    topology_declared:'Environment structure', team_started:'Team started', team_completed:'Team completed', team_failed:'Team failed',
    delegation_started:'Delegation started', delegation_completed:'Delegation completed', delegation_failed:'Delegation failed',
    symposium_started:'Symposium started', symposium_completed:'Symposium completed', symposium_failed:'Symposium failed',
    symposium_phase_started:'Phase started', symposium_phase_completed:'Phase completed', initial_work_started:'Initial work started', initial_work_completed:'Initial work completed',
    review_round_started:'Review round started', review_round_completed:'Review round completed', revision_round_started:'Revision round started', revision_round_completed:'Revision round completed',
    synthesis_started:'Synthesis started', synthesis_completed:'Final synthesis completed', tool_search:'Tool search', tool_execute:'Tool execution', tool_write:'File write', tool_safety_check:'Tool safety check', elo_started:'Elo run started', elo_completed:'Elo run completed', elo_failed:'Elo run failed', generation_started:'Generation started', generation_completed:'Generation completed', member_started:'Member started', member_completed:'Member completed', member_timed_out:'Member timed out', member_failed:'Member failed', pairings_declared:'Pairings created', match_started:'Match started', match_completed:'Match completed', member_eliminated:'Member eliminated', child_created:'Child created',   
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
function toolName(e) { return payload(e).tool || (e.source?.kind === 'tool' ? e.source.name : '') || (e.target?.kind === 'tool' ? e.target.name : ''); }
function isToolEvent(e) {
  const p = payload(e); const type = String(e.event_type || '');
  return type.startsWith('tool_') || !!p.tool || e.source?.kind === 'tool' || e.target?.kind === 'tool';
}
function isParticipantMessage(e) {
  const type = String(e.event_type || '');
  if (type === 'delegation_started' || type === 'delegation_completed' || type === 'delegation_failed') return true;
  const source = participantIdFromValue(e.source || payload(e).source); const target = participantIdFromValue(e.target || payload(e).target);
  return !!source && !!target && source !== target;
}
function isMilestoneEvent(e) {
  const type = String(e.event_type || '');
  return type.endsWith('_completed') || type.endsWith('_failed') || ['team_started','symposium_started'].includes(type);
}
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
function eventVisibleInTimeline(e, index) {
  if (isToolEvent(e)) return !!selectedParticipant && eventMatchesParticipant(e, selectedParticipant, index);
  if (!isMilestoneEvent(e) && !isParticipantMessage(e)) return false;
  return eventMatchesParticipant(e, selectedParticipant, index);
}
function eventSearchText(e, index) { return [e.event_type, e.message, e.stage, e.phase, sourceTargetText(e, index), text(payload(e))].join(' ').toLowerCase(); }
function extractTopology() {
  const topologies = events
    .filter(e => e.event_type === 'topology_declared' && payload(e).topology)
    .map(e => payload(e).topology);

  if (!topologies.length) {
    const fallback = events.find(e => payload(e).topology);
    return fallback ? payload(fallback).topology : null;
  }

  const nodes = new Map();
  const edges = new Map();

  for (const top of topologies) {
    for (const node of top.nodes || []) {
      const id = String(node.id || node.name);
      nodes.set(id, node);
    }
    for (const edge of top.edges || []) {
      const key = [edge.source, edge.target, edge.kind || ''].join('|');
      edges.set(key, edge);
    }
  }

  const latest = topologies[topologies.length - 1];
  return {
    ...latest,
    nodes: [...nodes.values()],
    edges: [...edges.values()],
  };
}

function extractFullTask() {
  if (manifest.task != null) return text(manifest.task);
  for (const e of events) {
    const p = payload(e);
    if (p.task != null) return text(p.task);
  }
  return manifest.task_preview || '';
}

function extractTaskPreview() {
  return manifest.task_preview || extractFullTask();
}

function extractFinal() {
  const preferred = ['elo_completed', 'team_completed', 'symposium_completed', 'synthesis_completed'];
  for (const type of preferred) {
    for (let i = events.length - 1; i >= 0; i--) {
      const p = payload(events[i]);
      if (events[i].event_type === type && (p.result || p.final)) return text(p.result || p.final);
    }
  }
  for (let i = events.length - 1; i >= 0; i--) {
    const p = payload(events[i]);
    if (p.result && String(events[i].event_type || '').endsWith('_completed')) return text(p.result);
  }
  return '';
}

function durationFromEvents() {
  for (let i = events.length - 1; i >= 0; i--) {
    const p = payload(events[i]);
    if (p.elapsed_seconds != null && ['elo_completed', 'team_completed', 'symposium_completed'].includes(events[i].event_type)) {
      return Number(p.elapsed_seconds);
    }
  }
  const ns = events.map(e => Number(e.monotonic_timestamp_ns)).filter(n => Number.isFinite(n));
  if (ns.length > 1) return (Math.max(...ns) - Math.min(...ns)) / 1e9;
  return null;
}
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
  const duration = durationFromEvents(); const status = manifest.status || 'unknown'; const task = extractFullTask(); const taskPreview = extractTaskPreview(); const workspace = likelyWorkspacePath();
  const milestoneCount = events.filter((e, i) => !isToolEvent(e) && (isMilestoneEvent(e) || isParticipantMessage(e, i))).length;
  $('statusBadge').innerHTML = `<span class='status ${statusClass(status)}'>${esc(status)}</span>`;
  const cancel = $('cancelEnvironmentRun'); cancel.hidden = manifest.launch_source !== 'dashboard' || !['queued','starting','running','cancelling'].includes(String(status).toLowerCase()); cancel.disabled = String(status).toLowerCase() === 'cancelling';
  $('summary').innerHTML = [
    ['Status', `<span class='status ${statusClass(status)}'>${esc(status)}</span>`, ''],
    ['Milestones', milestoneCount || '—', ''],
    ['Duration', fmtDuration(duration), ''],
    ['Workspace', workspace ? `<span class='workspace-path' title='${esc(workspace)}'>${esc(workspace)}</span><button class='copy-btn' id='copyWorkspace' type='button'>Copy</button>` : '<span class="muted">No workspace recorded</span>', 'workspace']
  ].map(([k,v,cls]) => `<div class='metric ${cls}'><div class='label'>${k}</div><div class='value'>${v}</div></div>`).join('');
  $('task').textContent = taskPreview || 'No task payload recorded.';
  $('copyTask').disabled = !task; $('copyTask').title = task && task !== taskPreview ? 'Copy the complete task, including text hidden by the preview truncation.' : 'Copy task';
  const copyWorkspace = $('copyWorkspace'); if (copyWorkspace) copyWorkspace.onclick = () => copyText(workspace, copyWorkspace, 'Copied');
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
    if (isToolEvent(e)) return;
    let s = participantIdFromValue(e.source || payload(e).source); let t = participantIdFromValue(e.target || payload(e).target);
    const owner = inferredOwner(e, i); if ((!s || !t) && owner && toolName(e)) { s = owner; t = owner; }
    addEdge(s, t, communicationKind(e));
  });
  return [...nodes, ...edgeMap.values()];
}
function edgeBetween(source, target) { if (!cy || !source || !target) return null; return cy.edges().filter(edge => edge.data('source') === source && edge.data('target') === target).first(); }
function graphPalette() {
  const dark = document.documentElement.dataset.theme === 'dark';
  return dark ? {
    text:'#eef2f7', outline:'#171b20', node:'#252b33', nodeBorder:'#55c6d2', environment:'#302b45', edge:'#7f8996', edgeText:'#b6bec9', active:'#8ab4ff', activeBg:'#1d3557', completed:'#81c995', completedBg:'#173a25', failed:'#f28b82', failedBg:'#44201f'
  } : {
    text:'#17202a', outline:'#fff', node:'#fff8e8', nodeBorder:'#2aaeba', environment:'#eee9ff', edge:'#8a949f', edgeText:'#5f6770', active:'#0b57d0', activeBg:'#e8f0fe', completed:'#188038', completedBg:'#e6f4ea', failed:'#b3261e', failedBg:'#fce8e6'
  };
}
function renderGraphNote() {
  if (!topology) return;
  if (!selectedParticipant) {
    $('graphNote').textContent = 'Select a scientist to focus the timeline and reveal that agent’s tool calls.';
    return;
  }
  const participant = nodeById(selectedParticipant); const role = participant?.role ? ` · ${participant.role}` : '';
  $('graphNote').innerHTML = `<span class='participant-filter'>Focused on ${esc(participantNameById(selectedParticipant))}</span>${esc(role)} · tool activity is now included.`;
}
function selectParticipant(id) {
  selectedParticipant = selectedParticipant === id ? null : id;
  const indexes = timelineIndexes(false); selected = indexes.length ? indexes[indexes.length - 1] : -1;
  renderAll();
}
function initializeGraph() {
  topology = extractTopology();
  if (!topology) { $('graph').innerHTML = `<div class='empty'>Waiting for a topology event…</div>`; $('graphNote').textContent = 'No topology has been recorded yet.'; return; }
  if (!window.cytoscape) { renderFallbackGraph(); $('graphNote').textContent = 'Cytoscape.js did not load, so a simplified graph fallback is shown.'; return; }
  if (cy) cy.destroy();
  const colors = graphPalette();
  cy = cytoscape({ container:$('graph'), elements:topologyToElements(topology), style:[
    { selector:'node', style:{ 'label':'data(label)', 'shape':'round-rectangle', 'font-size':12, 'font-weight':600, 'color':colors.text, 'text-outline-width':3, 'text-outline-color':colors.outline, 'text-valign':'bottom', 'text-margin-y':10, 'background-color':colors.node, 'background-image':SCIENTIST_BEAR_IMAGE, 'background-fit':'cover', 'background-repeat':'no-repeat', 'background-image-opacity':.98, 'border-width':3, 'border-color':colors.nodeBorder, 'width':76, 'height':76 } },
    { selector:'.node-environment', style:{ 'background-color':colors.environment, 'shape':'round-rectangle', 'width':88, 'height':66 } },
    { selector:'edge', style:{ 'curve-style':'bezier', 'target-arrow-shape':'triangle', 'line-color':colors.edge, 'target-arrow-color':colors.edge, 'label':'data(label)', 'font-size':9, 'font-weight':600, 'color':colors.edgeText, 'text-outline-width':2, 'text-outline-color':colors.outline, 'text-rotation':'autorotate', 'width':2 } },
    { selector:'node.active', style:{ 'background-color':colors.activeBg, 'border-color':colors.active, 'border-width':4 } },
    { selector:'edge.active', style:{ 'line-color':colors.active, 'target-arrow-color':colors.active, 'width':4 } },
    { selector:'node.completed', style:{ 'background-color':colors.completedBg, 'border-color':colors.completed, 'border-width':4 } },
    { selector:'edge.completed', style:{ 'line-color':colors.completed, 'target-arrow-color':colors.completed, 'width':3 } },
    { selector:'node.failed', style:{ 'background-color':colors.failedBg, 'border-color':colors.failed, 'border-width':4 } },
    { selector:'edge.failed', style:{ 'line-color':colors.failed, 'target-arrow-color':colors.failed, 'width':4 } },
    { selector:'node.selected', style:{ 'underlay-color':'#f9ab00', 'underlay-opacity':.24, 'underlay-padding':8 } }
  ], layout:{ name:'breadthfirst', directed:true, padding:46, spacingFactor:1.35 } });
  cy.on('tap', 'node', (evt) => selectParticipant(evt.target.id()));
  cy.on('tap', 'edge', (evt) => {
    const d = evt.target.data(); const upto = selected >= 0 ? selected : events.length - 1; let idx = -1;
    for (let i = upto; i >= 0; i--) {
      const e = events[i];
      if (participantIdFromValue(e.source || payload(e).source) === d.source && participantIdFromValue(e.target || payload(e).target) === d.target) { idx = i; break; }
    }
    if (idx >= 0) selectEvent(idx);
  });
  renderGraphNote();
  applyGraphState();
}
function renderFallbackGraph() {
  const top = topology || extractTopology(); const parts = (top && top.nodes) || [];
  $('graph').innerHTML = `<div class='fallback-graph'>${parts.map(p => `<div class='fallback-node' data-id='${esc(String(p.id || p.name))}'><span class='fallback-node-main'><span class='fallback-bear'><img src='${SCIENTIST_BEAR_IMAGE}' alt='' /></span><span>${esc(p.name || p.id)}</span></span><span class='chip'>${esc(p.role || p.kind || '')}</span></div>`).join('')}</div>`;
  $('graph').querySelectorAll('.fallback-node').forEach(el => el.onclick = () => selectParticipant(el.dataset.id));
  renderGraphNote();
}
function applyGraphState() {
  if (!topology) return; if (!cy) { renderFallbackGraph(); return; }
  cy.elements().removeClass('active completed failed selected');
  const upto = live ? events.length - 1 : (selected >= 0 ? selected : events.length - 1);
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
function timelineIndexes(includeSearch = true) {
  const q = includeSearch ? $('timelineSearch').value.trim().toLowerCase() : '';
  return events.map((_, i) => i).filter(i => eventVisibleInTimeline(events[i], i)).filter(i => !q || eventSearchText(events[i], i).includes(q));
}
function renderTimeline() {
  const indexes = timelineIndexes(); const rows = [...indexes].reverse(); const position = indexes.indexOf(selected); const scrub = $('scrub');
  scrub.max = Math.max(0, indexes.length - 1); scrub.value = position >= 0 ? position : Math.max(0, indexes.length - 1); scrub.disabled = !indexes.length;
  const clearParticipant = $('clearParticipant'); clearParticipant.hidden = !selectedParticipant;
  if (selectedParticipant) {
    const name = participantNameById(selectedParticipant); const toolCount = indexes.filter(i => isToolEvent(events[i])).length;
    $('timelineContext').innerHTML = `<span class='participant-filter'>${esc(name)}</span> · milestones, messages, and ${toolCount} tool call${toolCount === 1 ? '' : 's'}`;
    $('activityContext').textContent = `Latest visible event for ${name}`;
  } else {
    $('timelineContext').textContent = 'Milestones and agent-to-agent messages · Select a scientist to reveal tool calls';
    $('activityContext').textContent = 'Latest high-signal event';
  }
  $('timeline').innerHTML = rows.length ? rows.map(i => {
    const e = events[i]; const cls = eventLevel(e); const toolEvent = isToolEvent(e); const st = sourceTargetText(e, i); const p = payload(e); const dur = p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : (p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : '');
    const artifactCount = eventArtifacts(p).length;
    return `<article class='timeline-card ${cls} ${toolEvent ? 'tool' : ''} ${i === selected ? 'selected' : ''}' data-index='${i}'><div class='timeline-top'><div><div class='timeline-title'>${esc(readableType(e.event_type))}</div><div class='timeline-msg'>${esc(e.message || p.message || '')}</div></div><span class='chip'>#${esc(e.seq ?? i + 1)}</span></div><div class='timeline-meta'>${toolEvent ? `<span class='chip'>Tool</span>` : ''}${artifactCount ? `<span class='chip good'>${artifactCount === 1 ? 'Output' : `${artifactCount} outputs`}</span>` : ''}${phaseLabel(e) ? `<span class='chip'>${esc(phaseLabel(e))}</span>` : ''}${st ? `<span class='chip'>${esc(st)}</span>` : ''}${dur ? `<span class='chip'>${esc(dur)}</span>` : ''}</div></article>`;
  }).join('') : `<div class='empty'>${selectedParticipant ? 'No milestones, messages, or attributed tool calls match this agent and search.' : 'No milestones or agent-to-agent messages have been recorded yet.'}</div>`;
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
  const visible = timelineIndexes(false); const idx = visible.includes(selected) ? selected : visible[visible.length - 1]; const e = idx >= 0 ? events[idx] : null;
  if (!e) { $('currentEvent').innerHTML = `<div class='empty'>Waiting for a milestone or agent message…</div>`; return; }
  const p = payload(e); const owner = inferredOwner(e, idx); const tool = toolName(e); const cls = eventLevel(e); const artifacts = eventArtifacts(p);
  const chips = [phaseLabel(e), sourceTargetText(e, idx), p.returncode != null ? `Exit ${p.returncode}` : '', p.elapsed_seconds != null ? fmtDuration(p.elapsed_seconds) : '', p.elapsed_ms != null ? `${Math.round(p.elapsed_ms)}ms` : ''].filter(Boolean);
  let sections = '';
  if (tool && !owner) sections += block('Assignment', 'This tool event did not record the member that invoked it. The UI is not assigning it to a graph participant to avoid misleading attribution.');
  sections += block('Task / instruction', p.task || p.prompt);
  sections += block('Tool query / command', p.query || p.command || p.input, {markdown:false});
  sections += renderEventArtifacts(p);
  sections += block('Result', p.result || p.output || p.final);
  const artifactHasPath = artifacts.some(artifact => artifact.metadata && typeof artifact.metadata === 'object' && artifact.metadata.path);
  sections += block('File or path', artifactHasPath ? '' : (p.path || p.filename), {markdown:false});
  sections += block('Error', p.error || p.error_type, {markdown:false});
  sections += block('Safety rationale', p.reason);
  if (p.stdout_truncated || p.stderr_truncated) sections += block('Output note', 'The recorded output was truncated to the tool output limit.');
  if (!sections) sections = block('Details', p, {markdown:false});
  const message = e.message || p.message || '';
  $('currentEvent').innerHTML = `<div class='event-title-row'><div><div class='event-title'>${esc(readableType(e.event_type))}</div><div class='muted'>Event #${esc(e.seq ?? idx + 1)}${owner ? ` &middot; ${esc(participantNameById(owner))}` : ''}${tool ? ` &middot; ${esc(tool)}` : ''}</div></div><span class='chip ${cls === 'failed' ? 'bad' : cls === 'completed' ? 'good' : cls === 'active' ? 'active' : ''}'>${esc(cls || 'event')}</span></div>${message ? `<div class='event-message'>${mdToHtml(message)}</div>` : ''}<div class='timeline-meta'>${chips.map(c => `<span class='chip'>${esc(c)}</span>`).join('')}</div>${sections}`;
  $('currentEvent').querySelectorAll('[data-artifact-copy]').forEach(button => {
    const artifact = artifacts[Number(button.dataset.artifactCopy)]; if (!artifact) return;
    const metadata = artifact.metadata && typeof artifact.metadata === 'object' ? artifact.metadata : {};
    const mime = String(artifact.mime_type || '');
    const value = mime === 'application/vnd.ursa.file-reference' ? (metadata.path || artifact.content || '') : text(artifact.content ?? '');
    button.onclick = () => copyText(value, button, 'Copied');
  });
}
function selectEvent(i) { if (i == null || i < 0 || i >= events.length) return; selected = i; live = false; $('live').textContent = 'Resume live'; renderAll(); }
function selectAdjacent(direction) {
  const indexes = timelineIndexes(); if (!indexes.length) return; let position = indexes.indexOf(selected);
  if (position < 0) position = indexes.length - 1;
  selectEvent(indexes[Math.max(0, Math.min(indexes.length - 1, position + direction))]);
}
function renderAll() { updateSummary(); renderTimeline(); renderCurrentEvent(); applyGraphState(); renderGraphNote(); }
function addEvent(e) { if (!e || typeof e !== 'object') return; events.push(e); const index = events.length - 1; if (!topology && payload(e).topology) initializeGraph(); else if (topology && window.cytoscape) initializeGraph(); if (live && eventVisibleInTimeline(e, index)) selected = index; renderAll(); }
function setupControls() {
  $('live').onclick = () => { live = !live; $('live').textContent = live ? 'Pause live' : 'Resume live'; if (live) { const indexes = timelineIndexes(false); selected = indexes.length ? indexes[indexes.length - 1] : -1; renderAll(); } };
  $('prev').onclick = () => selectAdjacent(-1);
  $('next').onclick = () => selectAdjacent(1);
  $('scrub').oninput = () => { const indexes = timelineIndexes(); const index = indexes[Number($('scrub').value)]; if (index != null) selectEvent(index); };
  $('timelineSearch').oninput = renderTimeline;
  $('clearParticipant').onclick = () => selectParticipant(selectedParticipant);
  $('copyTask').onclick = () => copyText(extractFullTask(), $('copyTask'), 'Task copied');
  $('cancelEnvironmentRun').onclick = async () => { const button = $('cancelEnvironmentRun'); button.disabled = true; button.textContent = 'Cancelling…'; try { const response = await fetch('/environment-runs/'+encodeURIComponent(runId)+'/cancel', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reason:'user_request'})}); if (!response.ok) throw new Error('Cancellation request failed'); Object.assign(manifest, await response.json()); updateSummary(); } catch (error) { button.disabled = false; button.textContent = 'Cancel run'; console.error(error); } };
}
async function init() {
  try { const response = await fetch('/settings'); const data = response.ok ? await response.json() : null; let theme = data?.settings?.ui?.theme || 'system'; if (theme === 'system') theme = matchMedia('(prefers-color-scheme:dark)').matches ? 'dark' : 'light'; document.documentElement.dataset.theme = theme; } catch (_) {}
  setupControls(); updateSummary();
  const detail = await fetch('/environment-runs/'+encodeURIComponent(runId)).then(r => r.json()); Object.assign(manifest, detail); updateSummary();
  const data = await fetch('/environment-runs/'+encodeURIComponent(runId)+'/events').then(r => r.json()); (data.events || []).forEach(e => events.push(e));
  topology = extractTopology(); const indexes = timelineIndexes(false); selected = indexes.length ? indexes[indexes.length - 1] : -1; initializeGraph(); renderAll();
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
