"""First-run guide and the reusable default configuration editor."""

CONFIG_EDITOR_HTML = r"""
<div class="settingsPane hidden" data-settings-pane="defaults">
  <div class="guidedConfig" id="guidedConfig">
    <div class="guidedConfigEyebrow">YOUR FIRST CONNECTION</div>
    <ol class="guidedConfigSteps" aria-label="Model setup progress"><li>1. Base URL</li><li>2. API key</li><li>3. Model</li></ol>
    <div class="configNotice hidden" id="guidedConfigPriority">A launch configuration or environment override takes precedence over this file. The test checks the effective settings URSA will use.</div>
    <p class="guidedConfigLoaded" id="guidedConfigLoaded" role="status"></p>
    <section data-guided-step="0">
      <h2>Where is your model hosted?</h2>
      <p>To connect URSA to a language model, you need its <strong>Base URL</strong>: the API address where the model is hosted. Your model provider or your organization’s administrator can give you this address.</p>
      <label class="guidedField">Connection<select class="input" id="guidedProvider"></select></label>
      <label class="guidedField">Base URL<input class="input" id="guidedBaseUrl" placeholder="https://api.openai.com/v1" spellcheck="false" autocomplete="url" aria-describedby="guidedUrlHelp" /></label>
      <p class="guidedHelp" id="guidedUrlHelp">For OpenAI, use <code>https://api.openai.com/v1</code>. For a lab, company, or local model, paste the API address you were given—not the address of its chat website. Keep any <code>/v1</code> suffix your provider specifies.</p>
      <details class="guidedAdvanced"><summary>Connection name and API type</summary><p class="guidedHelp">Most OpenAI-compatible services work with Automatic. Change the API type only if your service uses a different API.</p><label class="guidedField">Connection name<input class="input" id="guidedProviderName" /></label><label class="guidedField">API type<select class="input" id="guidedApiType"></select></label></details>
    </section>
    <section data-guided-step="1" class="hidden">
      <h2>Give URSA permission to connect</h2>
      <p>An <strong>API key</strong> is a private credential from your model provider. Generate one in your provider’s account, or ask your administrator for one. It is not your account password.</p>
      <p>If you have not already set an API key in your environment, choose <strong>Enter a key · store securely</strong> below and paste your key. URSA saves it in your computer’s secure key store, not in the config file.</p>
      <label class="guidedField">How should URSA access your key?<select class="input" id="guidedKeySource"><option value="preserve">Keep current key setting</option><option value="keyring">Enter a key · store securely</option><option value="environment">Use an environment variable</option><option value="none">No key needed</option></select></label>
      <label class="guidedField hidden" id="guidedKeyRow">Paste your API key<input class="input" id="guidedApiKey" type="password" autocomplete="new-password" spellcheck="false" placeholder="Paste the key from your provider" /></label>
      <div id="guidedEnvRow" class="hidden"><label class="guidedField">Environment variable name<input class="input" id="guidedKeyEnv" placeholder="OPENAI_API_KEY" spellcheck="false" /></label><p class="guidedHelp">Enter the variable’s name, not the key itself. It must be set in the environment that launched this dashboard. Otherwise, choose “Enter a key · store securely.”</p></div>
      <p class="guidedHelp" id="guidedKeyHelp"></p>
    </section>
    <section data-guided-step="2" class="hidden">
      <h2>Choose the model you want to talk to</h2>
      <p>A provider can host several models. Choose one available to your account, or keep the model from your existing config.</p>
      <label class="guidedField">Language model<input class="input" id="guidedModel" list="configLlmModels" placeholder="provider:model-name" spellcheck="false" /></label>
      <div class="configTestActions"><button class="btn" type="button" id="guidedFindModelsBtn">Find available models</button><button class="btn" type="button" id="guidedTestBtn">Test connection</button></div>
      <p class="guidedHelp" id="guidedModelStatus" role="status" aria-live="polite">You can also type the model name supplied by your provider.</p>
      <div class="configTestResult" id="guidedTestResult" role="status" aria-live="polite" data-state="idle">Not tested yet.</div>
      <p class="guidedHelp">Testing sends a tiny request and may incur a small provider charge. Continue to test and save these defaults. Embeddings and advanced options can be edited later in <strong>Model and Agent Settings → Default config</strong>.</p>
    </section>
    <p id="guidedConfigError" role="alert"></p>
    <div class="guidedConfigActions"><button class="btn" type="button" id="guidedBackBtn">Back</button><button class="btn primary" type="button" id="guidedNextBtn">Continue to API key</button></div>
  </div>
  <div class="section defaultConfigEditor">
    <div class="configSectionHeading"><div class="sectionHead">Your default URSA configuration</div><button class="btn" type="button" id="reloadDefaultConfigBtn">Reload defaults</button></div>
    <p class="muted">Connect a model, then use it across the dashboard, CLI, and TUI. Add more providers whenever you need them.</p>
    <div class="configPath" id="defaultConfigPath"></div>
    <div class="configNotice hidden" id="defaultConfigPriority">A launch configuration or environment override has higher priority. The connection test uses your effective defaults.</div>
    <div id="defaultConfigStatus" role="status" aria-live="polite"></div>
    <div class="configSectionHeading"><h3>1. Connect your providers</h3><button class="btn" type="button" id="addConfigProviderBtn">+ Add provider</button></div>
    <p class="muted small">The Base URL is your service's API address. For a private service, ask its administrator for the URL, model names, and API key.</p>
    <div id="configProviders"></div>
    <h3>2. Choose your default models</h3>
    <p class="muted small">The language model powers your agents. An embedding model is optional and is used to search document collections.</p>
    <div class="configModelCard" id="configLlmFields"></div>
    <label class="configCheck"><input id="configEmbeddingEnabled" type="checkbox" /> Configure an embedding model for document search</label>
    <div class="configModelCard" id="configEmbeddingFields"></div>
    <h3>3. Test your connection</h3>
    <p class="muted small">Each test sends a tiny request to the selected model and may incur a small provider charge. Testing does not save your changes.</p>
    <div class="configTestActions configTests">
      <div><button class="btn" id="testConfigLlmBtn" type="button">Test language model</button><div class="configTestResult" id="configLlmTestResult" role="status" aria-live="polite" data-state="idle">Not tested yet.</div></div>
      <div><button class="btn" id="testConfigEmbeddingBtn" type="button">Test embeddings</button><div class="configTestResult" id="configEmbTestResult" role="status" aria-live="polite" data-state="idle">Not tested yet.</div></div>
    </div>
    <p class="muted small">Update saves these defaults to your user config and applies them to new sessions immediately. Existing sessions keep their settings. A backup is kept before each edit.</p>
  </div>
</div>
"""

ONBOARDING_HTML = r"""
<aside class="tourCard hidden" id="ursaTour" role="dialog" aria-labelledby="tourTitle" aria-describedby="tourCopy">
  <div class="tourTop"><span class="tourEyebrow">GET TO KNOW URSA</span><button type="button" class="tourDismiss" id="tourDismissBtn" aria-label="Close walkthrough">×</button></div>
  <div class="tourProgress" id="tourProgress"></div>
  <h2 id="tourTitle"></h2><div id="tourCopy"></div>
  <div class="tourStatus" id="tourStatus" role="status" aria-live="polite"></div>
  <div class="tourActions"><button class="btn" type="button" id="tourBackBtn">Back</button><button class="btn primary" type="button" id="tourNextBtn">Let's begin</button></div>
  <button class="tourSkip" id="tourSkipBtn" type="button">I'll explore on my own</button>
</aside>
"""

ONBOARDING_CSS = r"""
.sidebarNav { grid-template-columns: 1fr; }
.welcomeGuideLink { border:0; background:none; color:var(--muted); padding:8px 0; cursor:pointer; font:inherit; font-size:14px; text-decoration:underline; text-underline-offset:4px; }
.welcomeGuideLink:hover { color:var(--text); }
.configPath { font-family:var(--mono); font-size:12px; overflow-wrap:anywhere; padding:10px 12px; border:1px solid var(--border); border-radius:9px; margin-bottom:12px; }
.configNotice { padding:10px 12px; border-left:3px solid #c8953e; background:rgba(200,149,62,.08); font-size:13px; margin-bottom:12px; }
.configSectionHeading, .configTestActions { display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.configTestActions { justify-content:flex-start; }
.defaultConfigEditor h3 { font-size:15px; margin:22px 0 10px; }
.configProviderCard, .configModelCard { border:1px solid var(--border); border-radius:12px; padding:14px; margin:12px 0; }
.configFields { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.configFields label, .configModelCard label { display:flex; flex-direction:column; gap:6px; font-size:13px; min-width:0; }
.configFullWidth { grid-column:1 / -1; }
.configFields .input, .configModelCard .input { width:100%; box-sizing:border-box; }
.configCredentialHint { font-size:12px; color:var(--muted); overflow-wrap:anywhere; margin:8px 0 0; }
.configCheck { display:flex; align-items:center; gap:8px; font-size:14px; margin:14px 0; }
.configModelCard details { margin-top:12px; }
.configModelCard summary { cursor:pointer; color:var(--muted); font-size:13px; }
.configModelCard textarea { margin-top:8px; font-family:var(--mono); }
#defaultConfigStatus { font-size:14px; line-height:1.5; white-space:pre-line; }
#defaultConfigStatus[data-error="true"], #guidedConfigError, .tourStatus[data-state="failed"] { color:#a73528; }
.configTests { align-items:flex-start; }
.configTests > div { flex:1; min-width:180px; }
.configTestResult { margin:10px 0; padding:10px 12px; border:1px solid var(--border); border-radius:9px; color:var(--muted); font-size:14px; line-height:1.5; overflow-wrap:anywhere; }
.configTestResult[data-state="passed"], .tourStatus[data-state="passed"] { color:#17643c; }
.configTestResult[data-state="passed"] { border-color:#66a983; background:#edf8f1; }
.configTestResult[data-state="failed"] { color:#a73528; border-color:#c6766a; background:#fff1ed; }
.configTestResult[data-state="testing"] { color:#195777; border-color:#6597b5; background:#eef6fc; }
.guidedConfig { display:none; max-width:680px; margin:auto; padding:12px clamp(8px, 2vw, 28px) 24px; }
.guidedConfigEyebrow { color:#24627b; font-size:12px; font-weight:750; letter-spacing:.12em; }
.guidedConfigSteps { display:flex; list-style:none; padding:0; gap:12px; margin:18px 0 24px; }
.guidedConfigSteps li { flex:1; border-top:3px solid var(--border); padding-top:10px; font-size:13px; color:var(--muted); }
.guidedConfigSteps li.current { color:#24627b; border-color:#387e9c; font-weight:700; }
.guidedConfig h2 { font-size:clamp(24px, 2.5vw, 30px); line-height:1.2; margin:12px 0 16px; letter-spacing:-.025em; }
.guidedConfig p { font-size:16px; line-height:1.65; margin:12px 0 18px; }
.guidedConfig p.guidedHelp, .guidedConfig p.guidedConfigLoaded { font-size:14px; color:var(--muted); }
.guidedConfigLoaded:empty, #guidedConfigError:empty { display:none; }
.guidedField { display:flex; flex-direction:column; gap:8px; margin:20px 0 12px; font-size:15px; font-weight:650; }
.guidedField .input { width:100%; box-sizing:border-box; min-height:48px; font-size:16px; padding:12px; }
.guidedAdvanced { margin:20px 0 0; font-size:14px; }
.guidedAdvanced summary { cursor:pointer; color:var(--muted); }
.guidedConfigActions { display:flex; justify-content:space-between; gap:12px; padding:16px 0; margin-top:20px; border-top:1px solid var(--border); position:sticky; bottom:0; background:var(--panelSolid, #fff); }
.guidedConfigActions .btn { min-height:44px; }
.tourCard { position:fixed; z-index:150; bottom:20px; left:16px; width:304px; max-height:calc(100dvh - 40px); overflow:auto; box-sizing:border-box; padding:22px; border:2px solid #397d9c; border-top:5px solid #397d9c; border-radius:18px; background:#f3faff; color:var(--text); box-shadow:0 10px 40px rgba(21,73,99,.22); line-height:1.55; }
:root[data-theme="dark"] .tourCard { background:#182d3a; border-color:#78bbd9; }
:root[data-theme="dark"] .tourEyebrow, :root[data-theme="dark"] .guidedConfigEyebrow, :root[data-theme="dark"] .guidedConfigSteps li.current { color:#91d1ee; }
:root[data-theme="dark"] .configTestResult[data-state="passed"] { color:#a1e2b9; background:#163728; border-color:#508d65; }
:root[data-theme="dark"] .configTestResult[data-state="failed"] { color:#ffc2b8; background:#3e2527; border-color:#b27670; }
:root[data-theme="dark"] .configTestResult[data-state="testing"] { color:#acd7f4; background:#1b3549; }
:root[data-theme="dark"] #defaultConfigStatus[data-error="true"], :root[data-theme="dark"] #guidedConfigError, :root[data-theme="dark"] .tourStatus[data-state="failed"] { color:#ffc2b8; }
:root[data-theme="dark"] .tourStatus[data-state="passed"] { color:#a1e2b9; }
.tourCard.tourIntro { top:50%; bottom:auto; left:50%; transform:translate(-50%, -50%); width:min(460px, calc(100vw - 32px)); padding:32px; box-shadow:0 0 0 100vmax rgba(0,0,0,.3), 0 20px 80px rgba(0,0,0,.2); }
.tourTop { display:flex; justify-content:space-between; align-items:center; gap:12px; }
.tourEyebrow { font-size:11px; font-weight:800; letter-spacing:.14em; color:#24627b; }
.tourDismiss { border:0; background:none; color:var(--muted); font-size:24px; cursor:pointer; padding:0 3px; }
.tourCard h2 { font-size:23px; line-height:1.2; margin:16px 0 12px; letter-spacing:-.02em; }
#tourCopy { font-size:14px; }
#tourCopy p { margin:0 0 12px; }
.tourProgress { display:flex; gap:5px; margin-top:14px; }
.tourProgress span { height:3px; flex:1; background:var(--border); border-radius:3px; }
.tourProgress span.current { background:#397d9c; }
.tourActions { display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:20px; }
.tourSkip { background:none; border:0; padding:14px 0 0; font:inherit; font-size:12px; color:var(--muted); cursor:pointer; }
.tourStatus { font-size:13px; margin-top:8px; }
.tourHighlight { outline:3px solid #54869e !important; outline-offset:5px; border-radius:10px; }
.tourActive .modal .modalCard { left:344px; transform:none; width:calc(100vw - 368px); max-width:1000px; }
.tourActive .modal .smallModalCard { max-width:620px; max-height:calc(100dvh - 48px); overflow:auto; top:24px; }
.tourConfigActive .settingsShell { grid-template-columns:1fr; }
.tourConfigActive .settingsNav, .tourConfigActive .defaultConfigEditor, .tourConfigActive #settingsModal .modalCard > .topbar, .tourConfigActive #settingsModal .modalCard > .settingsFooter { display:none; }
.tourConfigActive .guidedConfig { display:block; }
@media(max-width:900px) {
  .tourActive .tourCard:not(.tourIntro) { bottom:12px; left:12px; width:calc(100vw - 24px); max-height:36dvh; padding:14px 18px; }
  .tourActive .app { height:calc(100dvh - var(--tour-dock-height, 280px)); }
  .tourActive .modal .modalCard { top:12px; left:12px; width:calc(100vw - 24px); max-width:none; height:calc(100dvh - var(--tour-dock-height, 280px) - 24px); max-height:calc(100dvh - var(--tour-dock-height, 280px) - 24px); }
  .tourActive .modal .smallModalCard { height:auto; }
  .tourActive .tourCard h2 { font-size:21px; margin:10px 0; }
  .tourActive .tourActions { margin-top:12px; }
  .tourActive .tourSkip { padding-top:8px; }
}
@media(max-width:600px) { .configFields { grid-template-columns:1fr; } .guidedConfigSteps { gap:8px; } }
"""

ONBOARDING_JS = r"""
  let defaultConfigDraft = null;
  let defaultConfigBusy = false;
  let lastConfigTest = null;
  let guidedConfigStep = 0;
  let guidedProviderIndex = 0;
  const configStatus = (message, error=false) => {
    const el = $('#defaultConfigStatus');
    el.textContent = message;
    el.dataset.error = String(error);
  };

  function configTestFeedback(kind, status, message) {
    const selectors = kind === 'chat' ? ['#configLlmTestResult', '#guidedTestResult'] : ['#configEmbTestResult'];
    for (const selector of selectors) {
      const result = $(selector);
      result.dataset.state = status;
      result.textContent = message;
    }
  }

  function invalidateConfigTests() {
    lastConfigTest = null;
    for (const kind of ['chat', 'embedding']) {
      const result = $(kind === 'chat' ? '#configLlmTestResult' : '#configEmbTestResult');
      if (result.dataset.state !== 'idle') configTestFeedback(kind, 'stale', 'Settings changed — test again to check this connection.');
    }
  }

  async function findConfigModels(kind) {
    if (defaultConfigBusy) return;
    const id = kind === 'chat' ? 'configLlm' : 'configEmb';
    const button = $(`#${id}Discover`);
    button.disabled = true;
    $('#guidedFindModelsBtn').disabled = true;
    try {
      configStatus('Finding models…');
      $('#guidedModelStatus').textContent = 'Finding models…';
      const response = await api('POST', '/user-config/models?kind='+kind, collectDefaultConfig());
      $(`#${id}Models`).innerHTML = response.models.map(model => `<option value="${escHtml(model.qualified_name)}">${escHtml(model.name)}</option>`).join('');
      const message = `${response.models.length} models available. Click the model field to choose, or type a name.`;
      configStatus(message);
      $('#guidedModelStatus').textContent = message;
      $(tourIndex === 1 ? '#guidedModel' : `#${id}Model`).focus();
    } catch(error) {
      configStatus(error.message, true);
      $('#guidedModelStatus').textContent = 'Could not load models. Check your Base URL and key, or type the model name supplied by your provider.';
    } finally { button.disabled = false; $('#guidedFindModelsBtn').disabled = false; }
  }

  function configProviderOptions(selected) {
    const names = [...new Set([...(defaultConfigDraft?.providers || []).map(x => x.name), selected].filter(Boolean))];
    return '<option value="">Direct endpoint</option>' + names.map(name => `<option value="${escHtml(name)}" ${name === selected ? 'selected' : ''}>${escHtml(name)}</option>`).join('');
  }

  function configModelFields(id, label, model) {
    const current = model || {model:'', options:{}};
    return `<div class="configFields">
      <label>${label}<input class="input" id="${id}Model" list="${id}Models" value="${escHtml(current.model || '')}" placeholder="provider:model-name" /><datalist id="${id}Models"></datalist></label>
      <label>Inference provider<select class="input" id="${id}Provider">${configProviderOptions(current.inference_provider)}</select></label>
      <label class="configFullWidth ${current.inference_provider ? 'hidden' : ''}" id="${id}UrlRow">Direct Base URL<input class="input" id="${id}Url" value="${escHtml(current.base_url || '')}" placeholder="https://your-endpoint/v1" /></label>
    </div><button class="btn" type="button" id="${id}Discover" style="margin-top:12px">Find available models</button><details><summary>Advanced model options</summary><textarea class="input" id="${id}Options" rows="4" aria-label="${label} advanced options">${escHtml(JSON.stringify(current.options || {}, null, 2))}</textarea></details>`;
  }

  function renderConfigModels() {
    $('#configLlmFields').innerHTML = configModelFields('configLlm', 'Language model', defaultConfigDraft.llm_model);
    $('#configEmbeddingFields').innerHTML = configModelFields('configEmb', 'Embedding model', defaultConfigDraft.emb_model);
    $('#configEmbeddingEnabled').checked = Boolean(defaultConfigDraft.emb_model);
    const showEmbedding = () => {
      $('#configEmbeddingFields').classList.toggle('hidden', !$('#configEmbeddingEnabled').checked);
      $('#testConfigEmbeddingBtn').disabled = !$('#configEmbeddingEnabled').checked;
    };
    $('#configEmbeddingEnabled').onchange = showEmbedding;
    showEmbedding();
    for (const id of ['configLlm', 'configEmb']) {
      $(`#${id}Provider`).onchange = () => $(`#${id}UrlRow`).classList.toggle('hidden', Boolean($(`#${id}Provider`).value));
      $(`#${id}Discover`).onclick = () => findConfigModels(id === 'configLlm' ? 'chat' : 'embedding');
    }
  }

  function renderConfigProviders() {
    const list = $('#configProviders');
    list.replaceChildren();
    for (const [index, provider] of defaultConfigDraft.providers.entries()) {
      const card = document.createElement('div');
      card.className = 'configProviderCard';
      card.dataset.index = index;
      card.innerHTML = `<div class="configFields">
        <label>Provider name<input class="input" data-field="name" value="${escHtml(provider.name)}" placeholder="my-lab" ${provider._new ? '' : 'readonly'} /></label>
        <label>API type<select class="input" data-field="model_provider">
          <option value="">Automatic</option><option value="openai">OpenAI compatible</option><option value="anthropic">Anthropic</option><option value="google_genai">Google Gemini</option><option value="ollama">Ollama</option><option value="azure_openai">Azure OpenAI</option>
        </select></label>
        <label class="configFullWidth">Base URL<input class="input" data-field="base_url" value="${escHtml(provider.base_url || '')}" placeholder="https://api.openai.com/v1" /></label>
        <label>API key source<select class="input" data-field="credential_mode"><option value="preserve">Keep current key setting</option><option value="keyring">Enter a key · store securely</option><option value="environment">Environment variable</option><option value="none">No key needed</option></select></label>
        <label data-key-field>API key<input class="input" type="password" data-field="api_key" autocomplete="new-password" placeholder="Stored securely, never in config.yaml" /></label>
        <label data-env-field>Environment variable name<input class="input" data-field="api_key_env" value="${escHtml(provider.api_key_env || '')}" placeholder="OPENAI_API_KEY" /></label>
      </div><p class="configCredentialHint">${escHtml(provider.credential_description ? 'Current setting: '+provider.credential_description : 'Choose how URSA should authenticate with this provider.')}</p>`;
      list.appendChild(card);
      $('[data-field="model_provider"]', card).value = provider.model_provider || '';
      if (provider.model_provider && !$('[data-field="model_provider"]', card).value) {
        const option = document.createElement('option');
        option.value = option.textContent = provider.model_provider;
        $('[data-field="model_provider"]', card).appendChild(option);
        $('[data-field="model_provider"]', card).value = provider.model_provider;
      }
      const source = $('[data-field="credential_mode"]', card);
      source.value = provider.credential_mode;
      const toggle = () => {
        $('[data-key-field]', card).classList.toggle('hidden', source.value !== 'keyring');
        $('[data-env-field]', card).classList.toggle('hidden', source.value !== 'environment');
      };
      source.onchange = toggle;
      toggle();
      $('[data-field="name"]', card).onchange = () => {
        const previousName = provider.name;
        provider.name = $('[data-field="name"]', card).value.trim();
        for (const id of ['configLlm', 'configEmb']) {
          const select = $(`#${id}Provider`);
          const selected = select.value === previousName ? provider.name : select.value;
          select.innerHTML = configProviderOptions(selected);
        }
      };
    }
  }

  async function loadDefaultConfig() {
    configStatus('Loading your defaults…');
    try {
      defaultConfigDraft = await api('GET', '/user-config');
      if (!defaultConfigDraft.exists) defaultConfigDraft.providers.forEach(provider => { provider._new = true; });
      lastConfigTest = null;
      configTestFeedback('chat', 'idle', 'Not tested yet.');
      configTestFeedback('embedding', 'idle', 'Not tested yet.');
      $('#defaultConfigPath').textContent = defaultConfigDraft.path;
      $('#defaultConfigPriority').classList.toggle('hidden', !defaultConfigDraft.higher_priority_config);
      renderConfigProviders();
      renderConfigModels();
      configStatus(defaultConfigDraft.exists ? 'Loaded your existing config. Unrelated settings will be kept.' : 'Start with these suggested defaults, or enter the provider and models you use.');
    } catch (error) {
      defaultConfigDraft = null;
      configStatus(error.message, true);
    }
  }

  function collectDefaultConfig() {
    if (!defaultConfigDraft) throw new Error('Load the default config before updating.');
    const providers = $$('.configProviderCard').map(card => {
      const value = name => $('[data-field="'+name+'"]', card).value.trim();
      return {name:value('name'), base_url:value('base_url') || null, model_provider:value('model_provider') || null, credential_mode:value('credential_mode'), api_key_env:value('api_key_env') || null, api_key:value('api_key') || null};
    });
    const model = id => ({model:$(`#${id}Model`).value.trim(), inference_provider:$(`#${id}Provider`).value || null, base_url:$(`#${id}Url`).value.trim() || null, options:_jsonObjectFromTextarea(`#${id}Options`, 'Model options')});
    return {revision:defaultConfigDraft.revision, providers, llm_model:model('configLlm'), emb_model:$('#configEmbeddingEnabled').checked ? model('configEmb') : null};
  }

  async function saveDefaultConfig() {
    if (defaultConfigBusy) return false;
    defaultConfigBusy = true;
    try {
      const payload = collectDefaultConfig();
      configStatus('Saving your default configuration…');
      const result = await api('PUT', '/user-config', payload);
      state.settings = result.settings || state.settings;
      applyTheme(state.settings?.ui?.theme || 'system');
      defaultConfigDraft = result;
      renderConfigProviders();
      configStatus('Defaults saved. New sessions will use them.');
      await refreshInferenceProviders();
      return true;
    } catch (error) {
      configStatus(error.message, true);
      return false;
    } finally { defaultConfigBusy = false; }
  }

  async function testDefaultConfig(kind) {
    if (defaultConfigBusy) return false;
    defaultConfigBusy = true;
    const button = $(kind === 'chat' ? '#testConfigLlmBtn' : '#testConfigEmbeddingBtn');
    const label = button.textContent;
    button.textContent = 'Testing…';
    for (const selector of ['#testConfigLlmBtn', '#testConfigEmbeddingBtn', '#guidedTestBtn']) $(selector).disabled = true;
    configTestFeedback(kind, 'testing', 'Testing… Waiting for the model (up to 35 seconds).');
    try {
      configStatus('Testing the model. This can take up to 35 seconds…');
      const payload = collectDefaultConfig();
      const result = await api('POST', '/user-config/test?kind='+kind, payload);
      if (JSON.stringify(payload) !== JSON.stringify(collectDefaultConfig())) {
        configTestFeedback(kind, 'stale', 'Settings changed during the test — test again before saving.');
        return false;
      }
      if (kind === 'chat') lastConfigTest = JSON.stringify(payload);
      const message = `Passed — ${result.model} responded successfully.`;
      configTestFeedback(kind, 'passed', message);
      configStatus(message);
      return true;
    } catch (error) {
      if (kind === 'chat') lastConfigTest = null;
      configTestFeedback(kind, 'failed', `Failed — ${error.message}`);
      configStatus(error.message, true);
      return false;
    } finally {
      defaultConfigBusy = false;
      button.textContent = label;
      $('#testConfigLlmBtn').disabled = false;
      $('#testConfigEmbeddingBtn').disabled = !$('#configEmbeddingEnabled').checked;
      $('#guidedTestBtn').disabled = false;
    }
  }

  // The friendly setup uses the same underlying draft as the full editor.
  // Only the chosen connection and language model are edited; other providers,
  // embeddings, and advanced options stay intact.
  function guidedProviderCard() { return $$('.configProviderCard')[guidedProviderIndex]; }

  function renderGuidedKeyFields() {
    const source = $('#guidedKeySource').value;
    $('#guidedKeyRow').classList.toggle('hidden', source !== 'keyring');
    $('#guidedEnvRow').classList.toggle('hidden', source !== 'environment');
    $('#guidedKeyHelp').textContent = source === 'preserve'
      ? (defaultConfigDraft.providers[guidedProviderIndex].credential_description || 'Your current key setting will be kept.')
      : source === 'none' ? 'Only choose this if your endpoint does not require authentication.'
      : source === 'keyring' ? 'Your key stays on this computer. It is not displayed again after saving. On a remote dashboard, use HTTPS.'
      : 'The connection test will check whether URSA can use this variable.';
  }

  function selectGuidedProvider() {
    guidedProviderIndex = Number($('#guidedProvider').value);
    const card = guidedProviderCard();
    const value = name => $('[data-field="'+name+'"]', card).value;
    $('#guidedBaseUrl').value = value('base_url');
    $('#guidedProviderName').value = value('name');
    $('#guidedProviderName').readOnly = $('[data-field="name"]', card).readOnly;
    $('#guidedApiType').innerHTML = $('[data-field="model_provider"]', card).innerHTML;
    $('#guidedApiType').value = value('model_provider');
    $('#guidedKeySource').value = value('credential_mode');
    if (!defaultConfigDraft.exists && value('api_key_env') && value('credential_mode') === 'preserve') {
      $('#guidedKeySource').value = 'environment';
      $('[data-field="credential_mode"]', card).value = 'environment';
    }
    $('#guidedApiKey').value = value('api_key');
    $('#guidedKeyEnv').value = value('api_key_env');
    $('#configLlmProvider').value = value('name');
    $('#configLlmProvider').onchange();
    renderGuidedKeyFields();
  }

  function syncGuidedField(inputId, field) {
    const input = $(inputId);
    const target = $('[data-field="'+field+'"]', guidedProviderCard());
    target.value = input.value;
    if (field === 'name') {
      target.onchange();
      $('#guidedProvider').selectedOptions[0].textContent = input.value || 'New connection';
    }
    if (field === 'credential_mode') { target.onchange(); renderGuidedKeyFields(); }
    invalidateConfigTests();
  }

  function showGuidedConfigStep(index) {
    guidedConfigStep = index;
    $$('[data-guided-step]').forEach(el => el.classList.toggle('hidden', Number(el.dataset.guidedStep) !== index));
    $$('.guidedConfigSteps li').forEach((el, i) => {
      el.classList.toggle('current', i <= index);
      if (i === index) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
    });
    $('#guidedConfigError').textContent = '';
    const labels = ['Continue to API key', 'Continue to model', 'Test, save & continue'];
    $('#guidedNextBtn').textContent = $('#tourNextBtn').textContent = labels[index];
    $('#tourTitle').textContent = ['Connect your model', 'Add your API key', 'Check your connection'][index];
    $('#tourCopy').innerHTML = [
      '<p>We’ll connect one language model, one step at a time. Start with the address where it is hosted.</p><p>Your existing defaults are loaded when available. You can add other providers later.</p>',
      '<p>Choose how URSA should authenticate. If you have a key to paste, select <strong>Enter a key · store securely</strong>.</p><p>You don’t need to set up environment variables to get started.</p>',
      '<p>Choose a language model, then test and save your connection. You’ll see an explicit <strong>Passed</strong> or <strong>Failed</strong> result.</p><p>Only a tiny test request is sent. Provider charges may apply.</p>',
    ][index];
    $('.settingsContent').scrollTop = 0;
    $(['#guidedBaseUrl', '#guidedKeySource', '#guidedModel'][index]).focus();
  }

  function validateGuidedConfigStep() {
    let message = '';
    if (!defaultConfigDraft) message = 'Your defaults could not be loaded. Close the walkthrough and check the configuration file before trying again.';
    else if (guidedConfigStep === 0) {
      const url = $('#guidedBaseUrl').value.trim();
      // Existing environment placeholders and provider-default URLs are valid.
      if (url && !url.includes('${')) {
        try { const parsed = new URL(url); if (!['http:', 'https:'].includes(parsed.protocol) || !parsed.hostname || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error(); }
        catch { message = 'Enter a complete http:// or https:// API address, without a key or password in the URL.'; }
      }
      if (!url && !defaultConfigDraft.exists) message = 'Paste the Base URL supplied by your provider to continue.';
      if (!/^[\w.-]+$/.test($('#guidedProviderName').value)) message = 'Use letters, numbers, dots, dashes, or underscores for the connection name.';
    } else if (guidedConfigStep === 1) {
      const source = $('#guidedKeySource').value;
      if (source === 'keyring' && !$('#guidedApiKey').value.trim()) message = 'Paste your API key, or choose a different key source.';
      if (source === 'environment' && !/^[A-Za-z_][A-Za-z0-9_]*$/.test($('#guidedKeyEnv').value.trim())) message = 'Enter the environment variable’s name, such as OPENAI_API_KEY—not the key itself.';
      const original = defaultConfigDraft.providers[guidedProviderIndex];
      if (source === 'preserve' && ($('#guidedBaseUrl').value.trim() || null) !== (original.base_url || null)) message = 'The Base URL changed. Choose the key source again to approve sending a key to this endpoint.';
    } else if (!$('#guidedModel').value.trim()) message = 'Choose or enter a language model before testing.';
    $('#guidedConfigError').textContent = message;
    return !message;
  }

  async function openDefaultConfig() {
    $('#settingsModal').classList.add('open');
    await loadSettings({mode:'global'});
    setSettingsSection('defaults');
    await loadDefaultConfig();
    if (!defaultConfigDraft) {
      $('#guidedConfigError').textContent = $('#defaultConfigStatus').textContent;
      return;
    }
    let index = defaultConfigDraft.providers.findIndex(provider => provider.name === defaultConfigDraft.llm_model?.inference_provider);
    const directConnection = index < 0;
    if (directConnection) {
      // Never silently replace an existing direct endpoint with an unrelated
      // provider just because it is first in the catalog.
      let name = 'my-provider';
      for (let suffix = 2; defaultConfigDraft.providers.some(provider => provider.name === name); suffix++) name = `my-provider-${suffix}`;
      index = defaultConfigDraft.providers.length;
      defaultConfigDraft.providers.push({name, base_url:defaultConfigDraft.llm_model?.base_url || '', credential_mode:'keyring', _new:true});
      renderConfigProviders();
      $('#configLlmProvider').innerHTML = configProviderOptions(name);
    }
    $('#guidedProvider').innerHTML = defaultConfigDraft.providers.map((provider, i) => `<option value="${i}">${escHtml(provider.name)}</option>`).join('');
    $('#guidedProvider').value = String(index);
    selectGuidedProvider();
    $('#guidedModel').value = $('#configLlmModel').value;
    $('#guidedConfigLoaded').textContent = directConnection
      ? 'Loaded your direct endpoint. This setup will give it a connection name; choose its API key source in the next step. Other providers stay unchanged.'
      : defaultConfigDraft.exists ? 'Loaded your existing defaults. Other providers and advanced settings will be kept.' : 'Suggested defaults are filled in. Replace them with the connection you use.';
    $('#guidedConfigPriority').classList.toggle('hidden', !defaultConfigDraft.higher_priority_config);
    showGuidedConfigStep(0);
  }

  const FIRST_TASK = 'Make a compelling plot of the spacings between the first 10,000 prime numbers and give it some cool Gen-alpha flair. Write and run Python to compute the primes, then save one polished, clearly labeled PNG named prime_spacings.png and the script in the workspace. Keep the math accurate, use a bold but readable design, and briefly explain the main patterns. Use local computation; no web search is needed.';
  const TOUR_STEPS = [
    {title:'Welcome to URSA', copy:'<p>Turn a question into code, analysis, and useful files—with agents that can plan, execute, and review their work.</p><p>This short walkthrough will connect your models and help you try your first problem. You can reopen it anytime from the welcome page.</p>', next:"Let's begin"},
    {title:'Connect your model', copy:'<p>We’ll start with one connection: its Base URL, API key, and language model.</p>', next:'Continue to API key'},
    {title:'A starting point for any question', copy:'<p>The welcome page offers examples of what URSA can do. Each one selects a behavior and prepares a prompt you can edit.</p><p>Use <strong>New examples</strong> for more ideas, or open a blank chat to start from your own question.</p>', target:'#welcomeTaskList'},
    {title:'Choose how URSA works', copy:'<p><strong>Chat</strong> opens a place to ask your question. Choose a behavior below the prompt: explore a topic, execute a task, or plan a larger project.</p><p>Hover over a behavior for an explanation of when it helps. You can choose a different behavior as your work evolves.</p>', target:'#composerAgentType'},
    {title:'An agent you can return to', copy:'<p>The <strong>+</strong> menu creates a persistent, named agent or reconnects an existing one.</p><p>Persistent agents can retain context and build on work with you. Ordinary sessions are the default; create a named agent when you want continuity across your work.</p>', target:'#sessionCreateMenu'},
    {title:'Follow the work and keep the results', copy:'<p><strong>Logs</strong> shows progress and tool activity. <strong>Artifacts</strong> lets you preview and download generated files. Toggle either panel when you need it.</p><p>Your <strong>workspace</strong> is the folder where URSA starts working in this session and saves its files. To change it later, use <strong>Set workspace</strong> in the Artifacts panel.</p>', target:'.panelToggleGroup'},
    {title:'Try a little prime-number flair', copy:'<p>Let’s compute 10,000 primes and make a bold plot of their spacings. <strong>Prepare example</strong> selects <strong>Chat</strong> for a quick, straightforward task.</p><p>After you press <strong>Send</strong>, a workspace picker opens. Choose the folder where URSA will start this session’s work, or a temporary workspace. Change it later with <strong>Artifacts → Set workspace</strong>.</p><p>Provider usage charges apply.</p>', next:'Prepare example', target:'#messageInput'},
    {title:'Bring a team to bigger problems', copy:'<p><strong>Environment runs</strong> supports teams and symposiums: multiple agents contributing, reviewing, and refining complex work.</p><p>Your plotting task can keep running while you explore. New environments are being developed—watch for them in future URSA releases.</p><a class="btn" href="/ui/environment-runs" target="_blank" rel="noopener">Explore environment runs ↗</a><p class="muted small" style="margin-top:8px">Opens in a separate tab so your work stays here.</p>', next:'Finish walkthrough', target:'#environmentRunsLink'},
  ];
  let tourIndex = -1;
  let tourBusy = false;
  let tourReturnFocus = null;
  let tourPreviousSession = null;
  let tourPreviousDraft = '';
  let tourPreparedExample = false;

  async function markTourSeen() {
    const result = await api('PATCH', '/settings', {patch:{ui:{walkthrough_seen:true}}});
    state.settings = result.settings || state.settings;
  }

  function closeTour() {
    if (tourBusy || defaultConfigBusy) return;
    $('#guidedApiKey').value = '';
    if (tourIndex === 1) {
      $('#settingsModal').classList.remove('open');
      $('#settingsModal').setAttribute('aria-hidden', 'true');
      $$('#settingsModal input[type="password"]').forEach(input => { input.value = ''; });
      defaultConfigDraft = null;
      lastConfigTest = null;
    }
    tourIndex = -1;
    $('#ursaTour').classList.add('hidden');
    $('.app').inert = false;
    document.body.classList.remove('tourActive', 'tourConfigActive');
    $$('.tourHighlight').forEach(el => el.classList.remove('tourHighlight'));
    state._sessionCreateMenuOpen = false;
    renderSessionCreateMenu();
    if (!tourPreparedExample && tourPreviousSession && !state.activeSessionId) {
      loadSession(tourPreviousSession).then(() => { $('#messageInput').value = tourPreviousDraft; }).catch(() => {});
    }
    tourReturnFocus?.focus();
  }

  async function showTourStep(index) {
    tourIndex = index;
    const step = TOUR_STEPS[index];
    $('#ursaTour').classList.remove('hidden');
    $('#ursaTour').classList.toggle('tourIntro', index === 0);
    $('#ursaTour').setAttribute('aria-modal', String(index === 0));
    $('.app').inert = index === 0;
    document.body.classList.toggle('tourActive', index > 0);
    document.body.classList.toggle('tourConfigActive', index === 1);
    $$('.tourHighlight').forEach(el => el.classList.remove('tourHighlight'));
    $('#tourTitle').textContent = step.title;
    $('#tourCopy').innerHTML = step.copy;
    $('#tourStatus').textContent = '';
    $('#tourStatus').dataset.state = '';
    $('#tourProgress').innerHTML = TOUR_STEPS.map((_, i) => `<span class="${i <= index ? 'current' : ''}"></span>`).join('');
    $('#tourBackBtn').disabled = index === 0;
    $('#tourNextBtn').textContent = step.next || 'Continue';
    if (index === 1) await openDefaultConfig();
    else { $('#settingsModal').classList.remove('open'); $('#settingsModal').setAttribute('aria-hidden', 'true'); }
    if (index === 2) {
      state.showChat = state.showRunLogs = state.showArtifacts = false;
      applyPanelVisibility();
    }
    if (index >= 3 && index <= 6) {
      if (index === 3 && state.activeSessionId) clearActiveSessionForDraft('chat_agent');
      state.showChat = true;
      if (index === 5 || index === 6) state.showRunLogs = state.showArtifacts = true;
      applyPanelVisibility();
      renderComposerAgentSelect();
    }
    state._sessionCreateMenuOpen = index === 4;
    renderSessionCreateMenu();
    if (step.target) $(step.target)?.classList.add('tourHighlight');
    if (index !== 1) $('#tourNextBtn').focus();
  }

  async function startTour() {
    tourReturnFocus = document.activeElement;
    tourPreviousSession = state.activeSessionId;
    tourPreviousDraft = $('#messageInput')?.value || '';
    tourPreparedExample = false;
    await markTourSeen();
    await showTourStep(0);
  }

  async function advanceTour() {
    if (tourBusy || defaultConfigBusy) return;
    tourBusy = true;
    $('#tourNextBtn').disabled = true;
    $('#guidedNextBtn').disabled = true;
    try {
      if (tourIndex === 1) {
        if (!validateGuidedConfigStep()) return;
        if (guidedConfigStep < 2) { showGuidedConfigStep(guidedConfigStep + 1); return; }
        $('#tourStatus').dataset.state = '';
        $('#tourStatus').textContent = 'Testing your connection, then saving your defaults…';
        if (JSON.stringify(collectDefaultConfig()) !== lastConfigTest && !(await testDefaultConfig('chat'))) {
          $('#tourStatus').dataset.state = 'failed';
          $('#tourStatus').textContent = 'Could not confirm the connection. See the result beside Test connection. Your defaults have not been saved.';
          return;
        }
        if (!(await saveDefaultConfig())) {
          $('#tourStatus').dataset.state = 'failed';
          $('#tourStatus').textContent = 'The connection passed, but the defaults could not be saved.';
          $('#guidedConfigError').textContent = $('#defaultConfigStatus').textContent;
          return;
        }
        $$('#settingsModal input[type="password"]').forEach(input => { input.value = ''; });
        await showTourStep(2);
        $('#tourStatus').dataset.state = 'passed';
        $('#tourStatus').textContent = 'Passed — your model connected and your defaults are saved.';
        return;
      }
      if (tourIndex === 6) {
        if (!tourPreparedExample) {
          clearActiveSessionForDraft('chat_agent');
          openComposerDraft('chat_agent', FIRST_TASK);
          tourPreparedExample = true;
          $('#tourCopy').innerHTML = '<p>Your example is ready with <strong>Chat</strong> selected. Press <strong>Send</strong>; URSA will ask where to work.</p><p>Pick a workspace folder for this session, or use a temporary one. Temporary files are removed when the session is deleted or the dashboard stops, so download anything you want to keep.</p><p>Watch Logs, then open <strong>prime_spacings.png</strong> in <strong>Artifacts</strong>. Use <strong>Set workspace</strong> there to change the working folder later.</p>';
          $('#tourNextBtn').textContent = 'Explore environments';
          return;
        }
      }
      if (tourIndex === TOUR_STEPS.length - 1) { tourBusy = false; closeTour(); }
      else await showTourStep(tourIndex + 1);
    } catch (error) { $('#tourStatus').textContent = error.message; }
    finally { tourBusy = false; $('#tourNextBtn').disabled = false; $('#guidedNextBtn').disabled = false; }
  }

  function goBackInTour() {
    if (tourBusy || defaultConfigBusy || tourIndex <= 0) return;
    $('#tourStatus').textContent = '';
    if (tourIndex === 1 && guidedConfigStep > 0) showGuidedConfigStep(guidedConfigStep - 1);
    else showTourStep(tourIndex - 1);
  }

  function setupOnboarding() {
    const draftPane = $('.settingsPane[data-settings-pane="defaults"]');
    for (const event of ['input', 'change']) draftPane.addEventListener(event, invalidateConfigTests);
    $('#guidedProvider').onchange = () => { selectGuidedProvider(); invalidateConfigTests(); };
    for (const [id, field] of [['#guidedBaseUrl', 'base_url'], ['#guidedProviderName', 'name'], ['#guidedApiType', 'model_provider'], ['#guidedKeySource', 'credential_mode'], ['#guidedApiKey', 'api_key'], ['#guidedKeyEnv', 'api_key_env']]) {
      $(id).addEventListener($(id).tagName === 'SELECT' ? 'change' : 'input', () => syncGuidedField(id, field));
    }
    $('#guidedModel').oninput = () => { $('#configLlmModel').value = $('#guidedModel').value; };
    $('#guidedFindModelsBtn').onclick = () => findConfigModels('chat');
    $('#guidedTestBtn').onclick = () => { if (validateGuidedConfigStep()) testDefaultConfig('chat'); };
    $('#guidedNextBtn').onclick = event => { event.stopPropagation(); advanceTour(); };
    $('#guidedBackBtn').onclick = event => { event.stopPropagation(); goBackInTour(); };
    const cancelSettings = $('#settingsBackdrop').onclick;
    $('#settingsBackdrop').onclick = () => { if (tourIndex === 1) closeTour(); else cancelSettings(); };
    // On compact screens reserve space below the dashboard/dialogs rather than
    // covering artifacts with the guide. Desktop keeps the guide in the left rail.
    new ResizeObserver(() => {
      document.body.style.setProperty('--tour-dock-height', `${$('#ursaTour').getBoundingClientRect().height + 24}px`);
    }).observe($('#ursaTour'));
    $('#reloadDefaultConfigBtn').onclick = () => {
      if (confirm('Reload defaults and discard edits in this config form?')) loadDefaultConfig();
    };
    $('#addConfigProviderBtn').onclick = () => {
      if (!defaultConfigDraft) return;
      const current = collectDefaultConfig();
      // Preserve entered keys and model edits when adding another provider.
      defaultConfigDraft.providers = current.providers.map((provider, index) => ({...defaultConfigDraft.providers[index], ...provider}));
      defaultConfigDraft.providers.push({name:'', base_url:'', credential_mode:'keyring', _new:true});
      renderConfigProviders();
      for (const [index, provider] of defaultConfigDraft.providers.entries()) {
        const card = $$('.configProviderCard')[index];
        $('[data-field="api_key"]', card).value = provider.api_key || '';
      }
      $$('.configProviderCard').at(-1)?.scrollIntoView({block:'nearest'});
    };
    $('#testConfigLlmBtn').onclick = () => testDefaultConfig('chat');
    $('#testConfigEmbeddingBtn').onclick = () => testDefaultConfig('embedding');
    $('#startWalkthroughBtn').onclick = () => startTour().catch(error => alert(error.message));
    $('#tourNextBtn').onclick = event => { event.stopPropagation(); advanceTour(); };
    $('#tourBackBtn').onclick = event => { event.stopPropagation(); goBackInTour(); };
    $('#tourDismissBtn').onclick = closeTour;
    $('#tourSkipBtn').onclick = closeTour;
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && tourIndex >= 0 && !tourBusy && !$('#workspaceChoiceModal').classList.contains('open')) { event.preventDefault(); closeTour(); }
    });
    if (!state.settings?.ui?.walkthrough_seen) startTour().catch(error => console.warn('Walkthrough unavailable:', error.message));
  }
"""
