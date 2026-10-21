import json
import shutil
import subprocess
from pathlib import Path

import pytest


def test_dashboard_skills_command_uses_catalog_without_submitting_message():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to exercise dashboard JavaScript")
    source = (
        Path(__file__).parents[2] / "src/ursa_dashboard/app.py"
    ).read_text()
    start = source.index("  async function sendMessage() {")
    end = source.index("  async function cancelActiveRun()", start)
    command = source[start:end]
    script = r"""
const assert = require('node:assert/strict');
const html = HTML_SOURCE;
const calls = [];
const panelContainer = {children: [], scrollHeight: 10,
  querySelectorAll() { return this.children.map(child => ({remove: () => { this.children = this.children.filter(x => x !== child); }})); },
  appendChild(child) { this.children.push(child); }
};
const textarea = {value: '/skills'};
const state = {activeSessionId: 'session-1'};
const document = {createElement(tag) { return {tag, style: {}}; }};
function $(selector) {
  assert.ok(html.includes(`id="${selector.slice(1)}"`), `Missing DOM element: ${selector}`);
  if (selector === '#sessionMessages') return panelContainer;
  if (selector === '#messageInput') return textarea;
  throw new Error(`Unexpected selector: ${selector}`);
}
async function ensureSessionWorkspaceConfigured() { return true; }
async function api(method, path) {
  calls.push([method, path]);
  assert.equal(method, 'GET');
  assert.equal(path, '/sessions/session-1/skills');
  return {text: 'Available skills: lab-report <literal text>'};
}
function showSendError(message) { throw new Error(message); }
COMMAND_SOURCE
(async () => {
  await sendMessage();
  assert.equal(calls.length, 1);
  assert.equal(textarea.value, '');
  assert.equal(panelContainer.children[0].textContent, 'Available skills: lab-report <literal text>');
  textarea.value = '/skills';
  await sendMessage();
  assert.equal(panelContainer.children.length, 1);
})().catch(error => { console.error(error); process.exitCode = 1; });
"""
    script = script.replace("HTML_SOURCE", json.dumps(source)).replace(
        "COMMAND_SOURCE", command
    )
    result = subprocess.run(
        [node, "-e", script], capture_output=True, text=True, timeout=15
    )
    assert result.returncode == 0, result.stderr
