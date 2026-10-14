"""Tests for the dashboard-wide web-tool opt-in (`--use-web`).

The dashboard mirrors the CLI/TUI `ursa --use-web` flag. The flag is transported
to the app (and to worker subprocesses, which inherit the environment) via the
``URSA_DASHBOARD_USE_WEB`` environment variable, which remains supported
directly for backward compatibility.
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

import ursa_dashboard.main as dashboard_main
from ursa.agents.deep_review_agent import DeepReviewAgent
from ursa_dashboard.registry import REGISTRY

# Truthy spelling accepted by ``create_app`` when reading the env var.
APP_TRUTHY = {"1", "true", "yes", "on"}


def _invoke(argv: list[str], monkeypatch) -> str | None:
    """Run the dashboard CLI, stubbing uvicorn, and report the env var it set.

    Returns the value of ``URSA_DASHBOARD_USE_WEB`` at the moment ``uvicorn.run``
    would have been called, which is exactly what ``create_app`` later reads.
    """
    captured: dict[str, str | None] = {}

    def fake_run(*args, **kwargs):
        captured["env"] = os.environ.get("URSA_DASHBOARD_USE_WEB")

    # `main()` does `import uvicorn` inside its body, so patch the module itself.
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    # Avoid opening a browser during tests.
    monkeypatch.setattr(dashboard_main.webbrowser, "open", lambda *a, **k: None)

    result = CliRunner().invoke(dashboard_main.app, argv)
    assert result.exit_code == 0, result.output
    assert "env" in captured, "uvicorn.run was never reached"
    return captured["env"]


def test_use_web_flag_enables_dashboard_web_opt_in(monkeypatch):
    """`--use-web` sets the env var that create_app reads."""
    monkeypatch.delenv("URSA_DASHBOARD_USE_WEB", raising=False)
    value = _invoke(["--use-web"], monkeypatch)
    assert value == "1"
    assert str(value).strip().lower() in APP_TRUTHY


def test_without_flag_web_tools_stay_opt_out(monkeypatch):
    """Web tools remain off by default; the flag must be explicit."""
    monkeypatch.delenv("URSA_DASHBOARD_USE_WEB", raising=False)
    value = _invoke([], monkeypatch)
    assert value is None


@pytest.mark.parametrize("raw", sorted(APP_TRUTHY) + ["TRUE", "On", "YES"])
def test_env_var_alone_still_enables_web_tools(raw, monkeypatch):
    """Backward compatibility: URSA_DASHBOARD_USE_WEB works without the flag.

    The flag deliberately does not use typer's ``envvar=``, so an unparsable
    value cannot abort startup; the value is passed through untouched to the
    app's own lenient parser.
    """
    monkeypatch.setenv("URSA_DASHBOARD_USE_WEB", raw)
    value = _invoke([], monkeypatch)
    assert value == raw
    assert str(value).strip().lower() in APP_TRUTHY


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "garbage", "2"])
def test_non_truthy_env_values_do_not_enable_web_tools(raw, monkeypatch):
    """Falsy/invalid env values leave web tools off and never crash startup."""
    monkeypatch.setenv("URSA_DASHBOARD_USE_WEB", raw)
    value = _invoke([], monkeypatch)
    # Passed through unchanged, and not interpreted as enabling web tools.
    assert value == raw
    assert str(value).strip().lower() not in APP_TRUTHY


def test_flag_overrides_falsy_env_value(monkeypatch):
    """An explicit `--use-web` wins over a falsy env var."""
    monkeypatch.setenv("URSA_DASHBOARD_USE_WEB", "0")
    value = _invoke(["--use-web"], monkeypatch)
    assert value == "1"
    assert str(value).strip().lower() in APP_TRUTHY


def test_deep_review_agent_participates_in_web_opt_in():
    """Deep Review must follow the dashboard-wide opt-in, like the CLI.

    ``ursa.cli.config`` fans `use_web` out to chat, execute, deep_review, and
    prompt. Deep Review was previously missing from the dashboard's opt-in set,
    so `--use-web` silently left it without web tools.
    """
    import re

    from ursa_dashboard import app as app_module

    source = re.search(
        r"web_opt_in_agent_ids = \{(.*?)\}",
        open(app_module.__file__).read(),
        re.S,
    )
    assert source is not None, "web_opt_in_agent_ids not found"
    agent_ids = set(re.findall(r'"(\w+)"', source.group(1)))

    assert "deep_review_agent" in agent_ids
    # Parity with the CLI fan-out list in ursa.cli.config.
    for expected in ("chat_agent", "execution_agent", "prompting_agent"):
        assert expected in agent_ids

    # Every id must name a real registered agent, or the opt-in silently no-ops.
    for agent_id in agent_ids:
        assert agent_id in REGISTRY, f"{agent_id} is not a registered agent_id"


def test_deep_review_agent_accepts_use_web_kwarg(chat_model, tmpdir):
    """The opt-in only works if DeepReviewAgent honors a `use_web` kwarg.

    Asserted behaviorally rather than by signature: ``BaseAgent.__init_subclass__``
    rewraps ``__init__`` without ``functools.wraps``, so the introspected
    signature is ``(self, *args, **kwargs)``.
    """
    no_web = DeepReviewAgent(llm=chat_model, workspace=tmpdir, use_web=False)
    with_web = DeepReviewAgent(llm=chat_model, workspace=tmpdir, use_web=True)

    def tool_names(agent):
        return {getattr(t, "name", str(t)) for t in agent.tools}

    extra = tool_names(with_web) - tool_names(no_web)
    assert extra, "use_web=True added no tools"
    # Web opt-in should bring in the search tools the dashboard advertises.
    assert any("search" in name for name in extra), extra
