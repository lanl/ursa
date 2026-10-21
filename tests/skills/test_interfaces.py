from types import SimpleNamespace

import pytest
from textual.widgets import Markdown

from tests.agents.utils import RecordingChatModel
from tests.cli._app_fakes import FakeHITL
from ursa.agents.chat_agent import ChatAgent
from ursa.cli.config import UrsaConfig
from ursa.cli.runtime import HITL, AgentHITL
from ursa.cli.tui.app import UrsaTextualApp
from ursa.cli.tui.widgets import InformationScreen, PromptArea
from ursa_dashboard.adapters import BaseAgentInProcessAdapter, RunContext


@pytest.mark.parametrize("direct", [True, False])
async def test_tui_skills_listing_is_local(tmp_path, make_skill, direct):
    folder = make_skill(tmp_path)
    hitl = FakeHITL(tmp_path)
    app = UrsaTextualApp(hitl)
    async with app.run_test(size=(100, 36)) as pilot:
        if direct:
            app.query_one(PromptArea).load_text("/skills")
            await pilot.press("enter")
        else:
            assert any(
                "skills" in choice for choice in app._hotlist_candidates("/")
            )
            await app._show_command("skills")
        await pilot.pause()
        assert isinstance(app.screen, InformationScreen)
        content = str(app.screen.query_one(Markdown)._markdown)
        assert "lab-report" in content
        assert str(folder / "SKILL.md") in content
        assert not hitl.calls


@pytest.mark.parametrize("explicit", [True, False])
async def test_cli_scope_survives_config_resolution(
    tmp_path, monkeypatch, make_skill, explicit
):
    home = tmp_path / "home"
    project = tmp_path / "project"
    make_skill(home, "home-only")
    make_skill(project, "project-only")
    monkeypatch.chdir(project)
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr(
        "ursa.cli.config.init_chat_model", lambda **kwargs: RecordingChatModel()
    )
    monkeypatch.setattr(
        "ursa.cli.runtime.validate_model_provider", lambda *_: None
    )
    monkeypatch.setattr("ursa.cli.runtime.start_mcp_client", lambda *_: None)
    config = UrsaConfig(**({"workspace": "."} if explicit else {})).resolve()
    hitl = HITL(config)
    try:
        names = [skill.name for skill in hitl.skill_scope.discover().skills]
        assert names == (
            ["project-only"] if explicit else ["home-only", "project-only"]
        )
        wrapper = await hitl.get_agent("chat")
        assert wrapper._agent.skill_scope == hitl.skill_scope
    finally:
        await hitl.aclose()


async def test_tui_history_does_not_activate_old_reference(
    tmp_path, make_skill
):
    make_skill(tmp_path)
    model = RecordingChatModel()
    wrapper = AgentHITL(ChatAgent)
    await wrapper.instantiate(
        llm=model, workspace=tmp_path, enable_metrics=False
    )
    await wrapper(
        "New task", last_agent_result="$lab-report was mentioned earlier"
    )
    assert "SKILL_BODY_SENTINEL" not in str(model.calls[-1])


async def test_dashboard_adapter_preserves_current_request(
    tmp_path, make_skill
):
    make_skill(tmp_path)
    model = RecordingChatModel()
    adapter = BaseAgentInProcessAdapter(
        lambda workspace, inputs: ChatAgent(
            llm=model, workspace=workspace, enable_metrics=False
        ),
        supports_streaming=False,
    )
    await adapter.ainvoke(
        ctx=RunContext(
            "run-1", "chat_agent", tmp_path, skill_request="new task"
        ),
        inputs="User: $lab-report old task\nUser: new task",
        sink=SimpleNamespace(emit=lambda event: None),
    )
    assert "SKILL_BODY_SENTINEL" not in str(model.calls[-1])
    await adapter.ainvoke(
        ctx=RunContext(
            "run-2",
            "chat_agent",
            tmp_path,
            skill_request="$lab-report new task",
        ),
        inputs="User: $lab-report new task",
        sink=SimpleNamespace(emit=lambda event: None),
    )
    assert "SKILL_BODY_SENTINEL" in str(model.calls[-1])
