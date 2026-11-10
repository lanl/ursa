"""Cross-model history portability (model switch mid-conversation).

Regression for: Claude reached through an OpenAI-compatible gateway with the
Responses API stores Responses-native items (``reasoning`` with
``encrypted_content``, ``function_call`` with ``status``) in history. Switching
the same thread to another model then failed:

* ChatOpenAI (Responses) replays those items verbatim -> OpenAI 400
  ``Unknown parameter: 'input[N].status'``;
* ChatGoogleGenerativeAI raises ``KeyError: 'reasoning'``;
* the summarizer rendered the signatures as text (1.5M-token request).
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.agents.utils import RecordingChatModel
from ursa.agents.base import _message_text, _render_for_summary
from ursa.agents.chat_agent import BasicChatAgent
from ursa.util.message_portability import is_foreign, make_portable

SIGNATURE = (
    "Q0FRUzlnWUtFZ2dURUFFWUFqZ0JRZ2gwYUdsdWEybHVaeElNMXQyY1o3dGdCckFH" * 20
)
CLAUDE = "aws-gov.claude-opus-5-5"


def _gateway_claude_turn(call_id: str = "toolu_01abc") -> AIMessage:
    """Shape copied from a real AIPortal checkpoint (Claude via Responses)."""
    return AIMessage(
        content=[
            {
                "id": "rs_fa198047",
                "type": "reasoning",
                "content": [
                    {
                        "type": "output_text",
                        "text": "check git",
                        "annotations": [],
                    }
                ],
                "encrypted_content": SIGNATURE,
                "status": "completed",
                "role": "assistant",
            },
            {
                "type": "text",
                "text": "Checking status.",
                "id": "msg_1",
                "annotations": [],
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": call_id,
                "name": "run_command",
                "arguments": '{"query": "git status"}',
                "status": "completed",
            },
        ],
        tool_calls=[
            {
                "name": "run_command",
                "args": {"query": "git status"},
                "id": call_id,
            }
        ],
        response_metadata={
            "model_provider": "openai",
            "model_name": CLAUDE,
            "id": "resp_1",
        },
        id="ai-1",
    )


def _history(call_id: str = "toolu_01abc") -> list:
    return [
        HumanMessage("what changed?", id="h-1"),
        _gateway_claude_turn(call_id),
        ToolMessage("M base.py", tool_call_id=call_id, id="t-1"),
        HumanMessage("now continue", id="h-2"),
    ]


class _NamedModel(RecordingChatModel):
    model_name: str = "gpt-5.4"


@pytest.mark.parametrize(
    ("origin", "target", "foreign"),
    [
        ("gpt-5.4-2026-03-05", "gpt-5.4", False),  # dated snapshot of the alias
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5", False),
        ("models/gemini-3-pro-preview", "gemini-3-pro-preview", False),
        (
            "us.anthropic.claude-sonnet-4-5-v1:0",
            "us.anthropic.claude-sonnet-4-5",
            False,
        ),
        ("claude-sonnet-4", "claude-sonnet-4-5", True),  # different model
        (CLAUDE, "gpt-5.4", True),
        (
            "aws-gov.claude-sonnet-5-5",
            CLAUDE,
            True,
        ),  # same gateway, other model
    ],
)
def test_origin_matching(origin, target, foreign):
    msg = AIMessage("x", response_metadata={"model_name": origin})
    assert is_foreign(msg, target) is foreign


def test_unknown_origin_is_left_alone():
    # No metadata (synthetic/legacy/summary) or unknown target -> unchanged,
    # which keeps single-model behavior and old checkpoints exactly as before.
    plain = AIMessage("hi")
    assert not is_foreign(plain, "gpt-5.4")
    assert not is_foreign(_gateway_claude_turn(), None)
    msgs = [HumanMessage("q"), plain]
    out, changed = make_portable(msgs, RecordingChatModel())
    assert not changed and out[1] is plain


def test_native_messages_pass_through_verbatim():
    msgs = _history()
    out, changed = make_portable(msgs, _NamedModel(model_name=CLAUDE))
    assert not changed
    assert all(a is b for a, b in zip(msgs, out))


def test_foreign_turn_keeps_only_text_and_tool_calls():
    out, changed = make_portable(_history(), _NamedModel())
    assert changed
    ai = out[1]
    assert ai.content == "Checking status."
    assert ai.tool_calls[0]["id"] == "toolu_01abc"
    assert ai.tool_calls[0]["args"] == {"query": "git status"}
    assert ai.id == "ai-1"  # same id -> add_messages replaces in place
    blob = json.dumps(ai.model_dump(), default=str)
    for leaked in ("encrypted_content", "rs_fa198047", '"status"', "resp_1"):
        assert leaked not in blob
    # Idempotent: the rebuilt message is never rebuilt again.
    again, changed_again = make_portable(out, _NamedModel())
    assert not changed_again and again == out


def test_reasoning_only_foreign_turn_is_dropped():
    msg = AIMessage(
        content=[
            {"type": "reasoning", "id": "rs_1", "encrypted_content": SIGNATURE}
        ],
        response_metadata={"model_name": CLAUDE},
    )
    out, changed = make_portable([HumanMessage("q"), msg], _NamedModel())
    assert changed and len(out) == 1


def test_unsafe_foreign_tool_ids_are_rewritten_consistently():
    # Kimi/vLLM-style ids break Anthropic's ^[A-Za-z0-9_-]+$ id pattern.
    out, _ = make_portable(_history("functions.run_command:0"), _NamedModel())
    new_id = out[1].tool_calls[0]["id"]
    assert (
        new_id.startswith("call_") and ":" not in new_id and len(new_id) <= 40
    )
    assert out[2].tool_call_id == new_id
    assert (
        make_portable(_history("functions.run_command:0"), _NamedModel())[0][
            1
        ].tool_calls[0]["id"]
        == new_id
    )


def test_openai_responses_payload_no_longer_replays_foreign_items(monkeypatch):
    langchain_openai = pytest.importorskip("langchain_openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    llm = langchain_openai.ChatOpenAI(model="gpt-5.4", use_responses_api=True)

    raw = json.dumps(llm._get_request_payload(_history()))
    assert "encrypted_content" in raw and '"status"' in raw  # the original bug

    fixed, _ = make_portable(_history(), llm)
    payload = json.dumps(llm._get_request_payload(fixed))
    assert "encrypted_content" not in payload and '"status"' not in payload
    assert "Checking status." in payload and "toolu_01abc" in payload


def test_google_payload_no_longer_crashes(monkeypatch):
    genai = pytest.importorskip("langchain_google_genai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    llm = genai.ChatGoogleGenerativeAI(model="gemini-3-pro-preview")
    fixed, _ = make_portable(_history(), llm)
    llm._prepare_request(fixed)  # raised KeyError: 'reasoning' before the fix


def test_summary_render_excludes_signatures_and_images():
    tool = ToolMessage(
        content=[
            {
                "type": "image",
                "base64": "iVBORw0KGgo" * 50,
                "mime_type": "image/png",
            }
        ],
        tool_call_id="toolu_01abc",
    )
    rendered = _render_for_summary([_gateway_claude_turn(), tool])
    assert SIGNATURE not in rendered and "encrypted_content" not in rendered
    assert "iVBORw0KGgo" not in rendered and "[image]" in rendered
    assert "Checking status." in rendered and "run_command" in rendered
    # The reasoning-only/function_call-only shape used to fall back to str(block).
    bare = AIMessage(content=_gateway_claude_turn().content[::2])
    assert _message_text(bare) == ""


def test_agent_switching_models_sends_portable_history(tmp_path):
    llm = _NamedModel()
    agent = BasicChatAgent(llm=llm, workspace=tmp_path)
    try:
        result = agent.invoke({"messages": _history()})
    finally:
        agent.close()
    sent = json.dumps([m.model_dump() for m in llm.calls[-1]], default=str)
    assert "encrypted_content" not in sent and "rs_fa198047" not in sent
    # The cleaned history is what persists, so later steps don't redo it.
    stored = json.dumps(
        [m.model_dump() for m in result["messages"]], default=str
    )
    assert "encrypted_content" not in stored
    assert result["messages"][-1].text == "OK"


def test_portable_flag_disables_rewrite(tmp_path):
    agent = BasicChatAgent(llm=_NamedModel(), workspace=tmp_path)
    try:
        state, overwrite = agent.prepare_messages_context(
            {"messages": _history()}, portable=False, summarize=False
        )
    finally:
        agent.close()
    assert "encrypted_content" in json.dumps(state["messages"][1].content)
