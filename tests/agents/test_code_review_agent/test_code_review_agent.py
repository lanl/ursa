from typing import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class _ToolReadyFakeChatModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


def test_code_review_run_delegates_to_invoke(tmp_path, monkeypatch):
    from ursa.agents.code_review_agent import CodeReviewAgent

    (tmp_path / "main.py").write_text("print('hello')\n", encoding="utf-8")

    chat_model = _ToolReadyFakeChatModel(messages=_message_stream("ok"))
    agent = CodeReviewAgent(llm=chat_model, workspace=tmp_path)
    captured = {}

    def fake_invoke(inputs=None, **kwargs):
        captured["inputs"] = inputs
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(agent, "invoke", fake_invoke)

    result = agent.run("Review the code", tmp_path)

    assert result == {"status": "ok"}
    assert captured["inputs"]["project_prompt"] == "Review the code"
    assert captured["inputs"]["code_files"] == ["main.py"]
    assert (
        captured["kwargs"]["config"]["configurable"]["thread_id"]
        == agent.thread_id
    )
