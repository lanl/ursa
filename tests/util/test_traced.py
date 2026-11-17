import subprocess

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from ursa.util.traced import TracedChatOllama, TracedChatOpenAI

OLLAMA_TEST_MODEL = "nemotron-3-nano:4b"


def test_traced_chat_openai(monkeypatch):
    responses = iter([
        AIMessage(content=[{"type": "text", "text": "ok"}]),
        AIMessage(content=[{"type": "reasoning", "reasoning": "because"}]),
    ])

    def fake_generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=next(responses))])

    monkeypatch.setattr(TracedChatOpenAI, "_generate", fake_generate)

    llm = TracedChatOpenAI(
        model="gpt-5.4-nano-2026-03-17",
        api_key="test-api-key",
        reasoning={"effort": "medium", "summary": "auto"},
    )
    llm.invoke("Hi")  # reasoning not activated
    llm.invoke(query := "Who is the greatest pianist of all time?")  # reasoning
    assert len(llm.messages) == 4
    assert llm.messages[0].content == "Hi"
    assert llm.messages[1].content_blocks[0]["type"] == "text"
    assert llm.messages[2].content == query
    assert llm.messages[3].content_blocks[0]["type"] == "reasoning"


def has_ollama_model(model_name: str):
    """Probe through the CLI so no module-level HTTP socket is left open."""
    try:
        result = subprocess.run(
            ["ollama", "show", model_name],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


@pytest.mark.skipif(
    not has_ollama_model(OLLAMA_TEST_MODEL),
    reason=f"{OLLAMA_TEST_MODEL} not available",
)
def test_traced_chat_ollama():
    llm = TracedChatOllama(model=OLLAMA_TEST_MODEL, reasoning=True)
    llm.invoke("Hi")
    llm.invoke(query := "Who is the greatest pianist of all time?")
    assert len(llm.messages) == 4
    assert llm.messages[0].content == "Hi"
    assert llm.messages[1].content_blocks[0]["type"] == "reasoning"
    assert llm.messages[2].content == query
    assert llm.messages[3].content_blocks[0]["type"] == "reasoning"
