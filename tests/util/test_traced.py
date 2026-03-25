import ollama
import pytest

from ursa.util.traced import TracedChatOllama, TracedChatOpenAI

OLLAMA_TEST_MODEL = "nemotron-3-nano:4b"


def test_traced_chat_openai():
    llm = TracedChatOpenAI(
        model="gpt-5.4-nano-2026-03-17",
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
    try:
        models = ollama.list()["models"]
    except Exception:
        print("ollama server is not running")
        return False

    for m in models:
        if m.model == model_name:
            return True
    return False


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
