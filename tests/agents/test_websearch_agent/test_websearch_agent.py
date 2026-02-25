from typing import Iterator

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ursa.agents.acquisition_agents import WebSearchAgent


class _FakeChatModel(GenericFakeChatModel):
    pass


def _message_stream(content: str) -> Iterator[AIMessage]:
    while True:
        yield AIMessage(content=content)


async def test_websearch_agent_fetches_items_without_network_or_llm(
    monkeypatch, tmpdir
):
    query = "test scientific query"
    search_url = "https://example.com/paper"

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def text(self, query, max_results=10, backend="auto"):
            return [
                {
                    "title": "Mock Result",
                    "href": search_url,
                    "body": "Mock snippet",
                }
            ]

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    monkeypatch.setattr("ursa.agents.acquisition_agents.DDGS", FakeDDGS)
    monkeypatch.setattr(
        "ursa.agents.acquisition_agents.requests.get",
        lambda *args, **kwargs: FakeResponse(
            "<html><body><p>mock extracted content with enough length to survive dedupe threshold in extraction</p></body></html>"
        ),
    )

    chat_model = _FakeChatModel(messages=_message_stream("summary"))
    agent = WebSearchAgent(
        llm=chat_model,
        summarize=False,
        max_results=1,
        workspace=tmpdir,
    )

    result = await agent.ainvoke(
        {"query": query, "context": "summarize this query"}
    )

    assert "items" in result
    assert len(result["items"]) == 1
    assert result["items"][0]["url"] == search_url
    assert "mock extracted content" in result["items"][0]["full_text"].lower()
