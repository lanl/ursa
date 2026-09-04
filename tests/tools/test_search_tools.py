from types import SimpleNamespace

import pytest

import ursa.tools.search_tools as search_tools


async def test_web_search_dispatches_progress_asynchronously(
    tmp_path, monkeypatch
):
    emitted = []

    class FakeEvents:
        def emit(self, *_args, **_kwargs):
            pytest.fail(
                "synchronous event dispatch can deadlock the Textual UI"
            )

        async def aemit(self, message, *, stage, **payload):
            emitted.append((message, stage, payload))

    class FakeWebSearchAgent:
        def __init__(self, **_kwargs):
            pass

        async def ainvoke(self, **_kwargs):
            return {"final_summary": "A useful result"}

    monkeypatch.setattr(
        search_tools.ToolEvents,
        "from_runtime",
        lambda *_args, **_kwargs: FakeEvents(),
    )
    monkeypatch.setattr(search_tools, "WebSearchAgent", FakeWebSearchAgent)
    runtime = SimpleNamespace(
        context=SimpleNamespace(llm=object(), den=tmp_path)
    )

    result = await search_tools.run_web_search.coroutine(
        prompt="Find evidence",
        query="ursa",
        runtime=runtime,
    )

    assert result == "[Web Search Agent Output]:\n A useful result"
    assert emitted == [
        (
            "Searching Web",
            "search",
            {"query": "ursa", "max_results": 3},
        ),
        (
            "Web search complete",
            "search_result",
            {"query": "ursa", "result_chars": 15},
        ),
    ]
