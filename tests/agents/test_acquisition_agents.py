import asyncio

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Send

import ursa.agents.acquisition_agents as acquisition_module
from ursa.agents.acquisition_agents import BaseAcquisitionAgent


class FakeAcquisitionAgent(BaseAcquisitionAgent):
    def __init__(self, *args, hits=None, **kwargs):
        self.hits = list(hits or [])
        self.active = 0
        self.maximum_active = 0
        super().__init__(*args, **kwargs)

    def _search(self, query):
        return self.hits

    async def _asearch(self, query):
        return self.hits

    def _materialize(self, hit):
        raise NotImplementedError

    async def _amaterialize(self, hit):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        try:
            await asyncio.sleep(hit.get("delay", 0))
            if error := hit.get("error"):
                raise RuntimeError(error)
            return {
                "id": hit["id"],
                "title": hit.get("title", ""),
                "url": hit.get("url", ""),
                "full_text": hit.get("text", hit["id"]),
            }
        finally:
            self.active -= 1

    def _id(self, hit):
        return hit["id"]

    def _citation(self, item):
        return f"source:{item['id']}"


async def test_langgraph_fan_out_is_bounded_ordered_and_failure_isolated(
    chat_model, tmp_path
):
    hits = [
        {"id": "first", "delay": 0.03},
        {
            "id": "bad",
            "title": "Broken",
            "url": "https://bad.test",
            "delay": 0.01,
            "error": "unavailable",
        },
        {"id": "last"},
    ]
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=hits,
        summarize=False,
        num_threads=2,
        workspace=tmp_path,
    )

    result = await agent.ainvoke({"query": "ursa", "context": "compare"})

    assert agent.maximum_active == 2
    assert [item["id"] for item in result["items"]] == [
        "first",
        "bad",
        "last",
    ]
    assert result["items"][1] == {
        "id": "bad",
        "title": "Broken",
        "url": "https://bad.test",
        "full_text": "[Error: unavailable]",
    }


async def test_langgraph_max_concurrency_replaces_local_semaphores(
    chat_model, tmp_path
):
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=[{"id": "a"}, {"id": "b"}],
        summarize=False,
        num_threads=1,
        workspace=tmp_path,
    )

    await agent.ainvoke({"query": "ursa", "context": "compare"})

    assert agent.maximum_active == 1
    assert agent.build_config()["max_concurrency"] == 1
    assert agent.num_threads == 1


async def test_cached_sources_are_fanned_out_filtered_and_sorted(
    chat_model, tmp_path
):
    database = tmp_path / "database"
    database.mkdir()
    (database / "b.html").write_text("second")
    (database / "a.txt").write_text("first")
    (database / "ignored.json").write_text("ignored")
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        summarize=False,
        download=False,
        database_path="database",
        workspace=tmp_path,
    )

    result = await agent.ainvoke({"query": "unused", "context": "cached"})

    assert [item["id"] for item in result["items"]] == ["a", "b"]
    assert [item["full_text"] for item in result["items"]] == [
        "first",
        "second",
    ]


async def test_langgraph_map_reduce_preserves_source_order(
    chat_model, tmp_path, monkeypatch
):
    active = 0
    maximum_active = 0
    aggregate_input = None

    class FakeChain:
        def __or__(self, _other):
            return self

        async def ainvoke(self, values, config=None):
            nonlocal active, maximum_active, aggregate_input
            if "retrieved_content" in values:
                active += 1
                maximum_active = max(maximum_active, active)
                try:
                    text = values["retrieved_content"]
                    await asyncio.sleep(
                        {"alpha": 0.03, "beta": 0.01, "gamma": 0}[text]
                    )
                    if text == "beta":
                        raise RuntimeError("model unavailable")
                    return f"summary:{text}"
                finally:
                    active -= 1
            aggregate_input = values["Summaries"]
            return "reduced answer"

    chain = FakeChain()
    monkeypatch.setattr(
        acquisition_module.ChatPromptTemplate,
        "from_template",
        lambda _template: chain,
    )
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=[
            {"id": "a", "text": "alpha"},
            {"id": "b", "text": "beta"},
            {"id": "c", "text": "gamma"},
        ],
        num_threads=2,
        workspace=tmp_path,
    )

    result = await agent.ainvoke({"query": "ursa", "context": "compare"})

    assert maximum_active == 2
    assert result["summaries"] == [
        "summary:alpha",
        "[Error summarizing item b: model unavailable]",
        "summary:gamma",
    ]
    expected_combined = (
        "\n\n[1] source:a\n\nSummary:\nsummary:alpha"
        "\n\n----------------------------------------\n\n"
        "[2] source:b\n\nSummary:\n"
        "[Error summarizing item b: model unavailable]"
        "\n\n----------------------------------------\n\n"
        "[3] source:c\n\nSummary:\nsummary:gamma"
    )
    assert aggregate_input == expected_combined
    assert result["final_summary"] == "reduced answer"


def test_graph_uses_direct_nodes_and_send_router(chat_model, tmp_path):
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=[{"id": "a"}, {"id": "b"}],
        workspace=tmp_path,
    )
    graph = agent.build_graph()

    assert set(graph.nodes) == {
        "_search_query",
        "_search_sources",
        "_process_source",
        "_reduce_sources",
    }
    routed = agent._fan_out_sources({
        "source_tasks": [
            {"index": 0, "context": "test", "hit": {"id": "a"}},
            {"index": 1, "context": "test", "hit": {"id": "b"}},
        ]
    })
    assert isinstance(routed, list)
    assert all(isinstance(command, Send) for command in routed)
    assert [command.node for command in routed] == [
        "_process_source",
        "_process_source",
    ]


async def test_sync_source_adapter_does_not_block_event_loop(
    chat_model, tmp_path
):
    agent = FakeAcquisitionAgent(llm=chat_model, workspace=tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    main_loop = asyncio.get_running_loop()

    def search(_query):
        main_loop.call_soon_threadsafe(started.set)
        asyncio.run_coroutine_threadsafe(release.wait(), main_loop).result()
        return []

    agent._search = search
    task = asyncio.create_task(BaseAcquisitionAgent._asearch(agent, "ursa"))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        assert not task.done()
    finally:
        release.set()
    assert await asyncio.wait_for(task, timeout=1) == []


async def test_existing_query_node_does_not_reemit_reducer_state(
    chat_model, tmp_path
):
    agent = FakeAcquisitionAgent(llm=chat_model, workspace=tmp_path)
    state = {
        "query": "ursa",
        "context": "compare",
        "processed_sources": [
            {"index": 0, "item": {"id": "a"}, "summary": None}
        ],
    }

    assert await agent._search_query(state) == {}
    assert len(state["processed_sources"]) == 1


async def test_generated_query_normalizes_structured_message_content(
    chat_model, tmp_path
):
    agent = FakeAcquisitionAgent(llm=chat_model, workspace=tmp_path)

    class StructuredModel:
        async def ainvoke(self, _prompt):
            return AIMessage(
                content=[{"type": "text", "text": "ursa web search"}]
            )

    agent.llm = StructuredModel()

    assert await agent._search_query({"context": "research URSA"}) == {
        "query": "ursa web search"
    }


async def test_summarize_false_skips_rag(chat_model, tmp_path, monkeypatch):
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=[{"id": "a"}],
        summarize=False,
        rag_embedding=object(),
        workspace=tmp_path,
    )

    async def fail_if_called(_state):
        raise AssertionError(
            "RAG should not run when summarization is disabled"
        )

    monkeypatch.setattr(agent, "_arag_node", fail_if_called)

    result = await agent.ainvoke({"query": "ursa", "context": "compare"})

    assert result["items"][0]["id"] == "a"
    assert "final_summary" not in result


async def test_summary_cache_write_failure_does_not_abort_source(
    chat_model, tmp_path, monkeypatch
):
    class FakeChain:
        def __or__(self, _other):
            return self

        async def ainvoke(self, _values, config=None):
            return "useful summary"

    monkeypatch.setattr(
        acquisition_module.ChatPromptTemplate,
        "from_template",
        lambda _template: FakeChain(),
    )
    agent = FakeAcquisitionAgent(llm=chat_model, workspace=tmp_path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only cache")

    monkeypatch.setattr(acquisition_module.Path, "write_text", fail_write)

    summary = await agent._summarize_source(
        {"id": "a", "full_text": "content"}, 0, "compare"
    )

    assert summary == "useful summary"


def test_synchronous_stream_is_explicitly_unsupported(chat_model, tmp_path):
    agent = FakeAcquisitionAgent(
        llm=chat_model,
        hits=[{"id": "a"}],
        summarize=False,
        workspace=tmp_path,
    )

    with pytest.raises(RuntimeError, match="do not support synchronous"):
        list(agent.stream({"query": "ursa", "context": "compare"}))


async def test_aggregate_cache_write_failure_keeps_model_answer(
    chat_model, tmp_path, monkeypatch
):
    class FakeChain:
        def __or__(self, _other):
            return self

        async def ainvoke(self, _values, config=None):
            return "reduced answer"

    monkeypatch.setattr(
        acquisition_module.ChatPromptTemplate,
        "from_template",
        lambda _template: FakeChain(),
    )
    agent = FakeAcquisitionAgent(llm=chat_model, workspace=tmp_path)

    def fail_write(*_args, **_kwargs):
        raise OSError("read-only cache")

    monkeypatch.setattr(acquisition_module.Path, "write_text", fail_write)

    answer = await agent._aggregate_sources(
        [{"id": "a"}], ["source summary"], "compare"
    )

    assert answer == "reduced answer"
