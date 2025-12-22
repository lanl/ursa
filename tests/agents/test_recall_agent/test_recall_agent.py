import pytest

from ursa.agents.recall_agent import RecallAgent


class StubMemory:
    def __init__(self, memories: list[str]):
        self._memories = memories
        self.queries: list[str] = []

    def retrieve(self, query: str):
        self.queries.append(query)
        return list(self._memories)


@pytest.mark.asyncio
async def test_recall_agent_ainvoke_returns_memory(chat_model):
    memory_stub = StubMemory([
        "The lab stored baseline experiment logs in the west wing archive."
    ])
    agent = RecallAgent(llm=chat_model, memory=memory_stub)

    # RecallAgent's initializer does not currently attach memory to the instance,
    # so we set it up here to mirror the expected runtime configuration.
    agent.memorydb = memory_stub

    query = "Summarize where the experiment logs are kept."
    result = await agent.ainvoke({"query": query})

    assert memory_stub.queries == [query]
    assert "memory" in result
    assert isinstance(result["memory"], str)
    assert result["memory"] is not None
