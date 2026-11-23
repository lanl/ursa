"""DSIAgent must send its per-question context with every model request."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver

from tests.agents.utils import RecordingChatModel
from tests.conftest import extraskip


def _ask(agent, question, thread_id):
    agent.invoke(
        agent.format_query(question),
        config={"configurable": {"thread_id": thread_id}},
    )


def test_dsi_context_reaches_every_question_of_a_thread(tmp_path):
    extraskip("dsi")
    from ursa.agents.dsi_agent import DSIAgent

    llm = RecordingChatModel()
    agent = DSIAgent(llm=llm, workspace=tmp_path, checkpointer=InMemorySaver())

    agent.db_schema = "SCHEMA_ONE"
    _ask(agent, "first question", "t")
    agent.db_schema = "SCHEMA_TWO"
    _ask(agent, "second question", "t")

    assert len(llm.calls) == 2
    for request, schema in zip(llm.calls, ["SCHEMA_ONE", "SCHEMA_TWO"]):
        systems = [m for m in request if isinstance(m, SystemMessage)]
        assert len(systems) == 1, "exactly one system message per request"
        assert request[0] is systems[0], "system message leads the request"
        assert agent.prompt in systems[0].content
        assert schema in systems[0].content
        assert isinstance(request[-1], HumanMessage)
    assert "SCHEMA_ONE" not in llm.calls[1][0].content


def test_dsi_context_persists_across_a_tool_loop(tmp_path):
    extraskip("dsi")
    from langchain_core.messages import AIMessage

    from tests.agents.utils import ScriptedRecordingChatModel
    from ursa.agents.dsi_agent import DSIAgent

    llm = ScriptedRecordingChatModel(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"filename": "x.txt"},
                        "id": "call_1",
                    }
                ],
            )
        ]
    )
    agent = DSIAgent(llm=llm, workspace=tmp_path)
    agent.db_schema = "SCHEMA_ONE"
    agent.invoke(agent.format_query("read x"))

    assert len(llm.calls) >= 2
    for request in llm.calls:
        assert "SCHEMA_ONE" in request[0].content
        assert sum(isinstance(m, SystemMessage) for m in request) == 1
