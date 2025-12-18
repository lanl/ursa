from pathlib import Path
from ursa.agents.acquisition_agents import ArxivAgent
from ursa.observability.timing import render_session_summary


async def test_arxiv_agent(chat_model, tmpdir: Path):
    agent = ArxivAgent(
        llm=chat_model,
        database_path=str(tmpdir / "papers"),
        summaries_path=str(tmpdir / "summaries"),
        vectorstore_path=str(tmpdir / "vectors"),
    )

    result = await agent.ainvoke({
        "context": "What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
        "query": "Experimental Constraints on neutron star radius",
    })
    print(result)
    print(agent.build_graph())
    render_session_summary(agent.thread_id)
    assert "final_summary" in result
