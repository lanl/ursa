from pathlib import Path

from ursa.agents.acquisition_agents import ArxivAgent


async def test_agent(chat_model, tmpdir):
    tmpdir_path = Path(tmpdir)
    agent = ArxivAgent(
        llm=chat_model,
        database_path=str(tmpdir_path / "papers"),
        summaries_path=str(tmpdir_path / "summaries"),
        vectorstore_path=str(tmpdir_path / "vectors"),
        workspace=tmpdir,
    )
    input = {
        "context": "What are the constraints on the neutron star radius and what uncertainties are there on the constraints?",
        "query": "Experimental Constraints on neutron star radius",
    }

    result = await agent.ainvoke(input)
    assert "final_summary" in result
