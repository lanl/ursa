import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.runnables import RunnableLambda

from ursa.agents import LammpsAgent
from ursa.agents.lammps_agent import working


class DummyPotential:
    """Minimal stub matching the potential interface used by the agent."""

    id = "dummy-potential"

    def download_files(self, workspace) -> None:
        workspace_path = Path(workspace)
        workspace_path.mkdir(parents=True, exist_ok=True)
        (workspace_path / "dummy.potential").write_text("pair_style eam\n")

    def pair_info(self) -> str:
        return "pair_style eam\npair_coeff * * dummy.potential Cu"


@pytest.mark.asyncio
async def test_lammps_agent_runs_with_preselected_potential(
    chat_model, tmp_path: Path, monkeypatch
) -> None:
    if not working:
        pytest.skip("LAMMPS agent optional dependencies are not installed")

    agent = LammpsAgent(llm=chat_model, max_fix_attempts=1)
    agent.workspace = tmp_path
    agent.workspace.mkdir(parents=True, exist_ok=True)

    original_author_chain = agent.author_chain

    def author_with_fallback(inputs: dict) -> str:
        output = original_author_chain.invoke(inputs)
        if output and output.strip():
            return output
        return json.dumps({
            "input_script": (
                "log ./log.lammps\n"
                "# fallback script when the LLM response is empty\n"
                "write_data data.fallback\n"
            )
        })

    agent.author_chain = RunnableLambda(author_with_fallback)

    subprocess_calls = []

    def fake_run(cmd, cwd, stdout, stderr, text, check):
        subprocess_calls.append((cmd, Path(cwd)))
        return SimpleNamespace(
            returncode=0, stdout="LAMMPS mock run", stderr=""
        )

    monkeypatch.setattr(
        "ursa.agents.lammps_agent.subprocess.run",
        fake_run,
    )

    inputs = {
        "simulation_task": "Prepare a minimal LAMMPS script for copper atoms.",
        "elements": ["Cu"],
        "template": "",
        "chosen_potential": DummyPotential(),
    }

    result = await agent.ainvoke(inputs)

    assert result["run_returncode"] == 0
    assert isinstance(result["input_script"], str)
    assert result["input_script"].strip()
    assert subprocess_calls
    assert agent.lammps_cmd in subprocess_calls[0][0]
    assert subprocess_calls[0][1] == tmp_path
    assert (tmp_path / "in.lammps").exists()
