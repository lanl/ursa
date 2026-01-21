import json

import pytest

from ursa.agents import MaterialsProjectAgent


class FakeDoc:
    def __init__(self, material_id: str, metadata: dict):
        self.material_id = material_id
        self._metadata = metadata

    def dict(self) -> dict:
        return self._metadata


@pytest.fixture
def stub_mprester(monkeypatch):
    search_calls: list[dict] = []
    docs_metadata = [
        ("mp-001", {"formula": "GaInO3", "band_gap": 1.8, "volume": 123.4}),
        ("mp-002", {"formula": "Ga2In", "band_gap": 2.3, "volume": 111.2}),
    ]

    class FakeSummary:
        def search(
            self,
            *,
            elements,
            band_gap,
            energy_above_hull,
            is_stable,
        ):
            search_calls.append({
                "elements": elements,
                "band_gap": band_gap,
                "energy_above_hull": energy_above_hull,
                "is_stable": is_stable,
            })
            return [FakeDoc(mid, meta) for mid, meta in docs_metadata]

    class FakeMaterials:
        def __init__(self):
            self.summary = FakeSummary()

    class FakeMPRester:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            self.materials = FakeMaterials()
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("ursa.agents.mp_agent.MPRester", FakeMPRester)
    return search_calls, docs_metadata


@pytest.mark.asyncio
async def test_materials_project_agent_ainvoke_produces_summary(
    chat_model, tmpdir, stub_mprester
):
    search_calls, docs_metadata = stub_mprester

    agent = MaterialsProjectAgent(
        llm=chat_model,
        max_results=1,
        summarize=True,
        workspace=tmpdir,
        enable_metrics=False,
    )
    database_path = agent.database_path
    summaries_path = agent.summaries_path

    query = {
        "elements": ["Ga", "In"],
        "band_gap_min": 1.5,
        "band_gap_max": 2.5,
    }
    context = (
        "Highlight the stability and band gaps for any promising compound."
    )

    result = await agent.ainvoke({"query": query, "context": context})

    assert len(search_calls) == 1
    search_kwargs = search_calls[0]
    assert search_kwargs["elements"] == query["elements"]
    assert search_kwargs["band_gap"] == (
        query["band_gap_min"],
        query["band_gap_max"],
    )
    assert search_kwargs["energy_above_hull"] == (0, 0)
    assert search_kwargs["is_stable"] is True

    assert "materials" in result
    assert len(result["materials"]) == 1
    first_material = result["materials"][0]
    expected_id = docs_metadata[0][0]
    assert first_material["material_id"] == expected_id
    assert (
        first_material["metadata"]["formula"] == docs_metadata[0][1]["formula"]
    )

    assert "summaries" in result
    assert len(result["summaries"]) == 1
    assert isinstance(result["summaries"][0], str)

    assert "final_summary" in result
    assert isinstance(result["final_summary"], str)

    kept_json = database_path / f"{expected_id}.json"
    kept_summary = summaries_path / f"{expected_id}_summary.txt"
    assert kept_json.exists()
    assert kept_summary.exists()
    persisted_metadata = json.loads(kept_json.read_text())
    assert persisted_metadata["formula"] == docs_metadata[0][1]["formula"]

    skipped_id = docs_metadata[1][0]
    assert not (database_path / f"{skipped_id}.json").exists()
    assert not (summaries_path / f"{skipped_id}_summary.txt").exists()
