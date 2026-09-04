from uuid import uuid4

from ursa.observability.timing import PerLLMTimer


def test_on_llm_error_records_sample_without_matching_start():
    timer = PerLLMTimer()

    timer.on_llm_error(RuntimeError("boom"), run_id=uuid4())

    assert len(timer.samples) == 1
    sample = timer.samples[-1]
    assert sample["ok"] is False
    assert sample["name"] == "llm:unknown"
    assert "boom" in sample["metrics"]["error"]
    assert isinstance(sample["t_start"], float)
    assert isinstance(sample["t_end"], float)
    assert sample["t_end"] >= sample["t_start"]


def test_on_llm_error_records_sample_with_matching_start():
    timer = PerLLMTimer()
    run_id = uuid4()
    timer.on_llm_start({"name": "fake-model"}, ["prompt"], run_id=run_id)

    timer.on_llm_error(RuntimeError("boom"), run_id=run_id)

    assert len(timer.samples) == 1
    sample = timer.samples[-1]
    assert sample["ok"] is False
    assert sample["name"].startswith("llm:")
    assert "boom" in sample["metrics"]["error"]
    assert sample["t_end"] >= sample["t_start"]
