#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model

from ursa.agents import ExecutionAgent, PlanningAgent, RAGAgent
from ursa.workflows import CriticalMineralsWorkflow

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional import
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_CORPUS = (
    "/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus"
)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    return str(value)


def _load_scenario(path: Path, scenario_name: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if scenario_name not in raw:
        available = ", ".join(sorted(raw.keys()))
        raise KeyError(
            f"Scenario '{scenario_name}' not found. Available: {available}"
        )
    scenario = raw[scenario_name]
    if not isinstance(scenario, dict):
        raise ValueError("Scenario payload must be a mapping.")
    return scenario


def _build_model(model_name: str):
    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": 0,
    }
    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url
    if api_key := os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = api_key
    return init_chat_model(**kwargs)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run end-to-end CMM demo workflow and emit artifacts."
    )
    parser.add_argument("--scenario", default="ndfeb_la_y_5pct_baseline")
    parser.add_argument(
        "--scenarios-path",
        default="configs/cmm_demo_scenarios.json",
    )
    parser.add_argument(
        "--corpus-path",
        default=os.getenv("CMM_CORPUS_PATH", DEFAULT_CORPUS),
    )
    parser.add_argument(
        "--vectorstore-path",
        default=os.getenv("CMM_VECTORSTORE_PATH", "cmm_vectorstore"),
    )
    parser.add_argument(
        "--summaries-path",
        default=os.getenv("CMM_SUMMARIES_PATH", "cmm_summaries"),
    )
    parser.add_argument(
        "--workspace",
        default=os.getenv("CMM_WORKSPACE_PATH", "cmm_demo_workspace"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("CMM_OUTPUT_DIR", "cmm_demo_outputs"),
    )
    parser.add_argument(
        "--planner-model",
        default=os.getenv("CMM_PLANNER_MODEL", "openai:gpt-5"),
    )
    parser.add_argument(
        "--executor-model",
        default=os.getenv("CMM_EXECUTOR_MODEL", "openai:gpt-5"),
    )
    parser.add_argument(
        "--rag-model",
        default=os.getenv("CMM_RAG_MODEL", "openai:gpt-5-nano"),
    )
    parser.add_argument("--print-result-json", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    scenario_path = Path(args.scenarios_path).expanduser().resolve()
    scenario = _load_scenario(scenario_path, args.scenario)

    workspace = Path(args.workspace).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    corpus_path = Path(args.corpus_path).expanduser().resolve()
    vectorstore_path = Path(args.vectorstore_path).expanduser().resolve()
    summaries_path = Path(args.summaries_path).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)

    planner = PlanningAgent(llm=_build_model(args.planner_model), workspace=workspace)
    executor = ExecutionAgent(
        llm=_build_model(args.executor_model),
        workspace=workspace,
    )

    rag_agent = RAGAgent(
        llm=_build_model(args.rag_model),
        workspace=workspace,
        database_path=corpus_path,
        vectorstore_path=vectorstore_path,
        summaries_path=summaries_path,
        vectorstore_backend=os.getenv("CMM_VECTORSTORE_BACKEND", "chroma"),
        retrieval_k=int(os.getenv("CMM_RETRIEVAL_K", "20")),
        return_k=int(os.getenv("CMM_RETURN_K", "5")),
        use_reranker=_parse_bool_env("CMM_USE_RERANKER", default=False),
        reranker_provider=os.getenv("CMM_RERANKER_PROVIDER", "none"),
    )

    workflow = CriticalMineralsWorkflow(
        planner=planner,
        executor=executor,
        rag_agent=rag_agent,
        workspace=workspace,
    )

    payload = {
        "task": scenario["task"],
        "local_corpus_path": str(corpus_path),
        "rag_context": scenario.get("rag_context", scenario["task"]),
        "source_queries": scenario.get("source_queries", {}),
        "optimization_input": scenario.get("optimization_input"),
        "execution_instruction": scenario.get(
            "execution_instruction",
            "Produce a source-grounded synthesis with uncertainty notes.",
        ),
    }

    result = workflow.invoke(payload)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / args.scenario / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(run_dir / "input_payload.json", payload)
    _write_json(run_dir / "workflow_result.json", result)
    _write_json(run_dir / "optimization_output.json", result.get("optimization"))
    _write_json(run_dir / "rag_metadata.json", result.get("rag", {}))

    final_summary = str(result.get("final_summary", "")).strip()
    (run_dir / "final_summary.md").write_text(
        final_summary + "\n",
        encoding="utf-8",
    )

    latest_dir = output_root / args.scenario / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "final_summary.md").write_text(
        final_summary + "\n", encoding="utf-8"
    )
    _write_json(latest_dir / "workflow_result.json", result)

    print(f"Scenario: {args.scenario}")
    print(f"Run artifacts: {run_dir}")
    print("Final summary preview:")
    print(final_summary[:1200])

    if args.print_result_json:
        print("\nFull result JSON:")
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
