#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from langchain.chat_models import init_chat_model

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional import
    load_dotenv = None  # type: ignore[assignment]


DEFAULT_CORPUS = (
    "/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus"
)


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _check_env(required: list[str]) -> list[CheckResult]:
    results = []
    for key in required:
        value = os.getenv(key, "").strip()
        if value:
            results.append(CheckResult(key, "PASS", "set"))
        else:
            results.append(CheckResult(key, "FAIL", "missing"))
    return results


def _count_manifest_ids(path: Path) -> int:
    if not path.exists():
        return 0
    return len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _corpus_scan(corpus_path: Path, sample_limit: int = 5000) -> tuple[int, int]:
    visited = 0
    matched = 0
    valid_ext = {".pdf", ".txt", ".md", ".csv", ".json", ".xml"}
    for path in corpus_path.rglob("*"):
        if not path.is_file():
            continue
        visited += 1
        if path.suffix.lower() in valid_ext:
            matched += 1
        if visited >= sample_limit:
            break
    return visited, matched


def _model_connectivity_check(model_name: str) -> CheckResult:
    kwargs = {
        "model": model_name,
        "temperature": 0,
    }
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key

    try:
        model = init_chat_model(**kwargs)
        response = model.invoke("Reply exactly with: connectivity_ok")
        text = getattr(response, "content", "")
        if isinstance(text, list):
            text = str(text)
        text = str(text)
        if "connectivity_ok" in text:
            return CheckResult("model_connectivity", "PASS", text)
        return CheckResult("model_connectivity", "WARN", text)
    except Exception as exc:
        return CheckResult("model_connectivity", "FAIL", str(exc))


def _print_results(results: list[CheckResult]) -> tuple[int, int, int]:
    passed = sum(1 for result in results if result.status == "PASS")
    warned = sum(1 for result in results if result.status == "WARN")
    failed = sum(1 for result in results if result.status == "FAIL")

    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")

    print(
        f"\nSummary: PASS={passed} WARN={warned} FAIL={failed}"
    )
    return passed, warned, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CMM demo readiness healthcheck."
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
        "--scenarios-path",
        default="configs/cmm_demo_scenarios.json",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("CMM_RAG_MODEL", "openai:gpt-5-nano"),
    )
    parser.add_argument("--skip-model-check", action="store_true")
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    corpus_path = Path(args.corpus_path).expanduser().resolve()
    vectorstore_path = Path(args.vectorstore_path).expanduser().resolve()
    scenarios_path = Path(args.scenarios_path).expanduser().resolve()

    results: list[CheckResult] = []
    results.extend(_check_env(["OPENAI_API_KEY", "OPENAI_BASE_URL"]))

    if corpus_path.exists() and corpus_path.is_dir():
        visited, matched = _corpus_scan(corpus_path)
        results.append(
            CheckResult(
                "corpus_path",
                "PASS" if matched > 0 else "WARN",
                f"exists, sampled_files={visited}, sampled_cmm_ext_matches={matched}",
            )
        )
    else:
        results.append(
            CheckResult("corpus_path", "FAIL", f"not found: {corpus_path}")
        )

    if scenarios_path.exists():
        try:
            payload = json.loads(scenarios_path.read_text(encoding="utf-8"))
            scenario_count = len(payload.keys()) if isinstance(payload, dict) else 0
            status = "PASS" if scenario_count > 0 else "WARN"
            results.append(
                CheckResult(
                    "scenarios_config",
                    status,
                    f"path={scenarios_path}, scenarios={scenario_count}",
                )
            )
        except Exception as exc:
            results.append(
                CheckResult("scenarios_config", "FAIL", f"invalid json: {exc}")
            )
    else:
        results.append(
            CheckResult(
                "scenarios_config",
                "FAIL",
                f"missing file: {scenarios_path}",
            )
        )

    if vectorstore_path.exists() and vectorstore_path.is_dir():
        manifest_count = _count_manifest_ids(vectorstore_path / "_ingested_ids.txt")
        has_index_files = any(vectorstore_path.iterdir())
        status = "PASS" if has_index_files else "WARN"
        results.append(
            CheckResult(
                "vectorstore_path",
                status,
                f"exists={has_index_files}, manifest_docs={manifest_count}, path={vectorstore_path}",
            )
        )
        if manifest_count == 0:
            results.append(
                CheckResult(
                    "vectorstore_manifest",
                    "WARN",
                    "manifest missing or empty; run scripts/reindex.py",
                )
            )
        else:
            results.append(
                CheckResult(
                    "vectorstore_manifest",
                    "PASS",
                    f"manifest doc ids: {manifest_count}",
                )
            )
    else:
        results.append(
            CheckResult(
                "vectorstore_path",
                "FAIL",
                f"not found: {vectorstore_path}",
            )
        )

    if args.skip_model_check:
        results.append(
            CheckResult("model_connectivity", "WARN", "skipped by flag")
        )
    else:
        results.append(_model_connectivity_check(args.model))

    _, _, failed = _print_results(results)
    return 1 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
