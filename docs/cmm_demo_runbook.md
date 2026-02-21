# CMM Demo Runbook: Full Workflow Build and Demo Readiness

This runbook provides a full, reproducible sequence to prepare and run the
Critical Minerals and Materials (CMM) demo on your local corpus.

It assumes this repository is at:

- `/Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa`

And your corpus is at:

- `/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus`

## 1. Prerequisites

- Python 3.10+ and `uv` installed.
- Valid `OPENAI_API_KEY`.
- Your OpenAI-compatible endpoint URL (custom `OPENAI_BASE_URL`).
- Network access from this machine to your model endpoint.

## 2. Open Repo and Install Dependencies

```bash
cd /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa

# Base + dev dependencies
uv sync --group dev

# Optional CMM extras (needed only for local/cohere/weaviate optional paths)
uv sync --extra cmm
```

## 3. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` and set at least:

```bash
OPENAI_API_KEY=<your_key>
OPENAI_BASE_URL=<your_openai_compatible_base_url>

# Demo defaults (OpenAI-only)
CMM_VECTORSTORE_BACKEND=chroma
CMM_EMBEDDING_MODEL=openai:text-embedding-3-large
CMM_EMBEDDING_DIMENSIONS=3072
CMM_USE_RERANKER=false
CMM_RERANKER_PROVIDER=none
CMM_HYBRID_ALPHA=0.7
CMM_VECTORSTORE_COLLECTION=cmm_chunks

URSA_RAG_LEGACY_MODE=false
```

## 4. Sanity Check Model Connectivity (Recommended)

```bash
uv run python - <<'PY'
import os
from langchain.chat_models import init_chat_model

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY")
model = init_chat_model(
    model="openai:gpt-5-nano",
    base_url=base_url,
    api_key=api_key,
    temperature=0,
)
print(model.invoke("Reply with exactly: connectivity_ok").content)
PY
```

Expected: `connectivity_ok`.

## 5. Preprocess + Ingest Corpus (Smoke Pass)

Start small to validate parsing and indexing behavior before full ingest.

```bash
uv run python scripts/reindex.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --backend chroma \
  --embedding-model openai:text-embedding-3-large \
  --embedding-dimensions 3072 \
  --include-extension pdf \
  --include-extension txt \
  --include-extension md \
  --exclude-extension py \
  --max-docs 250 \
  --reset
```

Review outputs for:

- `Docs indexed`
- `Chunks indexed`
- `Vectorstore count`
- `Commodity tag counts`
- `Subdomain tag counts`

## 6. Full Corpus Ingest

After smoke pass succeeds, run full ingest.

```bash
uv run python scripts/reindex.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --backend chroma \
  --embedding-model openai:text-embedding-3-large \
  --embedding-dimensions 3072 \
  --include-extension pdf \
  --include-extension txt \
  --include-extension md
```

Notes:

- Reindex writes a manifest at `cmm_vectorstore/_ingested_ids.txt`.
- This manifest prevents `RAGAgent` from re-ingesting already indexed docs.
- If you need a full rebuild, rerun with `--reset`.

## 7. Run Healthcheck and Demo Runner

You now have turnkey scripts:

- `scripts/demo_healthcheck.py`
- `scripts/run_cmm_demo.py`
- `configs/cmm_demo_scenarios.json`

### Healthcheck

```bash
uv run python scripts/demo_healthcheck.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore
```

### Run a scenario

```bash
uv run python scripts/run_cmm_demo.py \
  --scenario quick_gallium \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --output-dir /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_demo_outputs
```

Available scenarios:

- `quick_gallium`
- `graphite_battery`
- `multi_mineral_stress`

Artifacts are written under:

- `cmm_demo_outputs/<scenario>/<timestamp>/`
- `cmm_demo_outputs/<scenario>/latest/`

## 8. One-Command Demo Prep

Use the `just` target to run healthcheck + smoke demo in one command:

```bash
just demo-prep
```

If `just` is not installed:

```bash
uv tool install rust-just
```

Optional environment overrides:

```bash
export CMM_CORPUS_PATH=/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus
export CMM_VECTORSTORE_PATH=/Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore
export CMM_DEMO_SCENARIO=quick_gallium
export CMM_OUTPUT_DIR=/Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_demo_outputs
just demo-prep
```

## 9. Validate Demo Acceptance Criteria

Confirm the run output contains all of the following:

- Source-grounded RAG narrative in final summary.
- Retrieved context behavior that varies by query type.
- Optimization output with deterministic fields:
  - `objective_value`
  - `allocations`
  - `constraint_residuals`
  - `feasible` / `status`
  - `sensitivity_summary`

## 10. Regression / Confidence Tests

Run targeted tests used for this demo path:

```bash
uv run pytest -q \
  tests/agents/test_rag_agent/test_rag_agent.py \
  tests/agents/test_rag_agent/test_cmm_components.py \
  tests/tools/test_cmm_supply_chain_optimization_tool.py \
  tests/workflows/test_critical_minerals_workflow.py \
  tests/agents/test_execution_agent/test_execution_agent.py
```

## 11. Optional Architecture Toggles (Not Required for Demo)

### Weaviate backend

Set:

```bash
CMM_VECTORSTORE_BACKEND=weaviate
CMM_WEAVIATE_URL=<cluster_url>
CMM_WEAVIATE_API_KEY=<api_key>
```

Re-run `scripts/reindex.py` with `--backend weaviate`.

### Cohere reranker

Set:

```bash
CMM_USE_RERANKER=true
CMM_RERANKER_PROVIDER=cohere
COHERE_API_KEY=<api_key>
```

## 12. Demo-Day Checklist

- `.env` points to correct endpoint and key.
- `cmm_vectorstore` already built (avoid live full ingest on stage).
- `scripts/demo_healthcheck.py` reports no FAIL items.
- `scripts/run_cmm_demo.py` runs cleanly once pre-demo.
- Have 2-3 prepared prompts of increasing complexity.
- Keep one deterministic optimization scenario ready to replay.

## 13. Troubleshooting

- Empty RAG output:
  - Verify `cmm_vectorstore` path and `_ingested_ids.txt`.
  - Ensure corpus parseable files exist for selected extensions.
- Slow ingest:
  - Run a staged ingest with `--max-docs` batches.
  - Limit extensions to high-value types first (`pdf`, `txt`, `md`).
- API errors:
  - Re-check `OPENAI_BASE_URL` and endpoint auth requirements.
- Rebuild clean index:
  - Re-run `scripts/reindex.py ... --reset`.

## 14. Suggested Artifact Outputs for Demo Package

- Final `result` JSON from workflow run.
- `RAG_summary.txt` from summaries directory.
- Screenshot/log snippet showing indexed counts and tag distributions.
- A short one-page summary of insights and recommended actions.
