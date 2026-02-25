# URSA CMM Guide for Domain Scientists

This guide is a practical, end-to-end manual for domain scientists who want to
use the URSA Critical Minerals and Materials (CMM) workflow without digging
through the full codebase first.

It covers:

1. Environment setup.
2. API/model configuration.
3. Corpus preprocessing and indexing.
4. Running CMM scenarios and custom analyses.
5. Reading outputs (RAG + optimization).
6. Troubleshooting common failures.
7. Demo preparation workflow.

## 1. What This Workflow Does

At a high level, the CMM workflow combines:

- Planning (`PlanningAgent`): decomposes the task and response strategy.
- Retrieval (`RAGAgent`): pulls relevant evidence from your local corpus.
- Deterministic optimization (`run_cmm_supply_chain_optimization`): computes
  supply allocations, unmet demand, and composition-constraint feasibility.
- Synthesis (`ExecutionAgent`): writes the final decision narrative.

Primary workflow path:

- `src/ursa/workflows/critical_minerals_workflow.py`
- `scripts/run_cmm_demo.py`

## 2. Prerequisites

- macOS/Linux terminal access.
- Python 3.10+.
- [`uv`](https://docs.astral.sh/uv/) installed.
- A valid `OPENAI_API_KEY`.
- An OpenAI-compatible endpoint (`OPENAI_BASE_URL`) if you are not using
  `https://api.openai.com/v1`.

Repository path used below:

- `/Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa`

Corpus path used below:

- `/Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus`

## 3. Initial Setup

### 3.1 Enter the repo

```bash
cd /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa
```

### 3.2 Install dependencies

```bash
uv sync --group dev
uv sync --extra cmm
```

### 3.3 Create `.env`

```bash
cp .env.example .env
```

Set at least:

```bash
OPENAI_API_KEY=<your_key>
OPENAI_BASE_URL=<your_openai_compatible_base_url>
```

## 4. Recommended Runtime Profiles

The most important rule is: **embedding settings at runtime must match how the
vectorstore was indexed**.

### Profile A: Use existing local index (common in this workspace)

Use if your index was built with `text-embedding-3-small-project`:

```bash
CMM_EMBEDDING_MODEL=openai:text-embedding-3-small-project
CMM_EMBEDDING_DIMENSIONS=1536
```

### Profile B: Build a fresh index with large embeddings

If you reindex with `text-embedding-3-large`, use:

```bash
CMM_EMBEDDING_MODEL=openai:text-embedding-3-large
CMM_EMBEDDING_DIMENSIONS=3072
```

### Common CMM defaults

```bash
CMM_VECTORSTORE_BACKEND=chroma
CMM_USE_RERANKER=false
CMM_RERANKER_PROVIDER=none
CMM_HYBRID_ALPHA=0.7
CMM_VECTORSTORE_COLLECTION=cmm_chunks
URSA_RAG_LEGACY_MODE=false
```

### Model access note

If your endpoint does not allow `openai:gpt-5`, set explicit project models:

```bash
CMM_PLANNER_MODEL=openai:gpt-5.2-project
CMM_EXECUTOR_MODEL=openai:gpt-5.2-project
CMM_RAG_MODEL=openai:gpt-5.2-project
```

## 5. Healthcheck Before Indexing/Running

```bash
uv run python scripts/demo_healthcheck.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --model openai:gpt-5.2-project
```

A good run has no `FAIL` lines.

## 6. Corpus Indexing (Preprocessing + Ingestion)

### 6.1 Smoke test index (small batch)

```bash
uv run python scripts/reindex.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --backend chroma \
  --embedding-model openai:text-embedding-3-small-project \
  --embedding-dimensions 1536 \
  --include-extension pdf \
  --include-extension txt \
  --include-extension md \
  --exclude-extension py \
  --max-docs 250 \
  --reset
```

### 6.2 Full indexing

```bash
uv run python scripts/reindex.py \
  --corpus-path /Users/wash198/Documents/Projects/Science_Projects/MPII_CMM/Corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --backend chroma \
  --embedding-model openai:text-embedding-3-small-project \
  --embedding-dimensions 1536 \
  --include-extension pdf \
  --include-extension txt \
  --include-extension md
```

Expected summary includes:

- `Docs indexed`
- `Chunks indexed`
- `Vectorstore count`
- Tag counts by commodity/subdomain

### 6.3 Manifest behavior

The script writes:

- `cmm_vectorstore/_ingested_ids.txt`

This prevents re-ingesting documents already indexed.

## 7. Running the CMM Workflow

### 7.1 Available built-in scenarios

Defined in:

- `configs/cmm_demo_scenarios.json`

Current scenario IDs:

- `ndfeb_la_y_5pct_baseline`
- `ndfeb_la_y_5pct_quality_tightening`
- `ndfeb_la_y_5pct_supply_shock`

### 7.2 Recommended run command

Use an already-indexed vectorstore and an empty corpus path to avoid
reparsing all raw files during each demo run:

```bash
CMM_EMBEDDING_MODEL=openai:text-embedding-3-small-project \
CMM_EMBEDDING_DIMENSIONS=1536 \
CMM_USE_RERANKER=false \
CMM_RERANKER_PROVIDER=none \
CMM_VECTORSTORE_BACKEND=chroma \
uv run python scripts/run_cmm_demo.py \
  --scenario ndfeb_la_y_5pct_baseline \
  --planner-model openai:gpt-5.2-project \
  --executor-model openai:gpt-5.2-project \
  --rag-model openai:gpt-5.2-project \
  --corpus-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/empty_corpus \
  --vectorstore-path /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_vectorstore \
  --output-dir /Users/wash198/Library/CloudStorage/OneDrive-PNNL/Documents/Projects/Science_Projects/MPII_CMM/ursa/cmm_demo_outputs
```

### 7.3 Run all three scenario variants

Repeat the same command with:

- `--scenario ndfeb_la_y_5pct_baseline`
- `--scenario ndfeb_la_y_5pct_quality_tightening`
- `--scenario ndfeb_la_y_5pct_supply_shock`

## 8. Understanding Output Artifacts

Each run writes:

- `cmm_demo_outputs/<scenario>/<timestamp>/input_payload.json`
- `cmm_demo_outputs/<scenario>/<timestamp>/workflow_result.json`
- `cmm_demo_outputs/<scenario>/<timestamp>/optimization_output.json`
- `cmm_demo_outputs/<scenario>/<timestamp>/rag_metadata.json`
- `cmm_demo_outputs/<scenario>/<timestamp>/final_summary.md`

And copies current output to:

- `cmm_demo_outputs/<scenario>/latest/`

### 8.1 Key optimization fields

In `optimization_output.json` (or `workflow_result.json -> optimization`):

- `status`
- `feasible`
- `objective_value`
- `allocations`
- `unmet_demand`
- `constraint_residuals`
- `composition` (targets/actual/residuals/tolerance)
- `sensitivity_summary`

### 8.2 Status interpretation

- `optimal_greedy`: demand and active constraints satisfied.
- `infeasible_unmet_demand`: demand could not be fully satisfied.
- `infeasible_composition_constraints`: demand may be met, but composition
  targets violate tolerance.
- `infeasible_unmet_and_composition`: both demand and composition fail.

### 8.3 RAG metadata interpretation

In `rag_metadata`:

- `num_results`: retrieved chunk count.
- `relevance_scores`: retrieval scores.
- `query_type`: classifier type (`general`, `multi_hop`, etc.).
- `filter_fallback_used`: retrieval had to retry without metadata filters.

## 9. Creating a New Domain Scenario

1. Open `configs/cmm_demo_scenarios.json`.
2. Add a new top-level scenario object with fields:
   - `task`
   - `rag_context`
   - `execution_instruction`
   - `source_queries` (optional)
   - `optimization_input`
3. In `optimization_input`, include:
   - demand/suppliers/shipping/risk parameters
   - `composition_targets` (e.g., `LA`, `Y`)
   - `composition_tolerance`
   - per-supplier `composition_profile`
4. Run with `scripts/run_cmm_demo.py --scenario <new_id>`.

## 10. Testing and Regression Checks

Run targeted tests after edits:

```bash
uv run pytest -q \
  tests/agents/test_rag_agent/test_rag_agent.py \
  tests/agents/test_rag_agent/test_cmm_components.py \
  tests/tools/test_cmm_supply_chain_optimization_tool.py \
  tests/workflows/test_critical_minerals_workflow.py
```

## 11. Common Failure Modes and Fixes

### 11.1 `team_model_access_denied` / 401 model errors

Cause: model name is not allowed on your endpoint.

Fix: set model env/flags to endpoint-allowed models, for example:

```bash
CMM_PLANNER_MODEL=openai:gpt-5.2-project
CMM_EXECUTOR_MODEL=openai:gpt-5.2-project
CMM_RAG_MODEL=openai:gpt-5.2-project
```

### 11.2 RAG returns zero results

Check:

- `vectorstore-path` is correct.
- embedding model/dimensions match index build settings.
- corpus was actually indexed (`_ingested_ids.txt` non-empty).
- `rag_context` contains domain-specific terminology.

### 11.3 Runs are very slow

Cause: workflow reparses entire corpus each run.

Fix: use an indexed vectorstore and point `--corpus-path` to `empty_corpus` for
repeat demo runs.

### 11.4 Indexing appears stalled

Use smaller batches and frequent flushes:

```bash
--max-docs 500 --flush-docs 20
```

Then iterate by tranche.

## 12. Demo-Day Checklist

1. `.env` loaded and correct.
2. Model connectivity healthcheck passes.
3. Vectorstore is already built.
4. Scenario commands tested once before recording.
5. Artifact folders verified under `cmm_demo_outputs/.../latest/`.
6. Keep one feasible scenario and one infeasible scenario ready to show
   contrast.

## 13. Fast Command Reference

### Reindex

```bash
uv run python scripts/reindex.py --corpus-path <CORPUS> --vectorstore-path <VECTORSTORE> --backend chroma --embedding-model openai:text-embedding-3-small-project --embedding-dimensions 1536
```

### Healthcheck

```bash
uv run python scripts/demo_healthcheck.py --corpus-path <CORPUS> --vectorstore-path <VECTORSTORE> --model openai:gpt-5.2-project
```

### Run scenario

```bash
uv run python scripts/run_cmm_demo.py --scenario <SCENARIO_ID> --planner-model openai:gpt-5.2-project --executor-model openai:gpt-5.2-project --rag-model openai:gpt-5.2-project --corpus-path <CORPUS_OR_EMPTY> --vectorstore-path <VECTORSTORE> --output-dir <OUTPUT_DIR>
```

