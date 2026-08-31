#!/usr/bin/env bash
#SBATCH --job-name=ursa-harbor
#SBATCH --output=ursa-harbor-%A_%a.out
#SBATCH --error=ursa-harbor-%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?This script must run as a SLURM array job}"
: "${URSA_HARBOR_TASK_FILE:?Submit with examples/harbor/submit_slurm.sh}"
task=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$URSA_HARBOR_TASK_FILE")
test -n "$task"
repo_root=$(cd "$(dirname "$0")/../.." && pwd)
cache_dir=${URSA_HARBOR_SIF_CACHE:-"$repo_root/.harbor-sif-cache"}
jobs_dir=${URSA_HARBOR_JOBS_DIR:-"$repo_root/jobs"}
model=${URSA_HARBOR_MODEL:-openai/gpt-4.1-nano}
config_file=${URSA_HARBOR_CONFIG:-"$repo_root/examples/harbor/ursa.yaml"}
mkdir -p "$cache_dir" "$jobs_dir"
uv run --project "$repo_root/examples/harbor" --python 3.12 harbor run \
  --path "$task" --agent ursa.integrations.harbor:UrsaHarborAgent --model "$model" \
  --agent-kwarg "config_file=$config_file" \
  --agent-kwarg "ursa_source_dir=$repo_root" \
  --env ursa.integrations.harbor_singularity:DockerfileSingularityEnvironment \
  --environment-kwarg "singularity_image_cache_dir=$cache_dir" \
  --environment-kwarg singularity_no_mount=home,tmp \
  --jobs-dir "$jobs_dir" --n-concurrent 1 --yes
