#!/usr/bin/env bash
set -euo pipefail
benchmark_dir=${1:-"$(cd "$(dirname "$0")/benchmark/tasks" && pwd)"}
mapfile -t tasks < <(find "$benchmark_dir" -mindepth 1 -maxdepth 1 -type d | sort)
if ((${#tasks[@]} == 0)); then
  echo "No task directories found below $benchmark_dir" >&2
  exit 2
fi
# Keep the manifest on the benchmark's shared filesystem so compute nodes see it.
task_file=$(mktemp "$benchmark_dir/.ursa-harbor-tasks.XXXXXX")
printf '%s\n' "${tasks[@]}" > "$task_file"
sbatch --array="0-$((${#tasks[@]} - 1))" \
  --export="ALL,URSA_HARBOR_TASK_FILE=$task_file" \
  "$(dirname "$0")/slurm_array_job.sh"
echo "Task manifest: $task_file (remove it after the array completes)"
