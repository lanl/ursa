#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {easy|examples|medium} {codex-docker|codex-apptainer|ursa-docker|ursa-apptainer}" >&2
  exit 2
fi

wave=$1
configuration=$2
config_dir=jobs/wave-validation-793a452a34138ea8
config_file=$config_dir/$configuration.json

if [[ ! -f $config_file ]]; then
  echo "unknown configuration: $configuration" >&2
  exit 2
fi

case $wave in
  easy)
    wave_number=1
    task_source=(--dataset terminal-bench@2.0)
    tasks=(overfull-hbox cobol-modernization fix-git prove-plus-comm)
    ;;
  examples)
    wave_number=2
    task_source=(--path examples/harbor/benchmark/tasks)
    tasks=(install-gpaw install-parthenon install-pytorch)
    ;;
  medium)
    wave_number=3
    task_source=(--dataset terminal-bench@2.0)
    tasks=(log-summary-date-ranges nginx-request-logging db-wal-recovery git-leak-recovery)
    ;;
  *)
    echo "unknown wave: $wave" >&2
    exit 2
    ;;
esac

task_args=()
for task in "${tasks[@]}"; do
  task_args+=(-i "$task")
done

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

uv run --python 3.12 --extra harbor harbor run \
  --config "$config_file" \
  --job-name "wave${wave_number}-${wave}-${configuration}" \
  "${task_source[@]}" \
  "${task_args[@]}" \
  --yes
