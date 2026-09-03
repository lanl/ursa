#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 {easy|examples|medium} {codex-docker|codex-apptainer|ursa-docker|ursa-apptainer}" >&2
  exit 2
fi

wave=$1
configuration=$2
config_dir=jobs/wave-validation-6f5d75c9149c42f0
config_file=$config_dir/$configuration.json

account=$(id -un)
account_gid=$(id -g "$account")
account_group=$(id -gn "$account")
current_gid=$(id -g)

printf -v launcher '%q ' "$0" "$@"
if [[ $current_gid != "$account_gid" ]]; then
  if [[ ${URSA_HARBOR_GROUP_BOOTSTRAPPED:-} == 1 ]]; then
    echo "could not restore primary group $account_group" >&2
    exit 1
  fi
  export URSA_HARBOR_GROUP_BOOTSTRAPPED=1
  exec sg "$account_group" -c "$launcher"
fi

if ! docker info >/dev/null 2>&1; then
  if [[ ${URSA_HARBOR_GROUP_BOOTSTRAPPED:-} == 1 ]]; then
    echo "Docker is unavailable; start the daemon and refresh Docker-group membership" >&2
    exit 1
  fi
  export URSA_HARBOR_GROUP_BOOTSTRAPPED=1
  printf -v inner_launcher 'exec sg %q -c %q' "$account_group" "$launcher"
  exec sg docker -c "$inner_launcher"
fi

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
    tasks=(crack-7z-hash kv-store-grpc largest-eigenval multi-source-data-merger)
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
