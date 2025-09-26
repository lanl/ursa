#!/usr/bin/env bash
set -euo pipefail
source "${1:-.env}"

runai workspace submit "$WORKSPACE_NAME" \
  -p "$PROJECT" \
  -i "$IMAGE" \
  --gpu-devices-request "$GPU_DEVICES" \
  ${PREEMPTIBLE:+--preemptible} \
  --existing-pvc "claimname=${PVC_NAME},path=${PVC_MOUNT_PATH}" \
  --node-pools "$NODE_POOLS" \
  --command -- sleep infinity

