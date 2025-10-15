#!/usr/bin/env bash
set -euo pipefail
source "${1:-.env}"

runai workspace exec "$WORKSPACE_NAME" --tty --stdin -- bash

