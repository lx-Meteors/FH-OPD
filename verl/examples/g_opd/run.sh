#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.log"

nohup bash "${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.sh" "$@" \
    > "${LOG_FILE}" 2>&1 < /dev/null &

echo "Math training started: pid=$!, log=${LOG_FILE}"
