#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EVAL_SUITE=math

exec bash "${SCRIPT_DIR}/run_qwen3-single-teacher-prefix-extrapolation.sh" "$@"
