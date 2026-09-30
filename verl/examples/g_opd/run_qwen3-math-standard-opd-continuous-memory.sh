#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Canonical entry point for the continuous MemoryBank experiment. The old
# memory-tokens filename remains as a compatibility alias for existing jobs.
exec bash "${SCRIPT_DIR}/run_qwen3-math-standard-opd-memory-tokens.sh" "$@"
