#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Standard OPD: lambda=1 and no reward extrapolation. The student generates
# exactly one response token per training rollout for 50,000 optimizer steps.
export EXTRAPOLATION_LENGTH=0
export LAMBDA_VAL=1.0
export MAX_RESPONSE_LENGTH=1
export VAL_MAX_RESPONSE_LENGTH=16384
export TOTAL_TRAINING_STEPS=50000

# total_epochs is only an upper bound. Setting it to the step count guarantees
# that the explicit 50,000-step limit is reached even for a small dataloader.
export TOTAL_EPOCHS=50000

# Validation uses its own normal response length. Validate and save every
# 5,000 steps to keep long-run evaluation cost and checkpoint storage bounded.
export VAL_BEFORE_TRAIN=True
export TEST_FREQ="${TEST_FREQ:-5000}"
export SAVE_FREQ="${SAVE_FREQ:-5000}"
export RUN_POST_TRAIN_EVAL="${RUN_POST_TRAIN_EVAL:-1}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_standard_opd_rollout_1token_steps_50000}"

exec bash "${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.sh" 0 "$@"
