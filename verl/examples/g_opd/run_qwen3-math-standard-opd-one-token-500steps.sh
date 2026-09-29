#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Standard OPD: lambda=1 and no reward extrapolation. The student generates
# exactly one response token per training rollout for 500 optimizer steps.
export EXTRAPOLATION_LENGTH=0
export LAMBDA_VAL=1.0
export MAX_RESPONSE_LENGTH=1
export VAL_MAX_RESPONSE_LENGTH=16384
export TOTAL_TRAINING_STEPS=500

# total_epochs is only an upper bound. Setting it to the step count guarantees
# that the explicit 500-step limit is reached even for a small dataloader.
export TOTAL_EPOCHS=500

# Validation uses its own normal response length. Validate and save at the end
# of the 500-step run; validation also runs once before training.
export VAL_BEFORE_TRAIN=True
export TEST_FREQ="${TEST_FREQ:-500}"
export SAVE_FREQ="${SAVE_FREQ:-500}"
export RUN_POST_TRAIN_EVAL="${RUN_POST_TRAIN_EVAL:-1}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_standard_opd_rollout_1token_steps_500}"

exec bash "${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.sh" 0 "$@"
