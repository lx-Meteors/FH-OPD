#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# ==================== User-editable Math configuration ====================
# A positional argument still takes priority, for example: bash "$0" 2k
export EXTRAPOLATION_LENGTH="${EXTRAPOLATION_LENGTH:-full}"

export DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/../../G-OPD-Training-Data}"
export MODEL_ROOT="${MODEL_ROOT:-/personal/models}"
export STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
export BASE_MODEL_PATH="${BASE_MODEL_PATH:-${STUDENT_MODEL_PATH}}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500}"

export TRAIN_PATH="${TRAIN_PATH:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
export TEST_FILES="${TEST_FILES:-['${DATA_ROOT}/AIME2024/test.parquet', '${DATA_ROOT}/AIME2025/test.parquet']}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${VERL_ROOT}/G-OPD-checkpoints}"

export LAMBDA_VAL="${LAMBDA_VAL:-1.25}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"
export RUN_POST_TRAIN_EVAL="${RUN_POST_TRAIN_EVAL:-1}"
export EVAL_SUITE=math
# ===========================================================================

exec bash "${SCRIPT_DIR}/run_qwen3-single-teacher-prefix-extrapolation.sh" "$@"
