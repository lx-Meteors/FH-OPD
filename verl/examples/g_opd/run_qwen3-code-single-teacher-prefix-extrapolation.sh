#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash examples/g_opd/run_qwen3-code-single-teacher-prefix-extrapolation.sh LENGTH [HYDRA_OVERRIDES...]

LENGTH may be 0, a token count such as 2048, a k-suffix such as 2k, or full.

Code-specific environment variables:
  TEACHER_MODEL_PATH      Default: /personal/models/Qwen3-4B-Non-Thinking-RL-Code-Step300
  SANDBOX_FUSION_URL      Optional isolated code-execution endpoint
  SANDBOX_MAX_CONCURRENT  Default: 64
  REWARD_MANAGER          Default: prime
  CODE_EVAL_GPU           GPU used by automatic Code evaluation (default: 0)
  CODE_EVAL_N             Samples per Code benchmark problem (default: 4)

The common single-teacher variables MODEL_ROOT, STUDENT_MODEL_PATH, BASE_MODEL_PATH,
DATA_ROOT, CHECKPOINT_ROOT, LAMBDA_VAL, TRAINER_LOGGER, RUN_POST_TRAIN_EVAL,
and DRY_RUN are also supported.
EOF
    exit 0
fi

EXTRAPOLATION_SPEC="${1:-${EXTRAPOLATION_LENGTH:-full}}"
if [[ $# -gt 0 ]]; then
    shift
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMMON_SCRIPT="${SCRIPT_DIR}/run_qwen3-single-teacher-prefix-extrapolation.sh"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/../../G-OPD-Training-Data}"
export MODEL_ROOT="${MODEL_ROOT:-/personal/models}"
export TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Code-Step300}"
export TRAIN_PATH="${DATA_ROOT}/Eurus/code_train.parquet"
export TEST_FILES="['${DATA_ROOT}/Eurus/code_validation.parquet']"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export EXPERIMENT_NAME="qwen3_4b_code_single_teacher_prefix_${EXTRAPOLATION_SPEC}_lambda_${LAMBDA_VAL:-1.25}_steps_${TOTAL_TRAINING_STEPS}"
export EVAL_SUITE=code

SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-null}"
SANDBOX_MAX_CONCURRENT="${SANDBOX_MAX_CONCURRENT:-64}"
REWARD_MANAGER="${REWARD_MANAGER:-prime}"

if [[ "${SANDBOX_FUSION_URL}" == "null" || -z "${SANDBOX_FUSION_URL}" ]]; then
    echo "WARNING: SANDBOX_FUSION_URL is not set." >&2
    echo "Generated code will be checked locally by PRIME, which is not a security sandbox." >&2
fi

exec bash "${COMMON_SCRIPT}" "${EXTRAPOLATION_SPEC}" \
    data.val_batch_size=256 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager="${REWARD_MANAGER}" \
    reward_model.sandbox_fusion.url="${SANDBOX_FUSION_URL}" \
    reward_model.sandbox_fusion.max_concurrent="${SANDBOX_MAX_CONCURRENT}" \
    reward_model.sandbox_fusion.memory_limit_mb=1024 \
    "$@"
