#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# Data-free standard OPD. Training prompts are deterministic sequences drawn
# from an actor-owned continuous nn.Parameter bank. The tokenizer and LM
# vocabulary stay untouched; validation still uses normal Math text prompts.
export EXTRAPOLATION_LENGTH=0
export LAMBDA_VAL=1.0
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-16384}"
export TOTAL_TRAINING_STEPS=50
export TOTAL_EPOCHS=50

export MEMORY_BANK_SIZE=32
export MEMORY_PROMPT_LENGTH=32
export MEMORY_NUM_PROMPTS=1024
export MEMORY_BANK_INIT_STD=0.02
export MEMORY_BANK_ADVERSARIAL="${MEMORY_BANK_ADVERSARIAL:-False}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"

export TRAIN_PATH="memory://continuous-prompts"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_standard_opd_continuous_memory_${MEMORY_BANK_SIZE}x${MEMORY_PROMPT_LENGTH}_steps_${TOTAL_TRAINING_STEPS}}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${VERL_ROOT}/G-OPD-checkpoints}"
export ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/rollout_data}"
export SAVE_FREQ=50
export TEST_FREQ=10
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

echo "Data-free continuous memory-prompt configuration"
echo "  memory_bank_size=${MEMORY_BANK_SIZE}"
echo "  memory_prompt_length=${MEMORY_PROMPT_LENGTH}"
echo "  memory_num_prompts=${MEMORY_NUM_PROMPTS}"
echo "  memory_prompt_order=fixed_0_to_$((MEMORY_PROMPT_LENGTH - 1))"
echo "  memory_bank_init_std=${MEMORY_BANK_INIT_STD}"
echo "  memory_bank_lr=same_as_student"
echo "  memory_bank_adversarial=${MEMORY_BANK_ADVERSARIAL}"
echo "  tokenizer_vocab_resize=disabled"
echo "  required_vllm_version=>=0.10.0"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  train_response_length=${MAX_RESPONSE_LENGTH}"
echo "  validation_response_length=${VAL_MAX_RESPONSE_LENGTH}"
echo "  rollout_data_dir=${ROLLOUT_DATA_DIR}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
    python3 - <<'PY'
from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version

try:
    installed = version("vllm")
except PackageNotFoundError as exc:
    raise SystemExit("vLLM is not installed; continuous memory prompts require vLLM>=0.10.0") from exc

if Version(installed) < Version("0.10.0"):
    raise SystemExit(
        f"Found vLLM {installed}; continuous memory prompts require vLLM>=0.10.0 "
        "(vLLM 0.10.0 + PyTorch 2.7.1 is the repository-tested stack)"
    )
print(f"  detected_vllm_version={installed}")
PY
fi

exec bash "${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.sh" 0 \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.dataloader_num_workers=0 \
    data.custom_cls.path="${SCRIPT_DIR}/memory_token_dataset.py" \
    data.custom_cls.name=ContinuousMemoryPromptDataset \
    data.custom_cls.apply_to_train_only=True \
    data.disable_ref_retokenization=True \
    +data.memory_prompt.num_prompts="${MEMORY_NUM_PROMPTS}" \
    +data.memory_prompt.prompt_length="${MEMORY_PROMPT_LENGTH}" \
    +data.memory_prompt.bank_size="${MEMORY_BANK_SIZE}" \
    +data.memory_prompt.seed=42 \
    +data.memory_prompt.fixed_order=True \
    actor_rollout_ref.model.memory_token_count=0 \
    actor_rollout_ref.model.continuous_memory_bank_size="${MEMORY_BANK_SIZE}" \
    actor_rollout_ref.model.continuous_memory_init_std="${MEMORY_BANK_INIT_STD}" \
    actor_rollout_ref.model.continuous_memory_seed=42 \
    actor_rollout_ref.model.continuous_memory_adversarial="${MEMORY_BANK_ADVERSARIAL}" \
    actor_rollout_ref.model.continuous_memory_gradient_scale=1.0 \
    custom_reward_function.path="${SCRIPT_DIR}/memory_token_dataset.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    "$@"
