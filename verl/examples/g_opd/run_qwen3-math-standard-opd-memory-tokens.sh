#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Data-free standard OPD. Training prompts are deterministic sequences drawn
# from a bank of prompt-only virtual tokens whose embedding rows are optimized
# together with the student. Validation still uses the normal Math datasets.
export EXTRAPOLATION_LENGTH=0
export LAMBDA_VAL=1.0
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
export VAL_MAX_RESPONSE_LENGTH="${VAL_MAX_RESPONSE_LENGTH:-16384}"
export TOTAL_TRAINING_STEPS=50
export TOTAL_EPOCHS=50

export MEMORY_TOKEN_COUNT=512
export MEMORY_PROMPT_LENGTH=64
export MEMORY_NUM_PROMPTS=1024
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"

export TRAIN_PATH="memory://virtual-token-prompts"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_standard_opd_memory_${MEMORY_TOKEN_COUNT}x${MEMORY_PROMPT_LENGTH}_steps_${TOTAL_TRAINING_STEPS}}"
export SAVE_FREQ=50
export TEST_FREQ=10
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

echo "Data-free memory-token prompt configuration"
echo "  memory_token_count=${MEMORY_TOKEN_COUNT}"
echo "  memory_prompt_length=${MEMORY_PROMPT_LENGTH}"
echo "  memory_num_prompts=${MEMORY_NUM_PROMPTS}"
echo "  train_batch_size=${TRAIN_BATCH_SIZE}"
echo "  train_response_length=${MAX_RESPONSE_LENGTH}"
echo "  validation_response_length=${VAL_MAX_RESPONSE_LENGTH}"

exec bash "${SCRIPT_DIR}/run_qwen3-math-single-teacher-prefix-extrapolation.sh" 0 \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MEMORY_PROMPT_LENGTH}" \
    data.dataloader_num_workers=0 \
    data.custom_cls.path="${SCRIPT_DIR}/memory_token_dataset.py" \
    data.custom_cls.name=MemoryTokenDataset \
    data.custom_cls.apply_to_train_only=True \
    data.disable_ref_retokenization=True \
    +data.memory_prompt.num_prompts="${MEMORY_NUM_PROMPTS}" \
    +data.memory_prompt.prompt_length="${MEMORY_PROMPT_LENGTH}" \
    +data.memory_prompt.token_count="${MEMORY_TOKEN_COUNT}" \
    +data.memory_prompt.seed=42 \
    actor_rollout_ref.model.memory_token_count="${MEMORY_TOKEN_COUNT}" \
    custom_reward_function.path="${SCRIPT_DIR}/memory_token_dataset.py" \
    custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    "$@"
