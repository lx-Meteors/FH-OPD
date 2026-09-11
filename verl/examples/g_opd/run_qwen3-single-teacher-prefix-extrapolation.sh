#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash examples/g_opd/run_qwen3-math-single-teacher-prefix-extrapolation.sh LENGTH [HYDRA_OVERRIDES...]

LENGTH controls how many leading response tokens use reward extrapolation:
  0       Standard OPD. The actor base model is not loaded or forwarded.
  2k      Extrapolate the first 2048 response tokens.
  4096    Extrapolate the first 4096 response tokens.
  full    Extrapolate the full response (standard full G-OPD/ExOPD).

Environment variables:
  MODEL_ROOT           Local model directory (default: /personal/models).
  STUDENT_MODEL_PATH   Student model and default actor base model.
  BASE_MODEL_PATH      Flexible reference/base model used by extrapolation.
  TEACHER_MODEL_PATH   Math teacher model.
  DATA_ROOT            G-OPD-Training-Data directory.
  CHECKPOINT_ROOT      Root directory for checkpoints.
  LAMBDA_VAL           Reward scaling factor (default: 1.25).
  TOTAL_TRAINING_STEPS Paper setting for same-size G-OPD (default: 50).
  TOTAL_EPOCHS         Dataloader-pass upper bound (Math default: 1).
  TRAINER_LOGGER       Hydra logger list (default: ["console","wandb"]).
  RUN_POST_TRAIN_EVAL  Run merge and Math/Code evaluation after training (default: 1).
  MERGE_USE_CPU_INITIALIZATION  Merge safely on CPU (default: 1).
  DRY_RUN              Set to 1 to validate and print settings without training.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

EXTRAPOLATION_SPEC="${1:-${EXTRAPOLATION_LENGTH:-full}}"
if [[ $# -gt 0 ]]; then
    shift
fi

EXTRAPOLATION_NORMALIZED="$(printf '%s' "${EXTRAPOLATION_SPEC}" | tr '[:upper:]' '[:lower:]')"

case "${EXTRAPOLATION_NORMALIZED}" in
    full)
        EXTRAPOLATION_MAX_TOKENS=-1
        EXTRAPOLATION_LABEL="full"
        ;;
    0)
        EXTRAPOLATION_MAX_TOKENS=0
        EXTRAPOLATION_LABEL="0"
        ;;
    *k)
        K_VALUE="${EXTRAPOLATION_SPEC%[kK]}"
        if [[ ! "${K_VALUE}" =~ ^[0-9]+$ ]] || [[ "${K_VALUE}" -eq 0 ]]; then
            echo "Invalid extrapolation length: ${EXTRAPOLATION_SPEC}" >&2
            usage >&2
            exit 2
        fi
        EXTRAPOLATION_MAX_TOKENS=$((K_VALUE * 1024))
        EXTRAPOLATION_LABEL="${K_VALUE}k"
        ;;
    *)
        if [[ ! "${EXTRAPOLATION_SPEC}" =~ ^[0-9]+$ ]]; then
            echo "Invalid extrapolation length: ${EXTRAPOLATION_SPEC}" >&2
            usage >&2
            exit 2
        fi
        EXTRAPOLATION_MAX_TOKENS="${EXTRAPOLATION_SPEC}"
        EXTRAPOLATION_LABEL="${EXTRAPOLATION_SPEC}"
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-${VERL_ROOT}/../../G-OPD-Training-Data}"
MODEL_ROOT="${MODEL_ROOT:-/personal/models}"
STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${STUDENT_MODEL_PATH}}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Non-Thinking-RL-Math-Step500}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${VERL_ROOT}/G-OPD-checkpoints}"
LAMBDA_VAL="${LAMBDA_VAL:-1.25}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"wandb\"]}"

export PYTHONUNBUFFERED=1
export WANDB_API_KEY="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"
export WANDB_MODE="${WANDB_MODE:-online}"
export USED_MODEL="${USED_MODEL:-no_api}"

AIME24_TEST_PATH="${DATA_ROOT}/AIME2024/test.parquet"
AIME25_TEST_PATH="${DATA_ROOT}/AIME2025/test.parquet"
TRAIN_PATH="${TRAIN_PATH:-${DATA_ROOT}/DeepMath-103K/train_filtered_level6.parquet}"
TEST_FILES="${TEST_FILES:-['${AIME24_TEST_PATH}', '${AIME25_TEST_PATH}']}"
EVAL_SUITE="${EVAL_SUITE:-math}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_single_teacher_prefix_${EXTRAPOLATION_LABEL}_lambda_${LAMBDA_VAL}_steps_${TOTAL_TRAINING_STEPS}}"
OUTPUT_DIR="${OUTPUT_DIR:-${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}}"

echo "Single-teacher OPD configuration"
echo "  extrapolation_spec=${EXTRAPOLATION_SPEC}"
echo "  extrapolation_max_tokens=${EXTRAPOLATION_MAX_TOKENS}"
echo "  lambda=${LAMBDA_VAL}"
echo "  total_training_steps=${TOTAL_TRAINING_STEPS}"
echo "  max_dataloader_epochs=${TOTAL_EPOCHS}"
echo "  student=${STUDENT_MODEL_PATH}"
echo "  base_model=${BASE_MODEL_PATH}"
echo "  teacher=${TEACHER_MODEL_PATH}"
echo "  train_data=${TRAIN_PATH}"
echo "  validation_data=${TEST_FILES}"
echo "  output=${OUTPUT_DIR}"
echo "  post_train_eval=${RUN_POST_TRAIN_EVAL:-1} (${EVAL_SUITE})"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

cd "${VERL_ROOT}"

# total_epochs is only a dataloader-loop upper bound. The explicit paper setting
# below stops same-size G-OPD training exactly at TOTAL_TRAINING_STEPS.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=5.0 \
    algorithm.rollout_correction.rollout_rs=null \
    algorithm.rollout_correction.bypass_mode=false \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    data.train_files="${TRAIN_PATH}" \
    data.val_files="${TEST_FILES}" \
    data.train_batch_size=1024 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    data.seed=42 \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}" \
    +actor_rollout_ref.model.base_model_path="${BASE_MODEL_PATH}" \
    +actor_rollout_ref.ref.model.path="${TEACHER_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True \
    actor_rollout_ref.actor.policy_loss.lambda_vals="${LAMBDA_VAL}" \
    actor_rollout_ref.actor.policy_loss.extrapolation_max_tokens="${EXTRAPOLATION_MAX_TOKENS}" \
    actor_rollout_ref.actor.policy_loss.multi_teacher_distill=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=1024 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.val_before_train=True \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.log_val_generations=10 \
    trainer.project_name=on-policy-distillation \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.test_freq=10 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"

if [[ "${RUN_POST_TRAIN_EVAL:-1}" == "1" ]]; then
    bash "${SCRIPT_DIR}/post_train_merge_and_eval.sh" "${OUTPUT_DIR}" "${EVAL_SUITE}" "${EXPERIMENT_NAME}"
else
    echo "Post-training evaluation disabled (RUN_POST_TRAIN_EVAL=${RUN_POST_TRAIN_EVAL})."
fi
