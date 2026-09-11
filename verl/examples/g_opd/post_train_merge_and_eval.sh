#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 OUTPUT_DIR {math|code} [EXPERIMENT_NAME]" >&2
    exit 2
fi

OUTPUT_DIR="$1"
EVAL_SUITE="$2"
EXPERIMENT_NAME="${3:-$(basename -- "${OUTPUT_DIR}")}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd -- "${VERL_ROOT}/.." && pwd)"

latest_step=-1
latest_checkpoint=""
for checkpoint_dir in "${OUTPUT_DIR}"/global_step_*; do
    [[ -d "${checkpoint_dir}" ]] || continue
    step_name="$(basename -- "${checkpoint_dir}")"
    step_number="${step_name#global_step_}"
    [[ "${step_number}" =~ ^[0-9]+$ ]] || continue
    if (( step_number > latest_step )); then
        latest_step="${step_number}"
        latest_checkpoint="${checkpoint_dir}"
    fi
done

if [[ -z "${latest_checkpoint}" ]]; then
    echo "Error: no global_step_* checkpoint found under ${OUTPUT_DIR}" >&2
    exit 1
fi

actor_checkpoint="${latest_checkpoint}/actor"
if [[ ! -d "${actor_checkpoint}" ]]; then
    echo "Error: actor checkpoint not found: ${actor_checkpoint}" >&2
    exit 1
fi

run_label="${EXPERIMENT_NAME}_step_${latest_step}"
merged_model_dir="${OUTPUT_DIR}/merged_hf/${run_label}"
result_root="${OUTPUT_DIR}/evaluation/${run_label}"
mkdir -p "${result_root}"

echo "Post-training evaluation"
echo "  suite=${EVAL_SUITE}"
echo "  checkpoint=${latest_checkpoint}"
echo "  merged_model=${merged_model_dir}"
echo "  results=${result_root}"

if [[ ! -f "${merged_model_dir}/config.json" ]]; then
    mkdir -p "${merged_model_dir}"
    cd "${VERL_ROOT}"
    merge_args=(
        -m verl.model_merger merge
        --backend fsdp
        --local_dir "${actor_checkpoint}"
        --target_dir "${merged_model_dir}"
    )
    if [[ "${MERGE_USE_CPU_INITIALIZATION:-1}" == "1" ]]; then
        merge_args+=(--use_cpu_initialization)
    fi
    python3 "${merge_args[@]}" 2>&1 | tee "${result_root}/merge.log"
else
    echo "Merged model already exists; reusing ${merged_model_dir}"
fi

case "${EVAL_SUITE}" in
    math)
        math_result_dir="${result_root}/math"
        mkdir -p "${math_result_dir}"
        MODEL_PATH="${merged_model_dir}" \
        MODEL_NAME="${run_label}" \
        OUTPUT_ROOT="${math_result_dir}" \
            bash "${REPO_ROOT}/math_eval/run_eval_math.sh" \
            2>&1 | tee "${result_root}/math_eval.log"
        ;;
    code)
        evalplus_root="${result_root}/evalplus"
        lcb_root="${result_root}/livecodebench"
        mkdir -p "${evalplus_root}" "${lcb_root}"

        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${CODE_EVAL_GPU:-0}" \
        EVALPLUS_ROOT="${evalplus_root}" \
            bash "${REPO_ROOT}/code_eval/scripts/run_evalplus.sh" \
            humaneval "${merged_model_dir}" 0 1.0 1.0 "${CODE_EVAL_N:-4}" \
            2>&1 | tee "${result_root}/humaneval_plus.log"

        CUDA_VISIBLE_DEVICES="${CODE_EVAL_GPU:-0}" \
        EVALPLUS_ROOT="${evalplus_root}" \
            bash "${REPO_ROOT}/code_eval/scripts/run_evalplus.sh" \
            mbpp "${merged_model_dir}" 0 1.0 1.0 "${CODE_EVAL_N:-4}" \
            2>&1 | tee "${result_root}/mbpp_plus.log"

        lcb_data_dir="${REPO_ROOT}/code_eval/coding/LiveCodeBench/code_generation_lite"
        if [[ ! -f "${lcb_data_dir}/test6.jsonl" ]]; then
            if ! command -v hf >/dev/null 2>&1; then
                echo "Error: LiveCodeBench data is missing and the hf command is unavailable." >&2
                exit 1
            fi
            echo "Downloading LiveCodeBench data to ${lcb_data_dir}"
            hf download livecodebench/code_generation_lite \
                --repo-type dataset \
                --local-dir "${lcb_data_dir}"
        fi

        LCB_OUTPUT_ROOT="${lcb_root}" \
            bash "${REPO_ROOT}/code_eval/scripts/run_lcb_gen.sh" \
            --model Qwen3-4B-NonThinking \
            --local_model_path "${merged_model_dir}" \
            --gpu "${CODE_EVAL_GPU:-0}" \
            --n "${CODE_EVAL_N:-4}" \
            2>&1 | tee "${result_root}/livecodebench.log"
        ;;
    *)
        echo "Error: unsupported evaluation suite: ${EVAL_SUITE}" >&2
        exit 2
        ;;
esac

{
    echo "Evaluation completed successfully."
    echo "Checkpoint: ${latest_checkpoint}"
    echo "Merged model: ${merged_model_dir}"
    echo "Results: ${result_root}"
} | tee "${result_root}/RESULTS_LOCATION.txt"
