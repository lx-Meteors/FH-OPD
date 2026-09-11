#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
MODEL_NAME="${MODEL_NAME:-$(basename -- "${MODEL_PATH}")}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/eval_outputs}"

echo "Math evaluation model: ${MODEL_PATH}"
echo "Math evaluation results: ${OUTPUT_ROOT}"

# Create output directories if they don't exist
mkdir -p "${OUTPUT_ROOT}/aime24"
mkdir -p "${OUTPUT_ROOT}/aime25"
mkdir -p "${OUTPUT_ROOT}/hmmt25_feb"
mkdir -p "${OUTPUT_ROOT}/hmmt25_nov"

# aime24
CUDA_VISIBLE_DEVICES="${MATH_EVAL_GPUS_AIME24:-0,1}" python3 "${SCRIPT_DIR}/eval_math.py" \
    --input_file "${REPO_ROOT}/data/aime24/test.jsonl" \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_ROOT}/aime24/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &
pids=("$!")


# aime25
CUDA_VISIBLE_DEVICES="${MATH_EVAL_GPUS_AIME25:-2,3}" python3 "${SCRIPT_DIR}/eval_math.py" \
    --input_file "${REPO_ROOT}/data/aime25/test.jsonl" \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_ROOT}/aime25/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &
pids+=("$!")



# hmmt25-Feb
CUDA_VISIBLE_DEVICES="${MATH_EVAL_GPUS_HMMT_FEB:-4,5}" python3 "${SCRIPT_DIR}/eval_math.py" \
    --input_file "${REPO_ROOT}/data/hmmt25_feb/test.jsonl" \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_ROOT}/hmmt25_feb/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &
pids+=("$!")



# hmmt25-Nov
CUDA_VISIBLE_DEVICES="${MATH_EVAL_GPUS_HMMT_NOV:-6,7}" python3 "${SCRIPT_DIR}/eval_math.py" \
    --input_file "${REPO_ROOT}/data/hmmt25_nov/test.jsonl" \
    --model_path "${MODEL_PATH}" \
    --output_file "${OUTPUT_ROOT}/hmmt25_nov/${MODEL_NAME}.jsonl" \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &
pids+=("$!")

eval_status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        eval_status=1
    fi
done
if [[ "${eval_status}" -ne 0 ]]; then
    echo "One or more Math evaluations failed." >&2
    exit "${eval_status}"
fi
echo "Model ${MODEL_NAME} done. Results: ${OUTPUT_ROOT}"
