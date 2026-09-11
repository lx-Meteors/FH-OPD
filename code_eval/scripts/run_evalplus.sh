#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EVALPLUS_ROOT="${EVALPLUS_ROOT:-${REPO_ROOT}/evalplus_results}"

export HUMANEVAL_OVERRIDE_PATH="${REPO_ROOT}/code_eval/data/HumanEvalPlus.jsonl"
export MBPP_OVERRIDE_PATH="${REPO_ROOT}/code_eval/data/MbppPlus.jsonl"
export PYTHONPATH="${REPO_ROOT}/code_eval/coding/evalplus${PYTHONPATH:+:${PYTHONPATH}}"

# Set defaults if not specified - fix argument assignments
DATASET=${1:-humaneval}
MODEL=${2:-"Qwen/Qwen3-4B"}
GREEDY=${3:-1}
TEMP=${4:-0.8}
TOP_P=${5:-0.9}
N_SAMPLES=${6:-1}

# If greedy mode, force n_samples to 1
if [ "$GREEDY" -eq 1 ]; then
    N_SAMPLES=1
fi

echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Greedy: $GREEDY (1=yes, 0=no)"
echo "Temperature: $TEMP"
echo "Top-P: $TOP_P"
echo "Number of samples: $N_SAMPLES"

# Extract model identifier for output file
MODEL_BASE=$(basename "$MODEL")
echo "Model base: $MODEL_BASE"

# Execute command directly without quoting the arguments
if [ "$GREEDY" -eq 1 ]; then
    python3 "${REPO_ROOT}/code_eval/coding/evalplus/evalplus/codegen.py" --model "$MODEL" \
                    --dataset $DATASET \
                    --root "$EVALPLUS_ROOT" \
                    --backend vllm \
                    --trust_remote_code \
                    --greedy
    TEMP_VAL="0.0"
else
    echo "Running non-greedy mode"
    python3 "${REPO_ROOT}/code_eval/coding/evalplus/evalplus/codegen.py" --model "$MODEL" \
                    --dataset $DATASET \
                    --root "$EVALPLUS_ROOT" \
                    --backend vllm \
                    --temperature $TEMP \
                    --top-p $TOP_P \
                    --trust_remote_code \
                    --n-samples $N_SAMPLES
    TEMP_VAL="$TEMP"
fi

# The actual output file - use a glob pattern to find the file
echo "Waiting for output file to be generated..."
sleep 2  # Give some time for the file to be created

# Use find to locate the file with a more flexible pattern that matches actual filename format
OUTPUT_FILE=$(find "${EVALPLUS_ROOT}/${DATASET}" -name "*${MODEL_BASE}_vllm_temp_${TEMP_VAL}.jsonl" ! -name "*.raw.jsonl" -type f -print -quit)

if [[ -z "${OUTPUT_FILE}" ]]; then
    echo "Error: generated EvalPlus sample file was not found." >&2
    exit 1
fi

# Run evaluation with found file
python3 -m evalplus.evaluate --dataset "$DATASET" \
    --samples "$OUTPUT_FILE" \
    --output_file "${EVALPLUS_ROOT}/${DATASET}/${MODEL_BASE}_eval_results.json" \
    --min-time-limit 10.0 \
    --gt-time-limit-factor 8.0

echo "Evaluation complete. Results saved to ${EVALPLUS_ROOT}/${DATASET}/${MODEL_BASE}_eval_results.json"
