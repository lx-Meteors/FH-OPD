#!/usr/bin/env bash

set -euo pipefail

DATASET_REPO="${DATASET_REPO:-Keven16/G-OPD-Training-Data}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${1:-${REPO_ROOT}/../G-OPD-Training-Data}"

if ! command -v hf >/dev/null 2>&1; then
    echo "Error: the 'hf' command is not available in the current environment." >&2
    echo "Activate the environment that contains huggingface_hub, then run this script again." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Downloading dataset: ${DATASET_REPO}"
echo "Destination: ${OUTPUT_DIR}"

download_args=(
    download "${DATASET_REPO}"
    --repo-type dataset
    --local-dir "${OUTPUT_DIR}"
)

if [[ -n "${HF_TOKEN:-}" ]]; then
    download_args+=(--token "${HF_TOKEN}")
fi

hf "${download_args[@]}"

required_files=(
    "DeepMath-103K/train_filtered_level6.parquet"
    "AIME2024/test.parquet"
    "AIME2025/test.parquet"
    "Eurus/code_train.parquet"
    "Eurus/code_validation.parquet"
)

optional_files=(
    "DeepMath-103K/train_filtered_level6_with_one_correct_solution.parquet"
    "math_and_code/train.parquet"
)

missing_required=0
for relative_path in "${required_files[@]}"; do
    if [[ ! -f "${OUTPUT_DIR}/${relative_path}" ]]; then
        echo "Missing required file: ${relative_path}" >&2
        missing_required=1
    fi
done

for relative_path in "${optional_files[@]}"; do
    if [[ ! -f "${OUTPUT_DIR}/${relative_path}" ]]; then
        echo "Optional file not found: ${relative_path}" >&2
    fi
done

if [[ "${missing_required}" -ne 0 ]]; then
    echo "Dataset download finished, but required Math/Code single-teacher files are missing." >&2
    exit 1
fi

echo "Dataset is ready at: ${OUTPUT_DIR}"
