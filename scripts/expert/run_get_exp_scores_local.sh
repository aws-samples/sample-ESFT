#!/bin/bash

# ============================================================================
# Expert Scores Evaluation - Local Execution
# ============================================================================

set -e

# ==========================
# Configuration
# ==========================

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EVAL_DATASET=""
N_SAMPLE_TOKENS=524288  # -1 means run all data
GPUS_PER_PROCESS=8      # Options: 1, 2, 4, 8
WORLD_SIZE=1

# Paths
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
JOB_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${BASE_DIR}/outputs/expert_scores/job_${JOB_ID}"
LOG_DIR="${BASE_DIR}/outputs/logs/job_${JOB_ID}"

echo "========================================="
echo "Expert Scores - Local Mode"
echo "Model: ${MODEL}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Environment
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Activate venv if exists
[ -f "${BASE_DIR}/.venv/bin/activate" ] && source "${BASE_DIR}/.venv/bin/activate"

cd "${BASE_DIR}"
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"

# Run
python scripts/expert/get_expert_scores_hf.py \
    --eval_dataset="${EVAL_DATASET}" \
    --base_model_path="${MODEL}" \
    --output_dir="${OUTPUT_DIR}" \
    --n_sample_tokens=${N_SAMPLE_TOKENS} \
    --gpus_per_process=${GPUS_PER_PROCESS} \
    --world_size=${WORLD_SIZE} \
    2>&1 | tee "${LOG_DIR}/run.log"

echo "Done. Output: ${OUTPUT_DIR}"
