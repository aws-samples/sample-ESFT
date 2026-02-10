#!/bin/bash

# ============================================================================
# Expert Scores Evaluation - SageMaker Execution
# ============================================================================

set -e

# ==========================
# Configuration from Environment Variables (SageMaker hyperparameters)
# ==========================

# Default values
## Cache directories
export HF_HOME="/opt/ml/model/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

## Scoring hparams
MODEL="${SM_HP_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
EVAL_DATASET="${SM_HP_EVAL_DATASET:-datasets/train/intent.jsonl}"
N_SAMPLE_TOKENS="${SM_HP_N_SAMPLE_TOKENS:-131072}"
GPUS_PER_PROCESS="${SM_HP_GPUS_PER_PROCESS:-8}"
WORLD_SIZE="${SM_HP_WORLD_SIZE:-1}"

## Generating config hparams
SCORE_FUNC="${SM_HP_SCORE_FUNCTION:-token}"
SCORE_THRESHOLD="${SM_HP_SCORE_THRESHOLD:-0.2}"
IS_TRAIN_SHARED_EXPERTS=${SM_HP_IS_TRAIN_SHARED_EXPERTS:-false}
IS_TRAIN_NON_EXPERT_MODULES=${SM_HP_IS_TRAIN_NON_EXPERT_MODULES:-false}

# SageMaker paths
if [ -n "$SM_MODEL_DIR" ]; then
    # Running in SageMaker
    BASE_DIR="/opt/ml/code"
    OUTPUT_DIR="${SM_OUTPUT_DATA_DIR}/expert_scores"
    EXPERT_CONFIG_DIR="${SM_OUTPUT_DATA_DIR}/results/expert_configs.json"
    LOG_DIR="${SM_OUTPUT_DATA_DIR}/logs"
    JOB_ID="${TRAINING_JOB_NAME:-sagemaker_$(date +%Y%m%d_%H%M%S)}"
else
    # Fallback for testing outside SageMaker
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    OUTPUT_DIR="${BASE_DIR}/outputs/expert_scores/job_${JOB_ID}"
    EXPERT_CONFIG_DIR="${BASE_DIR}/outputs/results/expert_configs/job_${JOB_ID}.json"
    LOG_DIR="${BASE_DIR}/outputs/logs/job_${JOB_ID}"
    JOB_ID="sagemaker_test_$(date +%Y%m%d_%H%M%S)"
fi

echo "========================================="
echo "Expert Scores - SageMaker Mode"
echo "Model: ${MODEL}"
echo "Dataset: ${EVAL_DATASET}"
echo "Sample Tokens: ${N_SAMPLE_TOKENS}"
echo "GPUs per Process: ${GPUS_PER_PROCESS}"
echo "World Size: ${WORLD_SIZE}"
echo "Job ID: ${JOB_ID}"
echo "Base Dir: ${BASE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Log Dir: ${LOG_DIR}"
echo "========================================="

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Environment
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Activate venv if exists (for local testing)
if [ -f "${BASE_DIR}/.venv/bin/activate" ] && [ -z "$SM_MODEL_DIR" ]; then
    echo "Activating local virtual environment..."
    source "${BASE_DIR}/.venv/bin/activate"
fi

cd "${BASE_DIR}"
export PYTHONPATH="${PYTHONPATH}:${BASE_DIR}"

# Log system information
echo "========================================="
echo "System Information:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Available GPUs: $(nvidia-smi -L 2>/dev/null | wc -l || echo 'nvidia-smi not available')"
echo "Python path: $(which python)"
echo "Working directory: $(pwd)"
echo "========================================="

# Run the expert scores collection
echo "Starting expert scores collection..."
python scripts/expert/get_expert_scores_hf.py \
    --eval_dataset="${EVAL_DATASET}" \
    --base_model_path="${MODEL}" \
    --output_dir="${OUTPUT_DIR}" \
    --n_sample_tokens=${N_SAMPLE_TOKENS} \
    --gpus_per_process=${GPUS_PER_PROCESS} \
    --world_size=${WORLD_SIZE} \
    2>&1 | tee "${LOG_DIR}/scoring.log"

echo "Expert scores collection finished."

echo "========================================="
echo "Generating Expert Config - SageMaker Mode"
echo "Expert scores: ${OUTPUT_DIR}"
echo "Score function: ${SCORE_FUNC}"
echo "Score threshold: ${SCORE_THRESHOLD}"
echo "Train shared experts: ${IS_TRAIN_SHARED_EXPERTS}"
echo "Train non-expert modules: ${IS_TRAIN_NON_EXPERT_MODULES}"
echo "Expert config: ${EXPERT_CONFIG_DIR}"
echo "========================================="

# Generating the expert config
echo "Generating expert config..."

# Create the results directory
mkdir -p "$(dirname "${EXPERT_CONFIG_DIR}")"

# Build command arguments array
CMD_ARGS=(
    "scripts/expert/generate_expert_config.py"
    "--model_name_or_path" "${MODEL}"
    "--expert_scores_dir" "${OUTPUT_DIR}"
    "--output_path" "${EXPERT_CONFIG_DIR}"
    "--score_function" "${SCORE_FUNC}"
    "--top_p" "${SCORE_THRESHOLD}"
)

if [ "${IS_TRAIN_SHARED_EXPERTS}" = "true" ]; then
    CMD_ARGS+=("--train_shared_experts")
fi

if [ "${IS_TRAIN_NON_EXPERT_MODULES}" = "true" ]; then
    CMD_ARGS+=("--train_non_expert_modules")
fi

# Debug: Print the command that will be executed
echo "Command to execute:"
echo "python ${CMD_ARGS[@]}"
echo ""

# Run the expert config generation script
python "${CMD_ARGS[@]}" 2>&1 | tee -a "${LOG_DIR}/scoring.log"

echo "========================================="
echo "Expert scores collection completed!"
echo "Expert scores: ${OUTPUT_DIR}"
echo "Expert config: ${EXPERT_CONFIG_DIR}"
echo "Log directory: ${LOG_DIR}"
echo "========================================="