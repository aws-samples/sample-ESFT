#!/bin/bash
#SBATCH --job-name=esft-expert-scores-mp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/fsx/logs/test/job_%j/job_%j.out
#SBATCH --error=/fsx/logs/test/job_%j/job_%j.err
#SBATCH --partition=compute-gpu

# ============================================================================
# Expert Scores Evaluation - Multi-Process Mode
# Flexible GPU allocation: 1, 2, 4, or 8 GPUs per process
# ============================================================================

# ==========================
# Configuration
# ==========================

JOB_NAME="esft-expert-scores-mp"

# Model and Data
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
#MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
#MODEL="zai-org/GLM-4.5-Air"
EVAL_DATASET=""
N_SAMPLE_TOKENS=524288  # -1 means run out of the eval dataset

# GPU Configuration
# For 235B model: use 8 GPUs per process (1 process total)
# For 30B model: use 2 GPUs per process (4 processes total)
# For 7B model: use 1 GPU per process (8 processes total)
GPUS_PER_PROCESS=8  # Options: 1, 2, 4, 8
WORLD_SIZE=1        # Will be auto-calculated if not set (leave as 1 for auto)

# Paths
FSX_BASE="/fsx"
FSX_LOGS_DIR="${FSX_BASE}/logs/test"
FSX_OUTPUT_DIR="${FSX_BASE}/outputs/esft"
FSX_CACHE_DIR="${FSX_BASE}/cache"
WORK_DIR="./"

# Job directories
JOB_LOG_DIR="${FSX_LOGS_DIR}/job_${SLURM_JOB_ID}"
OUTPUT_DIR="${FSX_OUTPUT_DIR}/expert_scores_job_${SLURM_JOB_ID}"

echo "========================================="
echo "Expert Scores - Multi-Process Mode"
echo "========================================="
echo "Time: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs per process: ${GPUS_PER_PROCESS}"
echo "Model: ${MODEL}"
echo "Dataset: ${EVAL_DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================="

# Create directories
mkdir -p "${JOB_LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${FSX_CACHE_DIR}/modelscope"
mkdir -p "${FSX_CACHE_DIR}/huggingface"

# Cache directories
export MODELSCOPE_CACHE="${FSX_CACHE_DIR}/modelscope"
export HF_HOME="${FSX_CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${FSX_CACHE_DIR}/huggingface/datasets"
export TRANSFORMERS_CACHE="${FSX_CACHE_DIR}/huggingface/transformers"

# Local cache
export TRITON_CACHE_DIR="/tmp/triton_cache_${SLURM_JOB_ID}"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_${SLURM_JOB_ID}"
export XDG_CACHE_HOME="/tmp/xdg_cache_${SLURM_JOB_ID}"

# PyTorch settings
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "Environment configured"
echo "========================================="

# Activate environment
echo "Activating environment..."
source "${WORK_DIR}/.venv/bin/activate"

# Check PyTorch and transformers
echo "Checking PyTorch and transformers..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
nvidia-smi --list-gpus

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "========================================="

# Change to work directory
cd "${WORK_DIR}"
export PYTHONPATH="${PYTHONPATH}:${WORK_DIR}"

# Run with srun (SLURM job execution)
echo "Starting evaluation..."
echo "========================================="

srun --export=ALL \
     --nodes=1 \
     --ntasks=1 \
     --gpus-per-node=8 \
     bash -c "
    # Re-activate environment on compute node
    source '${WORK_DIR}/.venv/bin/activate'
    
    # Cache directories
    export MODELSCOPE_CACHE='${FSX_CACHE_DIR}/modelscope'
    export HF_HOME='${FSX_CACHE_DIR}/huggingface'
    export HF_DATASETS_CACHE='${FSX_CACHE_DIR}/huggingface/datasets'
    export TRANSFORMERS_CACHE='${FSX_CACHE_DIR}/huggingface/transformers'
    
    # Local cache
    export TRITON_CACHE_DIR='/tmp/triton_cache_${SLURM_JOB_ID}'
    export TORCH_EXTENSIONS_DIR='/tmp/torch_extensions_${SLURM_JOB_ID}'
    export XDG_CACHE_HOME='/tmp/xdg_cache_${SLURM_JOB_ID}'
    
    mkdir -p \"\${TRITON_CACHE_DIR}\"
    mkdir -p \"\${TORCH_EXTENSIONS_DIR}\"
    mkdir -p \"\${XDG_CACHE_HOME}\"
    
    # PyTorch settings
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    
    cd '${WORK_DIR}'
    export PYTHONPATH=\${PYTHONPATH}:'${WORK_DIR}'
    
    echo \"Starting multi-process evaluation on compute node...\"
    
    # Run Python with multiprocessing
    python scripts/expert/get_expert_scores_hf.py \
           --eval_dataset='${EVAL_DATASET}' \
           --base_model_path='${MODEL}' \
           --output_dir='${OUTPUT_DIR}' \
           --n_sample_tokens=${N_SAMPLE_TOKENS} \
           --gpus_per_process=${GPUS_PER_PROCESS} \
           --world_size=${WORLD_SIZE} \
           2>&1 | tee '${JOB_LOG_DIR}/node_${SLURM_JOB_ID}.log'
    
    echo \"Evaluation completed at \$(date)\"
"

echo "========================================="
echo "Job completed at $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${JOB_LOG_DIR}"
echo "========================================="

# Cleanup temp directories
echo "Cleaning up temporary directories..."
rm -rf "/tmp/triton_cache_${SLURM_JOB_ID}" 2>/dev/null || true
rm -rf "/tmp/torch_extensions_${SLURM_JOB_ID}" 2>/dev/null || true
rm -rf "/tmp/xdg_cache_${SLURM_JOB_ID}" 2>/dev/null || true

echo "Job ${SLURM_JOB_ID} completed successfully!"
