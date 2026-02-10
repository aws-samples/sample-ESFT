#!/bin/bash
# ============================================================================
# MS-Swift Megatron ESFT Training Script (Local/Single-Node Version)
# ============================================================================
# ESFT (Expert-Specific Fine-Tuning): Only train selected experts, freeze others
# ============================================================================

set -e  # Exit on error

# ==========================
# User Configuration
# ==========================

# Resource settings
GPUS_PER_NODE="${SM_HP_GPUS_PER_PROCESS:-8}"  # Number of GPUs

# SageMaker paths
if [ -n "$SM_MODEL_DIR" ]; then
    PROJECT_ROOT="/opt/ml/code"
    OUTPUT_DIR="${SM_MODEL_DIR}"
    EXPERT_CONFIG_DIR="${SM_HP_EXPERT_CONFIG_DIR:-${SM_OUTPUT_DATA_DIR}/results/expert_scores.json}"
    LOG_DIR="${SM_OUTPUT_DATA_DIR}/logs"
    JOB_ID="${TRAINING_JOB_NAME:-sagemaker_$(date +%Y%m%d_%H%M%S)}"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    OUTPUT_DIR="${PROJECT_ROOT}/outputs/model/job_${JOB_ID}"
    EXPERT_CONFIG_DIR="${SM_HP_EXPERT_CONFIG_DIR:-${PROJECT_ROOT}/outputs/results/expert_configs/job_${JOB_ID}.json}"
    LOG_DIR="${PROJECT_ROOT}/outputs/logs/job_${JOB_ID}"
    JOB_ID="sagemaker_test_$(date +%Y%m%d_%H%M%S)"
fi

# Model settings
MODEL="${SM_HP_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
MODEL_ARGS=""
if [[ "$MODEL" == *gpt-oss* ]]; then
    # GPT-OSS models have special arguments that may cause CLI parsing issues
    # Use ignore_unknown_args to handle model-specific arguments gracefully
    MODEL_ARGS="--padding_free false --template_backend jinja --ignore_args_error"
fi

# TODO Dataset settings
TRAIN_DATASETS="${SM_HP_TRAIN_DATASET:-datasets/train/intent.jsonl}"
VAL_DATASETS="${SM_HP_EVAL_DATASET:-datasets/eval/intent.jsonl}"

# Megatron parallelism settings
EXPERT_PARALLEL="${SM_HP_EXPERT_PARALLEL:-$GPUS_PER_NODE}"
PIPELINE_PARALLEL="${SM_HP_PIPELINE_PARALLEL:-1}"
TENSOR_PARALLEL=1  # ESFT currently doesn't support TP

# Training hyperparameters
NUM_TRAIN_EPOCHS="${SM_HP_TRAIN_EPOCHS:-1}"
MICRO_BATCH_SIZE="${SM_HP_MICRO_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${SM_HP_GLOBAL_BATCH_SIZE:-256}"
LEARNING_RATE="${SM_HP_LEARNING_RATE:-7e-6}"
WARMUP_RATIO="${SM_HP_WARMUP_RATIO:-0.1}"
WEIGHT_DECAY="${SM_HP_WEIGHT_DECAY:-0.1}"
MIN_LEARNING_RATE="${SM_HP_MIN_LEARNING_RATE:-0.0}"
MAX_LENGTH="${SM_HP_MAX_LENGTH:-16384}"
LR_DECAY_STYLE="${SM_HP_LR_DECAY_STYLE:-cosine}"

# Checkpoint settings
SAVE_INTERVAL="${SM_HP_SAVE_INTERVAL:-100}"

# WandB settings (optional)
USE_WANDB="${SM_HP_USE_WANDB:-false}"  # Set to "true" to enable and make sure setting WANDB_API_KEY / WANDB_ENTITY

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Distributed training environment variables (single node)
export NNODES=1  # ESFT currently only supports single node
export NODE_RANK=0
export WORLD_SIZE=$GPUS_PER_NODE
export NPROC_PER_NODE=$GPUS_PER_NODE
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

# GPU settings
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# Cache directories (modify as needed)
export HF_HOME="/opt/ml/model/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TRITON_CACHE_DIR="/tmp/triton_cache_esft"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_esft"

mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# PyTorch debugging (optional, enable for debugging)
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONUNBUFFERED=1

# ==========================
# WandB Configuration
# ==========================
WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_PROJECT="esft-sagemaker"
    WANDB_RUN_NAME="sagemaker_run_$(date +%Y%m%d_%H%M%S)"
    if [ -z "${WANDB_API_KEY}" ]; then
        echo "Error: WANDB_API_KEY is not set"
        exit 1
    fi
    if [ -z "${WANDB_ENTITY}" ]; then
        echo "Error: WANDB_ENTITY is not set"
        exit 1
    fi
    WANDB_ARGS="--wandb_project ${WANDB_PROJECT} --wandb_exp_name ${WANDB_RUN_NAME} --wandb_save_dir ${LOG_DIR}"
fi

# ==========================
# Generate Trainable Parameters Regex
# ==========================
echo "Generating trainable parameters regex..."
PARAMS_FILE="${LOG_DIR}/trainable_params_regex.txt"

python scripts/generate_trainable_params.py "$EXPERT_CONFIG_DIR" \
    --model "$MODEL" \
    --format regex \
    --pattern auto \
    --megatron \
    --ep-size $EXPERT_PARALLEL \
    --output "$PARAMS_FILE" 2>&1 | grep -v '^\['

if [ ! -f "$PARAMS_FILE" ] || [ ! -s "$PARAMS_FILE" ]; then
    echo "Error: Failed to generate trainable parameters regex"
    exit 1
fi

TRAINABLE_PARAMS_REGEX=$(cat "$PARAMS_FILE")
echo "Trainable params regex generated: $PARAMS_FILE"

# ==========================
# Start Training
# ==========================
echo "========================================="
echo "Starting training at $(date)"
echo "Model: $MODEL"
echo "GPUs: $GPUS_PER_NODE"
echo "Output: $OUTPUT_DIR"
echo "========================================="

torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    -m swift.cli._megatron.pt \
    --model $MODEL \
    $MODEL_ARGS \
    --dataset $TRAIN_DATASETS \
    --val_dataset $VAL_DATASETS \
    --use_hf true \
    --load_safetensors true \
    --save_safetensors true \
    --no_initialization true \
    --freeze_parameters_ratio 1.0 \
    --trainable_parameters_regex "$TRAINABLE_PARAMS_REGEX" \
    --expert_model_parallel_size $EXPERT_PARALLEL \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size $PIPELINE_PARALLEL \
    --context_parallel_size 1 \
    --sequence_parallel \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --global_batch_size $GLOBAL_BATCH_SIZE \
    --max_epochs $NUM_TRAIN_EPOCHS \
    --lr $LEARNING_RATE \
    --lr_decay_style $LR_DECAY_STYLE \
    --lr_warmup_fraction $WARMUP_RATIO \
    --min_lr $MIN_LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --seq_length $MAX_LENGTH \
    --save_interval $SAVE_INTERVAL \
    --log_interval 5 \
    --attn_impl 'sdpa' \
    --recompute_granularity selective \
    --moe_grouped_gemm false \
    --num_workers 4 \
    --save $OUTPUT_DIR \
    --no_save_optim \
    --use_distributed_optimizer true \
    --tensorboard_dir /opt/ml/output/tensorboard \
    --tensorboard_log_interval 1 \
    $WANDB_ARGS \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "========================================="
echo "Training completed at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "========================================="
