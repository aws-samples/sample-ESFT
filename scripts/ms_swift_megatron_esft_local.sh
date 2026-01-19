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
GPUS_PER_NODE=8             # Number of GPUs

# Environment settings
PROJECT_ROOT=""             # Project root directory (user must fill in)
ACTIVATE_FILE_PATH="$PROJECT_ROOT/.venv/bin/activate"

# Model settings
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPERT_CONFIG="../results/expert_configs/Qwen3-Coder-30B-A3B-Instruct_expert_config.json"
PARAM_PATTERN="auto"
GENERATE_PARAMS_SCRIPT="./generate_trainable_params.py"

# Dataset settings
TRAIN_DATASETS=""
VAL_DATASETS=""

# Megatron parallelism settings
EXPERT_PARALLEL=8
PIPELINE_PARALLEL=1

# Training hyperparameters
NUM_TRAIN_EPOCHS=3
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=256
LEARNING_RATE="7e-6"
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.10
MAX_LENGTH=16384
LR_DECAY_STYLE="constant"

# Checkpoint settings
SAVE_INTERVAL=91

# WandB settings (optional)
USE_WANDB="false"           # Set to "true" to enable
WANDB_PROJECT_NAME="megatron-esft"
WANDB_RUN_NAME="local_run_$(date +%Y%m%d_%H%M%S)"
WANDB_API_KEY=""
WANDB_ENTITY=""

# Output directories
OUTPUT_DIR="./outputs/$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/$(date +%Y%m%d_%H%M%S)"

# ==========================
# Environment Setup (usually no changes needed)
# ==========================

# Activate virtual environment
if [ -f "$ACTIVATE_FILE_PATH" ]; then
    source "$ACTIVATE_FILE_PATH"
else
    echo "Warning: Virtual environment not found at $ACTIVATE_FILE_PATH"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Distributed training environment variables (single node)
export NNODES=1
export NODE_RANK=0
export WORLD_SIZE=$GPUS_PER_NODE
export NPROC_PER_NODE=$GPUS_PER_NODE
export MASTER_ADDR="localhost"
export MASTER_PORT=29500

# GPU settings
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))

# Cache directories (modify as needed)
export HF_HOME="${HOME}/.cache/huggingface"
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
    export WANDB_API_KEY="${WANDB_API_KEY}"
    export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
    export WANDB_ENTITY="${WANDB_ENTITY}"
    WANDB_ARGS="--wandb_project ${WANDB_PROJECT_NAME} --wandb_exp_name ${WANDB_RUN_NAME} --wandb_save_dir ${LOG_DIR}"
fi

# ==========================
# Generate Trainable Parameters Regex
# ==========================
echo "Generating trainable parameters regex..."
PARAMS_FILE="${LOG_DIR}/trainable_params_regex.txt"

python "$GENERATE_PARAMS_SCRIPT" "$EXPERT_CONFIG" \
    --model "$MODEL" \
    --format regex \
    --pattern "$PARAM_PATTERN" \
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
    --min_lr 0.0 \
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
    $WANDB_ARGS \
    2>&1 | tee "${LOG_DIR}/training.log"

echo "========================================="
echo "Training completed at $(date)"
echo "Output: $OUTPUT_DIR"
echo "Logs: $LOG_DIR"
echo "========================================="
