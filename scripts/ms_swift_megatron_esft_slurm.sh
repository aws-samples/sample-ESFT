#!/bin/bash
#SBATCH --job-name=megatron-esft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
#SBATCH --time=3-00:00:00
#SBATCH --output=/fsx/logs/ms-swift/job_%j/job_%j.out
#SBATCH --error=/fsx/logs/ms-swift/job_%j/job_%j.err
#SBATCH --partition=compute-gpu

# ============================================================================
# MS-Swift Megatron ESFT Training Script (Slurm Version)
# ============================================================================
# ESFT (Expert-Specific Fine-Tuning): Only train selected experts, freeze others
# ============================================================================

# ==========================
# User Configuration - Modify as needed
# ==========================

# Job name (use a descriptive name for identification)
JOB_NAME="megatron-esft"

# Resource settings
NODES=1                      # Number of nodes to use
GPUS_PER_NODE=8             # GPUs per node (usually fixed)
TIME_LIMIT="3-00:00:00"     # Maximum runtime (DD-HH:MM:SS)
PARTITION="compute-gpu"      # Partition to use

# Environment activation
# Project root (location of environment setup)
PROJECT_ROOT=""
ACTIVATE_FILE_PATH="$PROJECT_ROOT/.venv/bin/activate"

source "$ACTIVATE_FILE_PATH"

# Model settings
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"  # Base model path
EXPERT_CONFIG="../results/expert_configs/Qwen3-Coder-30B-A3B-Instruct_expert_config.json"  # ESFT expert config file
PARAM_PATTERN="auto"  # Parameter naming pattern ("auto" for auto-detection, or specify manually)
GENERATE_PARAMS_SCRIPT="./generate_trainable_params.py"  # Parameter generation script path

# Dataset settings
TRAIN_DATASETS=""
VAL_DATASETS=""

# Megatron parallelism settings
EXPERT_PARALLEL=8           # Expert parallelism
PIPELINE_PARALLEL=1         # Pipeline parallelism
# Data Parallel = 8 / (EP * TP * PP * CP) (automatic)

# Training hyperparameters
NUM_TRAIN_EPOCHS=3
MICRO_BATCH_SIZE=1          # Equivalent to per_device_train_batch_size
GLOBAL_BATCH_SIZE=256       # Total batch size (replaces gradient_accumulation)
LEARNING_RATE="7e-6"
WARMUP_RATIO=0.1
WEIGHT_DECAY=0.10
MAX_LENGTH=16384
LR_DECAY_STYLE="constant"   # constant, linear, cosine

# Checkpoint save settings
SAVE_INTERVAL=91            # Save every N steps

# WandB settings
USE_WANDB="true"                                              # Whether to use WandB
WANDB_PROJECT_NAME="megatron-esft"                           # Project name
WANDB_RUN_NAME="${JOB_NAME}_${SLURM_JOB_ID}"                 # Run name
WANDB_API_KEY=""     # API key
WANDB_ENTITY=""                                  # Entity

# FSX directory base paths
FSX_BASE="/fsx"
FSX_DATA_DIR="${FSX_BASE}/data"
FSX_LOGS_DIR="${FSX_BASE}/logs/ms-swift"
FSX_OUTPUT_DIR="${FSX_BASE}/outputs/ms-swift"
FSX_CACHE_DIR="${FSX_BASE}/cache"

# Create log directory
JOB_LOG_DIR="${FSX_LOGS_DIR}/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_LOG_DIR"

# Output directory
OUTPUT_DIR="${FSX_OUTPUT_DIR}/job_${SLURM_JOB_ID}"


# Environment variable setup
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_NODEID
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))
export NPROC_PER_NODE=${GPUS_PER_NODE}
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# GPU settings (assuming 8 GPUs, adjust as needed)
if [ "$GPUS_PER_NODE" -eq 8 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$GPUS_PER_NODE" -eq 4 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    echo "Warning: Non-standard GPU count. Adjust CUDA_VISIBLE_DEVICES manually."
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE - 1)))
fi

# ModelScope/HuggingFace cache directory settings
export MODELSCOPE_CACHE="${FSX_CACHE_DIR}/modelscope"
export HF_HOME="${FSX_CACHE_DIR}/huggingface"
export HF_DATASETS_CACHE="${FSX_CACHE_DIR}/huggingface/datasets"
export TRANSFORMERS_CACHE="${FSX_CACHE_DIR}/huggingface/transformers"

# Triton cache directory settings (using local /tmp)
# Using fixed paths for reusability (possible race condition on first startup only)
export TRITON_CACHE_DIR="/tmp/triton_cache_esft"
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions_esft"
export XDG_CACHE_HOME="/tmp/xdg_cache_esft"

# EFA environment settings
export FI_PROVIDER=efa
#export NCCL_DEBUG=INFO

# PyTorch distributed settings
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONUNBUFFERED=1

# WandB environment variable setup
if [ "$USE_WANDB" = "true" ]; then
    export WANDB_API_KEY="${WANDB_API_KEY}"
    export WANDB_PROJECT="${WANDB_PROJECT_NAME}"
    export WANDB_ENTITY="${WANDB_ENTITY}"
    export WANDB_TAGS="${NODES}nodes,${WORLD_SIZE}gpus,megatron,esft,job${SLURM_JOB_ID}"
    export WANDB_RUN_GROUP="job_${SLURM_JOB_ID}"
fi

# Create WandB config file
if [ "$USE_WANDB" = "true" ]; then
    WCONFIG="${JOB_LOG_DIR}/wandb-config.yaml"
    cat > "$WCONFIG" <<EOF
slurm_nnodes:
  value: ${NNODES}
slurm_node_rank:
  value: ${NODE_RANK}
world_size:
  value: ${WORLD_SIZE}
slurm_master_addr:
  value: "${MASTER_ADDR}"
slurm_job_id:
  value: "${SLURM_JOB_ID}"
megatron_expert_parallel:
  value: ${EXPERT_PARALLEL}
megatron_tensor_parallel:
  value: 1
megatron_pipeline_parallel:
  value: ${PIPELINE_PARALLEL}
megatron_context_parallel:
  value: 1
esft_trainable_params_regex:
  value: "regex_pattern"
model_path:
  value: "${MODEL}"
expert_config:
  value: "${EXPERT_CONFIG}"
EOF
fi

# Set additional arguments for WandB
WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS="--wandb_project ${WANDB_PROJECT_NAME} --wandb_exp_name ${WANDB_RUN_NAME} --wandb_save_dir ${FSX_LOGS_DIR}/job_${SLURM_JOB_ID}"
fi

# Note: Using --use_hf true, so no pre-conversion from HF to Megatron format is needed
# Mcore-Bridge feature allows direct loading from safetensors
# Training with ms-swift using Megatron-LM as backend
srun --export=ALL --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    bash -x -c "
    echo \"Node \$SLURM_NODEID starting at \$(date)...\"
    
    # Environment activation
    PROJECT_ROOT=\"$PROJECT_ROOT\"
    ACTIVATE_FILE_PATH=\"\$PROJECT_ROOT/.venv/bin/activate\"
    
    # Activate virtual environment
    if [ -f \"\$ACTIVATE_FILE_PATH\" ]; then
        source \"\$ACTIVATE_FILE_PATH\"
    else
        echo \"Warning: Virtual environment not found at \$ACTIVATE_FILE_PATH\"
    fi
    
    # Verify ms-swift version
    echo \"=========================================\"
    echo \"Node \$SLURM_NODEID: Verifying ms-swift installation...\"
    python -c \"import swift; print(f'ms-swift version: {swift.__version__}'); print(f'ms-swift path: {swift.__file__}')\"
    echo \"=========================================\"
    
    # Re-export environment variables
    export NNODES=$NNODES
    export NODE_RANK=\$SLURM_NODEID
    export WORLD_SIZE=$WORLD_SIZE
    export NPROC_PER_NODE=$NPROC_PER_NODE
    export MASTER_ADDR=$MASTER_ADDR
    export MASTER_PORT=$MASTER_PORT
    export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
    
    # Re-export WandB environment variables
    if [ \"$USE_WANDB\" = \"true\" ]; then
        export WANDB_PROJECT='$WANDB_PROJECT_NAME'
        export WANDB_ENTITY='$WANDB_ENTITY'
        export WANDB_TAGS='${NODES}nodes,${WORLD_SIZE}gpus,megatron,esft,job${SLURM_JOB_ID}'
        export WANDB_RUN_GROUP='job_${SLURM_JOB_ID}'
        
        if [ \"\$SLURM_NODEID\" = \"0\" ]; then
            export WANDB_RUN_ID='${SLURM_JOB_ID}'
        else
            unset WANDB_RUN_ID
        fi
    fi
    
    # Create working directory for each node
    WORK_DIR=\"/tmp/ms-swift-work-${SLURM_JOB_ID}-node${SLURM_NODEID}\"
    mkdir -p \"\$WORK_DIR\"
    cd \"\$WORK_DIR\"
    
    # Copy WandB config file
    if [ \"$USE_WANDB\" = \"true\" ] && [ -f \"$WCONFIG\" ]; then
        cp \"$WCONFIG\" ./config-defaults.yaml
    fi
    
    # Additional environment variables for PyTorch distributed
    export RANK=\$((NODE_RANK * NPROC_PER_NODE))
    export LOCAL_RANK=0
    export LOCAL_WORLD_SIZE=$NPROC_PER_NODE
    
    # Cache directory settings
    export MODELSCOPE_CACHE=\"${FSX_CACHE_DIR}/modelscope\"
    export HF_HOME=\"${FSX_CACHE_DIR}/huggingface\"
    export HF_DATASETS_CACHE=\"${FSX_CACHE_DIR}/huggingface/datasets\"
    export TRANSFORMERS_CACHE=\"${FSX_CACHE_DIR}/huggingface/transformers\"
    
    # Local cache directories
    # Using fixed paths for reusability (possible race condition on first startup only)
    export TRITON_CACHE_DIR=\"/tmp/triton_cache_esft\"
    export TORCH_EXTENSIONS_DIR=\"/tmp/torch_extensions_esft\"
    export XDG_CACHE_HOME=\"/tmp/xdg_cache_esft\"

    mkdir -p \"\$TRITON_CACHE_DIR\"
    mkdir -p \"\$TORCH_EXTENSIONS_DIR\"
    mkdir -p \"\$XDG_CACHE_HOME\"
    
    # Generate trainable parameters regex on each node independently
    PARAMS_FILE=\"${FSX_LOGS_DIR}/job_${SLURM_JOB_ID}/trainable_params_regex.txt\"
    
    if [ \"\$NODE_RANK\" = \"0\" ]; then
        mkdir -p \"\$(dirname \"\$PARAMS_FILE\")\"
        python \"$GENERATE_PARAMS_SCRIPT\" \"$EXPERT_CONFIG\" \
            --model \"$MODEL\" \
            --format regex \
            --pattern \"$PARAM_PATTERN\" \
            --megatron \
            --ep-size $EXPERT_PARALLEL \
            --output \"\$PARAMS_FILE\" 2>&1 | grep -v '^\['
        
        if [ ! -f \"\$PARAMS_FILE\" ] || [ ! -s \"\$PARAMS_FILE\" ]; then
            echo \"Node 0: Error generating trainable parameters regex\"
            exit 1
        fi
    else
        # Wait for node 0 to generate the file
        TIMEOUT=180
        ELAPSED=0
        while [ ! -f \"\$PARAMS_FILE\" ] && [ \$ELAPSED -lt \$TIMEOUT ]; do
            sleep 1
            ELAPSED=\$((ELAPSED + 1))
        done
        
        if [ ! -f \"\$PARAMS_FILE\" ]; then
            echo \"Node \$NODE_RANK: Timeout waiting for parameters file\"
            exit 1
        fi
    fi
    
    TRAINABLE_PARAMS_REGEX=\$(cat \"\$PARAMS_FILE\")
    
    # Run ms-swift megatron pt using torchrun
    torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --nnodes=$NNODES \
        --node_rank=\$NODE_RANK \
        --master_addr=$MASTER_ADDR \
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
        --trainable_parameters_regex \"\$TRAINABLE_PARAMS_REGEX\" \
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
        2>&1 | tee -a ${JOB_LOG_DIR}/job_${SLURM_JOB_ID}_node\${NODE_RANK}.log
    
    echo \"Node \$NODE_RANK completed at \$(date)\"
    "

echo "========================================="
echo "Training completed at $(date)"
echo "Output saved to: $OUTPUT_DIR"
echo "Logs saved to: $JOB_LOG_DIR"
echo "========================================="

# Check checkpoints after training completion
echo "Checking for saved checkpoints..."
if [ -d "$OUTPUT_DIR" ]; then
    echo "Output directory contents:"
    ls -la "$OUTPUT_DIR"
    
    if ls "$OUTPUT_DIR"/*/checkpoint-* 1> /dev/null 2>&1; then
        echo "✅ Checkpoints found!"
        ls -la "$OUTPUT_DIR"/*/checkpoint-*
    else
        echo "⚠️ No checkpoints found. Checking for model files..."
        find "$OUTPUT_DIR" -name "*.safetensors" -o -name "*.bin" -o -name "*.pth" 2>/dev/null
    fi
    
    echo ""
    echo "Disk usage for output directory:"
    du -sh "$OUTPUT_DIR"
else
    echo "⚠️ Output directory not found: $OUTPUT_DIR"
fi

# Check FSX cache directory sizes
echo ""
echo "FSX cache directory sizes:"
du -sh "${FSX_CACHE_DIR}/modelscope" 2>/dev/null || echo "ModelScope cache: Not yet created"
du -sh "${FSX_CACHE_DIR}/huggingface" 2>/dev/null || echo "HuggingFace cache: Not yet created"

echo "========================================="
echo "Job ${SLURM_JOB_ID} completed successfully!"
echo "========================================="
