# ESFT: Expert-Specialized Fine-Tuning

ESFT (Expert-Specialized Fine-Tuning) is an efficient fine-tuning method for MoE (Mixture-of-Experts) architecture LLMs. By training only task-relevant experts, it significantly reduces computational resources and storage requirements while maintaining performance.

## Core Concept

1. **Collect Expert Routing Statistics**: Run inference on training data to record which experts each token is routed to
2. **Select Task-Relevant Experts**: Based on routing statistics, select experts that cumulatively handle top-p (e.g., 20%) of tokens
3. **Freeze Other Parameters**: Train only selected experts while freezing all other parameters

## Project Structure

```
ESFT/
├── model_patch/                    # MoE model monkey patch module
│   ├── __init__.py                 # Auto-apply all patches
│   ├── patch_qwen2_moe.py          # Qwen2 MoE support
│   ├── patch_qwen3_moe.py          # Qwen3 MoE support
│   ├── patch_glm4_moe.py           # GLM-4.5 MoE support
│   └── patch_gpt_oss.py            # GPT-OSS support
├── scripts/
│   ├── expert/                     # Expert analysis scripts
│   │   ├── get_expert_scores_hf.py       # Collect expert routing statistics
│   │   ├── generate_expert_config.py     # Generate expert config file
│   │   └── run_get_exp_scores_hf.sh      # SLURM submission script
│   ├── generate_trainable_params.py      # Generate trainable parameter list
│   └── ms_swift_megatron_esft_*.sh       # Training script examples
├── results/                        # Output directory
│   └── expert_configs/             # Generated expert config files
└── utils.py                        # Utility functions
```

## Supported Models

| Model Family | Example Model | Shared Expert |
|-------------|---------------|---------------|
| Qwen2 MoE | Qwen2-57B-A14B-Instruct | ✅ |
| Qwen3 MoE | Qwen3-Coder-30B-A3B-Instruct | ❌ |
| GLM-4 MoE | GLM-4.5-Air | ✅ |
| GPT-OSS | gpt-oss-20b | ❌ |

## Quick Start

### Environment Setup

Please use the virtual environment at `/fsx/users/pengin/ESFT/.venv`, which is built on Python 3.11 and the latest ms-swift.

```bash
source /fsx/users/pengin/ESFT/.venv/bin/activate
```

**Note**: ms-swift uses a modified version at `/fsx/users/pengin/ms-swift`. The main modification is in `swift/megatron/model/gpt_bridge.py`, which fixes bugs in mcore_bridge for loading and saving model checkpoints in ESFT scenarios (where only partial experts participate in training).

### Step 1: Collect Expert Routing Statistics

Run inference on training data to record expert routing information for each token:

```bash
# Edit configuration
vim scripts/expert/run_get_exp_scores_hf.sh

# Main configuration items:
MODEL="zai-org/GLM-4.5-Air"                    # MoE model path
EVAL_DATASET="/path/to/train.jsonl"            # Training data (jsonl format with 'messages' field)
N_SAMPLE_TOKENS=524288                         # Number of tokens to sample (-1 for all)
GPUS_PER_PROCESS=8                             # GPUs per process

# Submit job
sbatch scripts/expert/run_get_exp_scores_hf.sh
```

**GPU Configuration Recommendations**:
- 235B model (GLM-4.5-Air): `GPUS_PER_PROCESS=8` (1 process)
- 30B model (Qwen3-Coder-30B-A3B): `GPUS_PER_PROCESS=2` (4 processes)
- 7B model: `GPUS_PER_PROCESS=1` (8 processes)

**Output**: Expert routing logs saved to `/fsx/outputs/esft/expert_scores_job_<JOB_ID>/`

### Step 2: Generate Expert Configuration File

Generate `expert_config.json` based on routing statistics:

```bash
python scripts/expert/generate_expert_config.py \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --expert_scores_dir /path/to/expert_scores_output \
    --output_path ./results/expert_configs/my_expert_config.json \
    --score_function token \
    --top_p 0.2 \
    --train_shared_experts \
    --train_non_expert_modules
```

**Parameter Description**:
| Parameter | Description |
|-----------|-------------|
| `--model_name_or_path` | HuggingFace model name or local path (auto-reads n_layers, n_experts, top_k) |
| `--expert_scores_dir` | Routing log directory from Step 1 |
| `--output_path` | Output path for expert_config.json |
| `--score_function` | `token` (count-based) or `gate` (weight-based) |
| `--top_p` | Cumulative score threshold (e.g., 0.2 = select experts handling 20% of tokens) |
| `--train_shared_experts` | Whether to train shared experts (optional) |
| `--train_non_expert_modules` | Whether to train attention, embedding, and other non-expert modules (optional) |

**Output Format**:
```json
{
  "experts": {
    "1": [79, 51, 122, 70, 96],
    "2": [42, 78, 22],
    ...
  },
  "shared_experts": true,
  "non_expert_modules": false
}
```

- `experts`: Dictionary with layer number (1-indexed) as key and list of expert IDs to train as value
- `shared_experts`: Whether to train shared experts
- `non_expert_modules`: Whether to train non-expert modules

### Step 3: Execute ESFT Training with ms-swift Megatron

#### Configure Training Script

Edit `scripts/ms_swift_megatron_esft_sample.sh`:

```bash
# Model configuration
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPERT_CONFIG="./results/expert_configs/my_expert_config.json"
PARAM_PATTERN="auto"  # Auto-detect (recommended) or specify manually

# Dataset
TRAIN_DATASETS="/path/to/train.jsonl"
VAL_DATASETS="/path/to/val.jsonl"

# Parallelism configuration (adjust based on GPU count)
EXPERT_PARALLEL=8
TENSOR_PARALLEL=1
PIPELINE_PARALLEL=1
CONTEXT_PARALLEL=1

# Training parameters
NUM_TRAIN_EPOCHS=3
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
LEARNING_RATE="1e-5"
MAX_LENGTH=16384
```

#### Submit Training Job

```bash
sbatch scripts/ms_swift_megatron_esft_sample.sh
```

#### Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f /fsx/logs/ms-swift/job_<JOB_ID>/job_<JOB_ID>_node0.log
```

#### Configuration Details

**Parallelism Strategy**

Ensure `EXPERT_PARALLEL × TENSOR_PARALLEL × PIPELINE_PARALLEL × DATA_PARALLEL = Total GPUs`

> ⚠️ Due to ms-swift's mcore_bridge implementation limitations, `TENSOR_PARALLEL` and `CONTEXT_PARALLEL` are currently not supported.

For MoE models, prioritize `EXPERT_PARALLEL` to parallelize experts.

**Batch Size Calculation**

`GLOBAL_BATCH_SIZE` must be divisible by `MICRO_BATCH_SIZE × data_parallel_size`

Where: `data_parallel_size = Total_GPUs / (EP × TP × PP × CP)`

Example: 16 GPUs, EP=8, TP=1, PP=1, CP=1
- data_parallel_size = 16 / 8 = 2
- GLOBAL_BATCH_SIZE must be divisible by 2 (e.g., 2, 4, 8, 16, ...)

**Parameter Naming Pattern**

The script auto-detects by default (`PARAM_PATTERN="auto"`). Common patterns:

| Model | Parameter Pattern |
|-------|-------------------|
| Qwen3-MoE, DeepSeek-V2 | `model.layers.{layer}.mlp.experts.{expert}` |
| Some MoE models | `model.layers.{layer}.experts.{expert}` |
| Mixtral | `model.layers.{layer}.block_sparse_moe.experts.{expert}` |

Manual specification:
```bash
PARAM_PATTERN="model.layers.{layer}.mlp.experts.{expert}"
```

## Additional Tools

### Manual Trainable Parameter Generation

Test parameter generation before training:

```bash
# Auto-detect mode
python scripts/generate_trainable_params.py expert_config.json \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --pattern auto \
    --format regex \
    --megatron \
    --ep-size 8

# Use specific pattern
python scripts/generate_trainable_params.py expert_config.json \
    --pattern "model.layers.{layer}.mlp.experts.{expert}"
```

### Adding New Model Support

Refer to `model_patch/README.md` for instructions on adding expert routing log support for new MoE models.

## Citation

Original paper: [Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models](https://arxiv.org/abs/2407.01906)

## License

See [LICENSE-CODE](./LICENSE-CODE) for details.
