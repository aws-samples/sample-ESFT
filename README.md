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
├── ms-swift/                       # ms-swift (git submodule)
├── scripts/
│   ├── expert/                     # Expert analysis scripts
│   │   ├── get_expert_scores_hf.py       # Collect expert routing statistics
│   │   ├── generate_expert_config.py     # Generate expert config file
│   │   ├── run_get_exp_scores_local.sh   # Local execution script
│   │   └── run_get_exp_scores_slurm.sh   # SLURM submission script
│   ├── generate_trainable_params.py      # Generate trainable parameter list
│   ├── ms_swift_megatron_esft_local.sh   # Local training script
│   └── ms_swift_megatron_esft_slurm.sh   # SLURM training script
├── results/                        # Output directory
│   ├── expert_configs/             # Generated expert config files
│   └── expert_scores/              # Expert routing statistics
├── test_chat_template.py           # Chat template test script
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

```bash
# Clone with submodule
git clone --recurse-submodules <REPO_URL>
cd ESFT

# Or if already cloned
git submodule update --init --recursive

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ./ms-swift
```

### Step 1: Collect Expert Routing Statistics

Run inference on training data to record expert routing information for each token.

#### Local Execution

```bash
# Edit configuration
vim scripts/expert/run_get_exp_scores_local.sh

# Main configuration:
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EVAL_DATASET="/path/to/train.jsonl"
N_SAMPLE_TOKENS=524288
GPUS_PER_PROCESS=8

# Run
bash scripts/expert/run_get_exp_scores_local.sh
```

#### SLURM Cluster

```bash
# Edit configuration
vim scripts/expert/run_get_exp_scores_slurm.sh

# Submit job
sbatch scripts/expert/run_get_exp_scores_slurm.sh
```

**GPU Configuration Recommendations**:
- 235B model (GLM-4.5-Air): `GPUS_PER_PROCESS=8` (1 process)
- 30B model (Qwen3-Coder-30B-A3B): `GPUS_PER_PROCESS=2` (4 processes)
- 7B model: `GPUS_PER_PROCESS=1` (8 processes)

**Output**: Expert routing logs saved to `results/expert_scores/` (local) or `/fsx/outputs/esft/` (SLURM)

### Step 2: Generate Expert Configuration File

Generate `expert_config.json` based on routing statistics:

```bash
python scripts/expert/generate_expert_config.py \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --expert_scores_dir ./results/expert_scores/job_xxx \
    --output_path ./results/expert_configs/my_expert_config.json \
    --score_function token \
    --top_p 0.2 \
    --train_shared_experts \
    --train_non_expert_modules
```

**Parameters**:
| Parameter | Description |
|-----------|-------------|
| `--model_name_or_path` | HuggingFace model name or local path |
| `--expert_scores_dir` | Routing log directory from Step 1 |
| `--output_path` | Output path for expert_config.json |
| `--score_function` | `token` (count-based) or `gate` (weight-based) |
| `--top_p` | Cumulative score threshold (e.g., 0.2 = top 20%) |
| `--train_shared_experts` | Train shared experts (optional) |
| `--train_non_expert_modules` | Train attention, embedding, etc. (optional) |

**Output Format**:
```json
{
  "experts": {
    "1": [79, 51, 122, 70, 96],
    "2": [42, 78, 22]
  },
  "shared_experts": true,
  "non_expert_modules": false
}
```

### Step 3: Execute ESFT Training with ms-swift Megatron

#### Local Execution

```bash
# Edit configuration
vim scripts/ms_swift_megatron_esft_local.sh

# Main configuration:
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
EXPERT_CONFIG="./results/expert_configs/my_expert_config.json"
TRAIN_DATASETS="/path/to/train.jsonl"
EXPERT_PARALLEL=8

# Run
bash scripts/ms_swift_megatron_esft_local.sh
```

#### SLURM Cluster

```bash
# Edit configuration
vim scripts/ms_swift_megatron_esft_slurm.sh

# Submit job
sbatch scripts/ms_swift_megatron_esft_slurm.sh

# Monitor
squeue -u $USER
tail -f /fsx/logs/ms-swift/job_<JOB_ID>/job_<JOB_ID>_node0.log
```

#### Configuration Details

**Parallelism Strategy**

`EXPERT_PARALLEL × TENSOR_PARALLEL × PIPELINE_PARALLEL × DATA_PARALLEL = Total GPUs`

> ⚠️ Due to ms-swift's mcore_bridge limitations, `TENSOR_PARALLEL` and `CONTEXT_PARALLEL` are currently not supported.

**Batch Size Calculation**

`GLOBAL_BATCH_SIZE` must be divisible by `MICRO_BATCH_SIZE × data_parallel_size`

Where: `data_parallel_size = Total_GPUs / (EP × TP × PP × CP)`

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

```bash
python scripts/generate_trainable_params.py expert_config.json \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --pattern auto \
    --format regex \
    --megatron \
    --ep-size 8
```

### Adding New Model Support

See `model_patch/README.md` for instructions on adding expert routing log support for new MoE models.

## Citation

Original paper: [Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural Large Language Models](https://arxiv.org/abs/2407.01906)

## License

See [LICENSE-CODE](./LICENSE-CODE) for details.
