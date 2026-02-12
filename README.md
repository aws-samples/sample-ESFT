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
│   │   ├── get_expert_scores_hf.py         # Collect expert routing statistics
│   │   ├── generate_expert_config.py       # Generate expert config file
│   │   ├── run_get_exp_scores_local.sh     # Local execution script
│   │   ├── run_get_exp_scores_sagemaker.sh # SageMaker execution script
│   │   └── run_get_exp_scores_slurm.sh     # SLURM submission script
│   ├── generate_trainable_params.py        # Generate trainable parameter list
│   ├── ms_swift_megatron_esft_local.sh     # Local training script
│   ├── ms_swift_megatron_esft_sagemaker.sh # SageMaker training script
│   └── ms_swift_megatron_esft_slurm.sh     # SLURM training script
├── results/                        # Output directory
│   ├── expert_configs/             # Generated expert config files
│   └── expert_scores/              # Expert routing statistics
├── run_sagemaker.py                # Training ESFT by SageMaker script
├── sagemaker_entrypoint.py         # SageMaker entrypoint script
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

## Quick Start (AWS SageMaker AI)

### Environment Setup

For running on AWS SageMaker AI, you can use the pre-built container or build your own.

**Provided Docker container**

*Will be updated soon (currently uploaded at private ECR)*

**Build and Push your own Docker Image to ECR**:
```bash
# Build the Docker image
ACCOUNT_ID=...
REGION=...
DOCKER_IMAGE_NAME=...

DOCKER_DOMAIN=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

sudo docker build -t $DOCKER_IMAGE_NAME .
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $DOCKER_DOMAIN
sudo docker tag ${DOCKER_IMAGE_NAME}:latest $DOCKER_DOMAIN/${DOCKER_IMAGE_NAME}:latest
sudo docker push $DOCKER_DOMAIN/${DOCKER_IMAGE_NAME}:latest
```

Install sagemaker

```bash
pip install sagemaker boto3
```

### Run ESFT by SageMaker AI

#### Basic Usage

```bash
# Recommended GPU setting
INSTANCE=ml.p5.48xlarge
NUM_GPU=8

# Pre-built SageMaker container
SM_IMAGE_URI=...

# Complete ESFT pipeline (scoring + training)
python run_sagemaker.py \
    --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" \
    --train_dataset "datasets/train/intent.jsonl" \
    --eval_dataset "datasets/eval/intent.jsonl" \
    --n_sample_tokens 8096 \
    --gpus_per_process "${NUM_GPU}" \
    --world_size 1 \
    --score_function token \
    --score_threshold 0.2 \
    --train_epochs 3 \
    --learning_rate 7e-6 \
    --expert_parallel "${NUM_GPU}" \
    --sagemaker_image "${SM_IMAGE_URI}" \
    --instance_type "${INSTANCE}" \
    --max_run_hours 8
```

#### Advanced Configuration Options

**Execution Modes**:
```bash
# Only collect expert routing statistics
python run_sagemaker.py --mode scoring [other options...]

# Only run training (requires existing expert config)
python run_sagemaker.py --mode training --expert_config_dir "path/to/expert_config.json" [other options...]

# Full pipeline (default)
python run_sagemaker.py --mode both [other options...]
```

**Instance Type Recommendations**:
| Model Size | Recommended Instance | GPUs | Memory |
|------------|---------------------|------|--------|
| 7B-30B | `ml.p4d.24xlarge` | 8x A100 | 1.1TB |
| 30B-70B | `ml.p5.48xlarge` | 8x H100 | 2TB |
| 70B+ | `ml.p5.48xlarge` | 8x H100 | 2TB |

**Training Parameters**:
```bash
python run_sagemaker.py \
    --micro_batch_size 1 \
    --global_batch_size 256 \
    --learning_rate 7e-6 \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --max_length 16384 \
    --save_interval 100 \
    --use_wandb  # Enable W&B logging
```

### Monitoring and Results

#### 1. Job Status Monitoring

**AWS Console**:
- Navigate to SageMaker → Training Jobs
- Find your job (named `esft-YYYY-MM-DD-HH-MM-SS-XXX`)
- Monitor status, metrics, and logs in real-time

**CLI Monitoring**:
```bash
# List recent training jobs
aws sagemaker list-training-jobs --sort-by CreationTime --sort-order Descending --max-items 1

# Get specific job details
aws sagemaker describe-training-job --training-job-name <JOB_NAME>

# Stream CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow --log-stream-name <JOB_NAME>/algo-1-<TIMESTAMP>
```

#### 2. Output Locations

**S3 Output Structure**:
```
s3://sagemaker-{region}-{account-id}/
├── esft-{job-id}/
│   ├── output/
│   │   ├── model.tar.gz              # Final trained model
│   │   └── data/
│   │       ├── expert_scores/        # Expert routing statistics
│   │       │   └── job_{job_id}/
│   │       │       ├── expert_scores_rank_*.json
│   │       │       └── routing_logs/
│   │       ├── results/
│   │       │   └── expert_configs.json  # Generated expert configuration
│   │       └── logs/
│   │           └── job_{job_id}/
│   │               ├── run.step1.score_collection.log
│   │               ├── run.step2.training.log
│   │               └── training.log
│   └── tensorboard/                  # TensorBoard logs
│       └── events.out.tfevents.*
```

**Downloading Results**:
```bash
# Download trained model
aws s3 cp s3://sagemaker-{region}-{account-id}/esft-{job-id}/output/model.tar.gz ./

# Extract model
tar -xzf model.tar.gz

# Download expert configuration
aws s3 cp s3://sagemaker-{region}-{account-id}/esft-{job-id}/output/data/results/expert_configs.json ./

# Download logs
aws s3 sync s3://sagemaker-{region}-{account-id}/esft-{job-id}/output/data/logs/ ./logs/
```

#### 3. TensorBoard Monitoring

**Access TensorBoard**:
```bash
# Install TensorBoard
pip install tensorboard

# Download TensorBoard logs
aws s3 sync s3://sagemaker-{region}-{account-id}/output/tensorboard/ ./tensorboard_logs/

# Launch TensorBoard
tensorboard --logdir ./tensorboard_logs --port 6006
```

## Local Start

### Environment Setup

If you use vanilla EC2 or local machine, follow the steps below to set up the CUDA environment.

```bash
# check nvidia GPUs
lspci | grep -i nvidia

# install gcc
sudo apt-get update
sudo apt install wget gcc linux-headers-generic software-properties-common curl alsa-utils git -y
gcc --version

# Ubuntu-driver
sudo apt install ubuntu-drivers-common -y

# Check recommended nvidia-driver version
ubuntu-drivers devices | grep recommended

# Nvidia driver repository setup
distribution=$(. /etc/os-release;echo $ID$VERSION_ID | sed -e 's/\.//g')
wget https://developer.download.nvidia.com/compute/cuda/repos/$distribution/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
cat /etc/apt/sources.list.d/cuda-$distribution-x86_64.list

# Install Nvidia driver
driver_ver=$(ubuntu-drivers devices | grep recommended | awk '{print $3}' | cut -d'-' -f3-3)
sudo apt install nvidia-driver-$driver_ver-server nvidia-fabricmanager-$driver_ver -y

# NVCC install
# sudo apt install nvidia-cuda-toolkit
sudo apt install -y cuda-toolkit-12-8 cudnn-cuda-12 libnccl-dev
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
source ~/.bashrc
```

If you don't use Python 3.11, please install Python 3.11 first.

```bash
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
```

**(REQUIRED)** Install repositories and install dependencies

```bash
# Clone with submodule
git clone --recurse-submodules <REPO_URL>
cd ESFT-ms-swift

# Initialize and update submodule
git submodule update --init --recursive

# Apply patches to ms-swift
./scripts/apply_patches.sh

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install ninja cmake
pip install torch==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.26.4 psutil==7.1.1
pip install --no-build-isolation flash-attn==2.8.1
pip install --no-build-isolation "transformer-engine[pytorch]==2.11.0"
pip install -r requirements.txt
pip install -e ./ms-swift

# Install Apex for gradient_accumulation_fusion
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

> **Note**: The ms-swift submodule includes patches for SequentialMLP support with pipeline parallelism. See `patches/README.md` for details.

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

This project is licensed under the Apache-2.0 License.
