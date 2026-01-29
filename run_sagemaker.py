"""
ESFT Run Script with SageMaker SDK v3
"""

import os
import boto3
import argparse
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    SourceCode, 
    InputData, 
    Compute,
    TensorBoardOutputConfig,
)
from sagemaker.core.helper.session_helper import get_execution_role
import time
from utils import s3_upload


def run_esft(args):
    # AWS setting
    sts = boto3.client('sts')
    account_id = sts.get_caller_identity()['Account']
    region = boto3.Session().region_name or 'us-east-1'
    
    try:
        role = get_execution_role()
    except:
        role = f"arn:aws:iam::{account_id}:role/AmazonSageMaker-ExecutionRole"
    
    image_uri = args.sagemaker_image
    s3_bucket = f"sagemaker-{region}-{account_id}"  # Default SageMaker bucket
    
    print(f"Account ID: {account_id}")
    print(f"Region: {region}")
    print(f"Image URI: {image_uri}")
    print(f"S3 Bucket: {s3_bucket}")
    print(f"Role: {role}")

    # Training dataset (Must be uploaded at S3 bucket)
    s3_upload(args.train_dataset, f"s3://{s3_bucket}/input/data/train/")
    s3_upload(args.eval_dataset, f"s3://{s3_bucket}/input/data/eval/")
    train_dataset = f"/opt/ml/input/data/train/{os.path.basename(args.train_dataset)}"
    eval_dataset = f"/opt/ml/input/data/eval/{os.path.basename(args.eval_dataset)}"

    # Expert config (If exist, and MUST be uploaded at S3 bucket) 
    expert_config = ""
    if args.expert_config_dir:
        s3_upload(args.expert_config_dir, f"s3://{s3_bucket}/input/data/expert_config/")
        expert_config = f"/opt/ml/input/data/expert_config/{os.path.basename(args.expert_config_dir)}" 

    # Hyperparameters setting
    hyperparameters = {
        "model": args.model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "n_sample_tokens": str(args.n_sample_tokens),
        "gpus_per_process": str(args.gpus_per_process),
        "world_size": str(args.world_size),
        "score_function": args.score_function,
        "score_threshold": str(args.score_threshold),
        "train_shared_experts": str(args.train_shared_experts).lower(),
        "train_non_expert_modules": str(args.train_non_expert_modules).lower(),
        "expert_config_dir": expert_config,
        "mode": args.mode,
        # Training parameters
        "train_epochs": str(args.train_epochs),
        "micro_batch_size": str(args.micro_batch_size),
        "global_batch_size": str(args.global_batch_size),
        "learning_rate": str(args.learning_rate),
        "warmup_ratio": str(args.warmup_ratio),
        "weight_decay": str(args.weight_decay),
        "min_learning_rate": str(args.min_learning_rate),
        "max_length": str(args.max_length),
        "lr_decay_style": args.lr_decay_style,
        "save_interval": str(args.save_interval),
        "use_wandb": str(args.use_wandb).lower(),
        "expert_parallel": str(args.expert_parallel or args.gpus_per_process),
        "pipeline_parallel": str(args.pipeline_parallel),
    }
    
    # SageMaker setting
    source_code = SourceCode(
        command="python /opt/ml/code/sagemaker_entrypoint.py",  # already uploaded in the docker img
    )
    
    compute = Compute(
        instance_count=args.instance_count,
        instance_type=args.instance_type,
        volume_size_in_gb=args.volume_size,
    )

    tb_config = TensorBoardOutputConfig(
        s3_output_path=f"s3://{s3_bucket}/output/tensorboard",
        local_path="/opt/ml/output/tensorboard"
    )

    input_data_config = [
        InputData(
            data_source=f"s3://{s3_bucket}/input/data/train/",
            channel_name="train",
        ),
        InputData(
            data_source=f"s3://{s3_bucket}/input/data/eval/",
            channel_name="eval",
        )
    ]
    if expert_config:
        input_data_config.append(
            InputData(
                data_source=f"s3://{s3_bucket}/input/data/expert_config/",
                channel_name="expert_config",
            )
        )

    model_trainer = ModelTrainer(
        training_image=image_uri,
        source_code=source_code,
        compute=compute,
        hyperparameters=hyperparameters,
        role=role,
        base_job_name="esft",
        environment={
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    model_trainer.with_tensorboard_output_config(tb_config)
    
    # Job Info
    print("="*50)
    print("SageMaker Training Start")
    print(f"Instance Type: {args.instance_type}")
    print(f"Instance Count: {args.instance_count}")
    print(f"Hyperparameters: {hyperparameters}")
    print(f"Maximum Run: {args.max_run_hours} hours")
    print(f"Output Location Details: {model_trainer.output_data_config}")
    print("="*50)
    
    model_trainer.train(
        input_data_config=input_data_config,
        wait=True,   # Wait for the training job to complete
        logs=True    # Display the training container logs
    )
    
    print("="*50)
    print("Job Submitted!")
    print("="*50)
    
    return model_trainer

def main():
    parser = argparse.ArgumentParser(description="SageMaker ESFT Expert Scores Collection and Training (SDK v3)")

    # ESFT mode
    parser.add_argument("--mode", type=str, default="both", choices=["scoring", "training", "both"],
                        help="Execution mode: scoring only, training only, or both")
    
    # Model and dataset parameters
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                       help="Model name or path")
    parser.add_argument("--train_dataset", type=str, default="datasets/train/intent.jsonl",
                       help="Training dataset path")
    parser.add_argument("--eval_dataset", type=str, default="datasets/eval/intent.jsonl",
                       help="Evaluation dataset path")
    parser.add_argument("--n_sample_tokens", type=int, default=131072,
                       help="Number of sample tokens (-1 for entire dataset)")
    parser.add_argument("--gpus_per_process", type=int, default=8,
                       help="Number of GPUs per process")
    parser.add_argument("--world_size", type=int, default=1,
                       help="Number of processes")

    # ESFT config parameters
    parser.add_argument("--score_function", type=str, default="token", choices=["token", "gate"], 
                        help="ESFT score function")
    parser.add_argument("--score_threshold", type=float, default=0.2,
                        help="ESFT top_p threshold")
    parser.add_argument("--train_shared_experts", action="store_true",
                        help="training shared experts")
    parser.add_argument("--train_non_expert_modules", action="store_true",
                        help="training non-expert modules")
    parser.add_argument("--expert_config_dir", type=str, default="",
                        help="Path to existing expert config (skip scoring if provided)")
    
    # Training parameters
    parser.add_argument("--train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--micro_batch_size", type=int, default=1,
                        help="Micro batch size")
    parser.add_argument("--global_batch_size", type=int, default=256,
                        help="Global batch size")
    parser.add_argument("--learning_rate", type=float, default=7e-6,
                        help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Warmup ratio")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--min_learning_rate", type=float, default=0.0,
                        help="Minimum learning rate")
    parser.add_argument("--max_length", type=int, default=16384,
                        help="Maximum sequence length")
    parser.add_argument("--lr_decay_style", type=str, default="cosine",
                        help="Learning rate decay style")
    parser.add_argument("--save_interval", type=int, default=100,
                        help="Save interval")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--expert_parallel", type=int, default=None,
                        help="Expert parallel size (defaults to gpus_per_process)")
    parser.add_argument("--pipeline_parallel", type=int, default=1,
                        help="Pipeline parallel size")
    
    # SageMaker instance parameters
    parser.add_argument("--sagemaker_image", type=str, required=True,
                        help="SageMaker docker image")
    parser.add_argument("--instance_type", type=str, default="ml.p4d.24xlarge",
                       help="SageMaker instance type")
    parser.add_argument("--instance_count", type=int, default=1,
                       help="Number of instances")
    parser.add_argument("--max_run_hours", type=int, default=1,
                       help="Maximum run time in hours")
    parser.add_argument("--volume_size", type=int, default=100,
                       help="EBS volume size in GB")
    
    args = parser.parse_args()
    
    # Validate instance type with warning
    recommended_instances = ["ml.p3.16xlarge", "ml.p4d.24xlarge", "ml.p4de.24xlarge", "ml.g4dn.12xlarge", "ml.g5.12xlarge"]
    if args.instance_type not in recommended_instances:
        print(f"WARNING: '{args.instance_type}' is not in the recommended instance types.")
        print(f"Recommended types: {', '.join(recommended_instances)}")
        print("Proceeding with the specified instance type...")
    
    # 실행
    model_trainer = run_esft(args)

if __name__ == "__main__":
    main()