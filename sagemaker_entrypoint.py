#!/usr/bin/env python3
"""
SageMaker AI Entry Point for ESFT Expert Scores Collection and Training
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_hyperparameters():
    """Load hyperparameters from SageMaker"""
    hyperparameters_file = "/opt/ml/input/config/hyperparameters.json"
    
    # Default hyperparameters
    default_params = {
        "model": "",
        "train_dataset": "",
        "eval_dataset": "",
        "n_sample_tokens": "131072",
        "gpus_per_process": "8",
        "world_size": "1",
        "score_function": "token",
        "score_threshold": "0.2",
        "train_shared_experts": "false",
        "train_non_expert_modules": "false",
        "expert_config_dir": "",  # If provided, skip scoring and go straight to training
        "mode": "both",  # "scoring", "training", or "both"
        # Training specific parameters
        "train_epochs": "1",
        "micro_batch_size": "1",
        "global_batch_size": "256",
        "learning_rate": "7e-6",
        "warmup_ratio": "0.1",
        "weight_decay": "0.1",
        "min_learning_rate": "0.0",
        "max_length": "16384",
        "lr_decay_style": "cosine",
        "save_interval": "100",
        "use_wandb": "false",
        "expert_parallel": "",  # Will default to gpus_per_process if not set
        "pipeline_parallel": "1"
    }
    
    if os.path.exists(hyperparameters_file):
        with open(hyperparameters_file, 'r') as f:
            params = json.load(f)
        # Merge with defaults
        default_params.update(params)
        logger.info(f"Loaded hyperparameters: {params}")
    else:
        logger.info("Using default hyperparameters")
    
    return default_params

def setup_environment():
    """Setup environment variables"""
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:/opt/ml/code"
    
    # Change to code directory
    os.chdir("/opt/ml/code")

def run_expert_scores_collection(params):
    """Run the expert scores collection script"""
    
    # Create output directories
    job_id = f"sagemaker_{os.environ.get('TRAINING_JOB_NAME', 'local')}"
    output_dir = f"/opt/ml/output/data/expert_scores/job_{job_id}"
    log_dir = f"/opt/ml/output/data/logs/job_{job_id}"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info("========================================")
    logger.info("Expert Scores Collection - SageMaker AI")
    logger.info(f"Model: {params['model']}")
    logger.info(f"Dataset: {params['train_dataset']}")
    logger.info(f"Output: {output_dir}")
    logger.info("========================================")
    
    # Set environment variables for the script
    env = os.environ.copy()
    env.update({
        'SM_HP_MODEL': params['model'],
        'SM_HP_EVAL_DATASET': params['train_dataset'],  # Note that use variable name as SM_HP_EVAL_DATASET for scoring scripts
        'SM_HP_N_SAMPLE_TOKENS': str(params['n_sample_tokens']),
        'SM_HP_GPUS_PER_PROCESS': str(params['gpus_per_process']),
        'SM_HP_WORLD_SIZE': str(params['world_size']),
        'SM_OUTPUT_DATA_DIR': '/opt/ml/output/data',
        'TRAINING_JOB_NAME': os.environ.get('TRAINING_JOB_NAME', 'sagemaker_job'),
        "SM_HP_SCORE_FUNCTION": params['score_function'],
        "SM_HP_SCORE_THRESHOLD": str(params['score_threshold']),
        "SM_HP_IS_TRAIN_SHARED_EXPERTS": str(params['train_shared_experts']).lower(),
        "SM_HP_IS_TRAIN_NON_EXPERT_MODULES": str(params['train_non_expert_modules']).lower()
    })
    
    # Run the SageMaker script
    cmd = ["bash", "scripts/expert/run_get_exp_scores_sagemaker.sh"]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        with open(f"{log_dir}/run.step1.score_collection.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                log_file.flush()
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
        logger.info(f"Expert scores collection completed successfully!")
        logger.info(f"Output saved to: {output_dir}")
        logger.info(f"Logs saved to: {log_dir}")
        
        # Return the expert config path for training
        expert_config_path = f"/opt/ml/output/data/results/expert_configs.json"
        return expert_config_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Expert scores collection failed with return code {e.returncode}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None


def run_esft_training(params, expert_config_path):
    """Run the ESFT training script"""
    
    job_id = f"sagemaker_{os.environ.get('TRAINING_JOB_NAME', 'local')}"
    log_dir = f"/opt/ml/output/data/logs/job_{job_id}"
    
    os.makedirs(log_dir, exist_ok=True)
    
    logger.info("========================================")
    logger.info("ESFT Training - SageMaker AI")
    logger.info(f"Model: {params['model']}")
    logger.info(f"Train Dataset: {params['train_dataset']}")
    logger.info(f"Eval Dataset: {params['eval_dataset']}")
    logger.info(f"Expert Config: {expert_config_path}")
    logger.info("========================================")
    
    # Set environment variables for the training script
    env = os.environ.copy()
    env.update({
        'SM_HP_MODEL': params['model'],
        'SM_HP_TRAIN_DATASET': params['train_dataset'],
        'SM_HP_EVAL_DATASET': params['eval_dataset'],
        'SM_HP_EXPERT_CONFIG_DIR': expert_config_path,
        'SM_HP_GPUS_PER_PROCESS': str(params['gpus_per_process']),
        'SM_HP_TRAIN_EPOCHS': str(params['train_epochs']),
        'SM_HP_MICRO_BATCH_SIZE': str(params['micro_batch_size']),
        'SM_HP_GLOBAL_BATCH_SIZE': str(params['global_batch_size']),
        'SM_HP_LEARNING_RATE': str(params['learning_rate']),
        'SM_HP_WARMUP_RATIO': str(params['warmup_ratio']),
        'SM_HP_WEIGHT_DECAY': str(params['weight_decay']),
        'SM_HP_MIN_LEARNING_RATE': str(params['min_learning_rate']),
        'SM_HP_MAX_LENGTH': str(params['max_length']),
        'SM_HP_LR_DECAY_STYLE': params['lr_decay_style'],
        'SM_HP_SAVE_INTERVAL': str(params['save_interval']),
        'SM_HP_USE_WANDB': str(params['use_wandb']).lower(),
        'SM_HP_EXPERT_PARALLEL': str(params.get('expert_parallel', params['gpus_per_process'])),
        'SM_HP_PIPELINE_PARALLEL': str(params['pipeline_parallel']),
        'SM_MODEL_DIR': '/opt/ml/model',
        'SM_OUTPUT_DATA_DIR': '/opt/ml/output/data',
        'TRAINING_JOB_NAME': os.environ.get('TRAINING_JOB_NAME', 'sagemaker_job')
    })
    
    # Run the training script
    cmd = ["bash", "scripts/ms_swift_megatron_esft_sagemaker.sh"]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run the command
    try:
        with open(f"{log_dir}/run.step2.training.log", "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                log_file.write(line)
                log_file.flush()
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
                
        logger.info(f"ESFT training completed successfully!")
        logger.info(f"Model saved to: /opt/ml/model")
        logger.info(f"Logs saved to: {log_dir}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"ESFT training failed with return code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return False


def main():
    """Main entry point"""
    logger.info("Starting SageMaker AI ESFT Pipeline")
    
    try:
        # Setup environment
        setup_environment()
        
        # Get hyperparameters
        params = get_hyperparameters()
        
        # Determine execution mode
        mode = params.get('mode', 'both')
        expert_config_dir = params.get('expert_config_dir', '').strip()
        
        # If expert_config_dir is provided, skip scoring and go straight to training
        if expert_config_dir:
            logger.info(f"Expert config provided: {expert_config_dir}")
            logger.info("Skipping scoring phase, proceeding directly to training")
            mode = 'training'
            expert_config_path = expert_config_dir
        else:
            expert_config_path = None
        
        success = True
        
        # Execute based on mode
        if mode in ['scoring', 'both']:
            logger.info("=== Phase 1: Expert Scores Collection ===")
            expert_config_path = run_expert_scores_collection(params)
            if not expert_config_path:
                logger.error("Expert scores collection failed!")
                sys.exit(1)
            logger.info(f"Expert config generated: {expert_config_path}")
        
        if mode in ['training', 'both']:
            if not expert_config_path:
                logger.error("No expert config available for training!")
                sys.exit(1)
            
            logger.info("=== Phase 2: ESFT Training ===")
            success = run_esft_training(params, expert_config_path)
            if not success:
                logger.error("ESFT training failed!")
                sys.exit(1)
        
        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()