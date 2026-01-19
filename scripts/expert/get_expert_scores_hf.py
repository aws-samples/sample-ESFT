import json
import os
import sys
import torch
import argparse
import random

# Add project root to Python path to import project modules
project_root = os.path.join(os.path.dirname(__file__), '../..')
sys.path.insert(0, project_root)

import model_patch  # Apply MoE logging patches before loading model

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_formatted_input_and_target
import torch.multiprocessing as mp
from tqdm import tqdm


def load_model_on_gpus(args, rank, gpu_ids):
    """Load model and distribute across specified GPUs"""
    print(f"Process {rank}: Loading model on GPUs {gpu_ids}...", flush=True)
    
    # Set visible devices for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    
    # Load model with device_map="auto" to distribute across visible GPUs
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto"  # Auto distribute across visible GPUs
    )
    
    # Enable expert logging
    model.config.log_expert_weights = True
    model.config.expert_log_dir = os.path.join(args.output_dir, f"rank_{rank}")
    os.makedirs(model.config.expert_log_dir, exist_ok=True)
    
    model.eval()
    print(f"Process {rank}: Model loaded successfully on GPUs {gpu_ids}", flush=True)
    
    # Get device for input tensors
    device = next(model.parameters()).device
    
    return model, tokenizer, device


def eval_expert(rank, args, dataset, gpu_ids):
    """Evaluate expert weights on a subset of data"""
    try:
        model, tokenizer, device = load_model_on_gpus(args, rank, gpu_ids)
        
        # Shard dataset by rank
        cur_dataset = dataset[rank::args.world_size]
        
        # Check if we should process entire dataset or limit by tokens
        process_entire_dataset = args.n_sample_tokens == -1
        
        if process_entire_dataset:
            print(f"Process {rank}: Processing entire dataset ({len(cur_dataset)} samples)", flush=True)
            pbar = tqdm(
                total=len(cur_dataset),
                desc=f"Process {rank} (GPUs {gpu_ids})",
                position=rank,
                unit="samples"
            )
        else:
            # Calculate tokens per process
            n_sample_tokens = args.n_sample_tokens // args.world_size
            print(f"Process {rank}: Processing {len(cur_dataset)} samples, target {n_sample_tokens} tokens", flush=True)
            pbar = tqdm(
                total=n_sample_tokens,
                desc=f"Process {rank} (GPUs {gpu_ids})",
                position=rank,
                unit="tokens"
            )
        
        done_tokens = 0
        processed_samples = 0
        skipped_samples = 0
        
        with torch.no_grad():
            for idx, instance in enumerate(cur_dataset):
                # Check if we should stop (only when not processing entire dataset)
                if not process_entire_dataset and done_tokens >= n_sample_tokens:
                    print(f"Process {rank}: Reached target tokens, stopping at sample {idx}", flush=True)
                    break
                
                input_ids, target_ids = get_formatted_input_and_target(
                    instance['messages'],
                    tokenizer,
                    -100
                )
                
                # Skip if input_ids or target_ids is None
                if input_ids is None or target_ids is None:
                    skipped_samples += 1
                    continue
                
                # Process sample
                input_tensor = torch.tensor(input_ids).unsqueeze(0).to(device)
                target_tensor = torch.tensor(target_ids).unsqueeze(0).to(device)
                
                model(input_ids=input_tensor, labels=target_tensor)
                
                done_tokens += len(input_ids)
                processed_samples += 1
                
                # Update progress bar
                if process_entire_dataset:
                    pbar.update(1)  # Update by samples
                else:
                    pbar.update(len(input_ids))  # Update by tokens
        
        pbar.close()
        print(f"Process {rank} completed: {done_tokens} tokens, {processed_samples} samples processed, {skipped_samples} skipped", flush=True)

    except Exception as e:
        print(f"Error in process {rank}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate expert weights with configurable GPU allocation")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to the evaluation dataset (jsonl)")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the evaluation results")
    parser.add_argument("--n_sample_tokens", type=int, required=True, help="Total tokens to sample for expert evaluation (-1 to process entire dataset)")
    parser.add_argument("--gpus_per_process", type=int, default=8, help="Number of GPUs per process (1, 2, 4, or 8)")
    parser.add_argument("--world_size", type=int, default=None, help="Number of processes (auto-calculated if not specified)")
    
    args = parser.parse_args()
    
    # Auto-calculate world_size based on total GPUs and gpus_per_process
    total_gpus = torch.cuda.device_count()
    if args.world_size is None:
        args.world_size = total_gpus // args.gpus_per_process
    
    if args.world_size * args.gpus_per_process > total_gpus:
        raise ValueError(f"Not enough GPUs: need {args.world_size * args.gpus_per_process}, have {total_gpus}")
    
    print("=" * 60)
    print(f"Expert Scores Evaluation - Multi-Process Mode")
    print("=" * 60)
    print(f"Total GPUs: {total_gpus}")
    print(f"GPUs per process: {args.gpus_per_process}")
    print(f"Number of processes: {args.world_size}")
    print(f"Model: {args.base_model_path}")
    print(f"Dataset: {args.eval_dataset}")
    print(f"Output: {args.output_dir}")
    if args.n_sample_tokens == -1:
        print(f"Mode: Process entire dataset")
    else:
        print(f"Target tokens: {args.n_sample_tokens}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(5934875)
    
    # Load and shuffle dataset
    print(f"Loading dataset from {args.eval_dataset}...")
    dataset = [json.loads(line) for line in open(args.eval_dataset).readlines()]
    random.shuffle(dataset)
    print(f"Loaded {len(dataset)} samples")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Assign GPU IDs to each process
    gpu_assignments = []
    for rank in range(args.world_size):
        start_gpu = rank * args.gpus_per_process
        end_gpu = start_gpu + args.gpus_per_process
        gpu_ids = list(range(start_gpu, end_gpu))
        gpu_assignments.append(gpu_ids)
        print(f"Process {rank} will use GPUs: {gpu_ids}")
    
    print("\nStarting evaluation...")
    
    # Spawn processes
    mp.set_start_method('spawn', force=True)
    processes = []
    for rank in range(args.world_size):
        p = mp.Process(target=eval_expert, args=(rank, args, dataset, gpu_assignments[rank]))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    print("\nAll processes completed!")
