import argparse
import json
import os
import numpy as np
from transformers import AutoConfig


def parse_line(line):
    expert_ids, expert_weights = line.split("\t\t")
    expert_ids = [int(i) for i in expert_ids.split("\t")]
    expert_weights = [float(i) for i in expert_weights.split("\t")]
    return expert_ids, expert_weights


def get_moe_config(model_name_or_path):
    """Extract MoE config from HuggingFace model config."""
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    # Number of layers
    n_layers = getattr(config, "num_hidden_layers", None)
    if n_layers is None:
        raise ValueError(f"Cannot find num_hidden_layers in config for {model_name_or_path}")
    
    # Number of experts - different models use different attribute names
    n_experts = (
        getattr(config, "n_routed_experts", None) or  # GLM-4, DeepSeek
        getattr(config, "num_experts", None) or       # Qwen, Mixtral
        getattr(config, "num_local_experts", None)    # Some other models
    )
    if n_experts is None:
        raise ValueError(f"Cannot find number of experts in config for {model_name_or_path}")
    
    # Top-k experts per token - different models use different attribute names
    top_k = (
        getattr(config, "num_experts_per_tok", None) or  # GLM-4, DeepSeek
        getattr(config, "num_selected_experts", None) or # Qwen
        getattr(config, "num_experts_per_token", None)   # Some other models
    )
    if top_k is None:
        raise ValueError(f"Cannot find top_k (num_experts_per_tok) in config for {model_name_or_path}")
    
    print(f"Model: {model_name_or_path}")
    print(f"  n_layers: {n_layers}, n_experts: {n_experts}, top_k: {top_k}")
    
    return n_layers, n_experts, top_k


def get_summary(expert_scores_dir, files, top_k, n_experts, n_layers):
    """
    Load expert routing logs and compute normalized scores.
    
    Returns:
        summary: dict with token_scores and gate_scores (1-indexed layer keys)
        moe_layers: set of layer indices (1-indexed) that have MoE data
    """
    gate_scores = np.zeros((n_layers, n_experts))
    token_scores = np.zeros((n_layers, n_experts))
    moe_layers = set()  # Track which layers have MoE data

    print("Loading files...")
    for rank, file in files:
        # File format: expert_weights_{layer_id}.txt (1-indexed)
        layer_id_1indexed = int(file.split(".")[0].split("_")[2])
        layer_id = layer_id_1indexed - 1  # Convert to 0-indexed for array
        moe_layers.add(layer_id_1indexed)  # Track as 1-indexed

        with open(os.path.join(expert_scores_dir, rank, file)) as f:
            data = f.readlines()
            for line in data:
                expert_ids, expert_weights = parse_line(line)
                np.add.at(gate_scores[layer_id], expert_ids, expert_weights)
                np.add.at(token_scores[layer_id], expert_ids, np.ones_like(expert_weights) / top_k)
    
    # Calculate total from MoE layers only
    total = 0
    for layer_1idx in sorted(moe_layers):
        layer_total = sum(token_scores[layer_1idx - 1])
        if layer_total > 0:
            total = layer_total
            break
    
    if total == 0:
        raise ValueError("No expert scores found in any layer. Check if log files exist and are properly formatted.")
    
    print(f"Found {len(moe_layers)} MoE layers: {sorted(moe_layers)[:5]}...{sorted(moe_layers)[-5:]}" 
          if len(moe_layers) > 10 else f"Found {len(moe_layers)} MoE layers: {sorted(moe_layers)}")
    
    gate_scores = gate_scores / total
    token_scores = token_scores / total

    # Convert to 1-indexed dict format
    summary = {"token_scores": token_scores, "gate_scores": gate_scores}
    summary = {k: {str(i+1): {str(j): round(v, 4) for j, v in enumerate(l)} for i, l in enumerate(v)} for k, v in summary.items()}

    return summary, moe_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--expert_scores_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--score_function", type=str, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--train_shared_experts", action="store_true")
    parser.add_argument("--train_non_expert_modules", action="store_true")

    args = parser.parse_args()

    # Get MoE config from model
    n_layers, n_experts, top_k = get_moe_config(args.model_name_or_path)

    expert_cfg = {
        "experts": {},
        "shared_experts": args.train_shared_experts,
        "non_expert_modules": args.train_non_expert_modules
    }

    # Walk inside expert_scores_dir and get file names
    file_names = []
    for rank in [i for i in os.listdir(args.expert_scores_dir) if 'rank' in i]:
        for file in os.listdir(os.path.join(args.expert_scores_dir, rank)):
            file_names.append([rank, file])

    summary_file = os.path.join(args.expert_scores_dir, "summary.json")
    summary, moe_layers = get_summary(args.expert_scores_dir, file_names, top_k, n_experts, n_layers)

    with open(summary_file, "w") as f:
        f.write(json.dumps(summary))

    # Only process MoE layers (skip dense layers)
    scores = summary[f"{args.score_function}_scores"]
    for layer in sorted(moe_layers):
        layer_str = str(layer)
        l_score = [(int(k), v) for k, v in scores[layer_str].items()]
        l_score = sorted(l_score, key=lambda x: x[1], reverse=True)
        
        selected_experts = []
        current_score = 0
        for expert, score in l_score:
            if current_score >= args.top_p:
                break
            selected_experts.append(expert)
            current_score += score
        expert_cfg["experts"][layer_str] = selected_experts

    print(f"Generated config for {len(expert_cfg['experts'])} MoE layers")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(expert_cfg, f)
    
    print(f"Saved to {args.output_path}")
