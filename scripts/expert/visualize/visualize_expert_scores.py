import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

    return gate_scores, token_scores, moe_layers


def visualize_expert_distribution(gate_scores, token_scores, moe_layers, n_experts, output_dir, score_type="gate", top_p=0.2):
    """
    Visualize expert score distribution.
    
    Args:
        gate_scores: numpy array of shape (n_layers, n_experts)
        token_scores: numpy array of shape (n_layers, n_experts)
        moe_layers: set of 1-indexed layer numbers that have MoE
        n_experts: number of experts
        output_dir: directory to save plots
        score_type: "gate" or "token"
        top_p: threshold for calculating number of experts needed (e.g., 0.2 for top 20%)
    """
    scores = gate_scores if score_type == "gate" else token_scores
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot average across all MoE layers (line plot)
    avg_scores = np.mean([scores[layer-1] for layer in moe_layers], axis=0)
    sorted_indices = np.argsort(avg_scores)[::-1]  # Sort by score descending
    sorted_scores = avg_scores[sorted_indices]
    
    # Well-balanced case (uniform distribution)
    well_balanced_score = 1.0 / n_experts
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_experts), sorted_scores, 'b-', linewidth=2, label='Actual Distribution')
    plt.axhline(y=well_balanced_score, color='r', linestyle='--', linewidth=2, label='Well-Balanced (Uniform)')
    plt.xlabel('Expert Rank (sorted by score)')
    plt.ylabel(f'Average {score_type.capitalize()} Score')
    plt.title(f'Average Expert {score_type.capitalize()} Score Distribution Across All MoE Layers')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'expert_{score_type}_distribution_average.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1-2. Plot cumulative distribution
    cumulative_scores = np.cumsum(sorted_scores)
    
    # Find number of experts needed to reach top_p threshold
    top_p_idx = np.argmax(cumulative_scores >= top_p)
    experts_for_top_p = top_p_idx + 1  # +1 because index is 0-based
    top_p_percentage = top_p * 100
    expert_percentage = (experts_for_top_p / n_experts) * 100
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_experts), cumulative_scores, 'b-', linewidth=2, label='Actual Cumulative')
    plt.plot(range(n_experts), np.linspace(well_balanced_score, 1.0, n_experts), 'r--', linewidth=2, label='Well-Balanced Cumulative')
    
    # Add horizontal line at top_p threshold
    plt.axhline(y=top_p, color='g', linestyle=':', linewidth=2, alpha=0.7, label=f'Top {top_p_percentage:.0f}% Threshold')
    
    # Add vertical line showing number of experts needed
    plt.axvline(x=top_p_idx, color='g', linestyle=':', linewidth=2, alpha=0.7)
    
    # Add annotation
    plt.annotate(f'{expert_percentage:.1f}% experts\n({experts_for_top_p}/{n_experts} experts)\nfor top {top_p_percentage:.0f}%', 
                xy=(top_p_idx, top_p), xytext=(top_p_idx + n_experts*0.1, top_p + 0.1),
                arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                fontsize=10, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    plt.xlabel('Expert Rank (sorted by score)')
    plt.ylabel(f'Cumulative {score_type.capitalize()} Score')
    plt.title(f'Cumulative Expert {score_type.capitalize()} Score Distribution\n({expert_percentage:.1f}% experts ({experts_for_top_p}/{n_experts}) needed for top {top_p_percentage:.0f}%)')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'expert_{score_type}_distribution_cumulative.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot individual layers (up to 16 layers per figure)
    moe_layers_list = sorted(list(moe_layers))
    layers_per_fig = 16
    well_balanced_score = 1.0 / n_experts
    
    for fig_idx in range(0, len(moe_layers_list), layers_per_fig):
        layers_subset = moe_layers_list[fig_idx:fig_idx + layers_per_fig]
        n_rows = int(np.ceil(len(layers_subset) / 4))
        n_cols = min(4, len(layers_subset))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if len(layers_subset) == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, layer in enumerate(layers_subset):
            layer_scores = scores[layer-1]  # Convert to 0-indexed
            sorted_indices = np.argsort(layer_scores)[::-1]
            sorted_scores = layer_scores[sorted_indices]
            
            axes[i].plot(range(n_experts), sorted_scores, 'b-', linewidth=2, label='Actual')
            axes[i].axhline(y=well_balanced_score, color='r', linestyle='--', linewidth=1, alpha=0.7, label='Well-Balanced')
            axes[i].set_title(f'Layer {layer}')
            axes[i].set_xlabel('Expert Rank')
            axes[i].set_ylabel(f'{score_type.capitalize()} Score')
            axes[i].grid(True, alpha=0.3)
            if i == 0:  # Add legend only to first subplot
                axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(layers_subset), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Expert {score_type.capitalize()} Score Distribution by Layer (Layers {layers_subset[0]}-{layers_subset[-1]})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'expert_{score_type}_distribution_layers_{layers_subset[0]}_{layers_subset[-1]}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Top-k expert usage across layers (only for gate scores)
    if score_type == "gate":
        top_k_values = [5, 10, 20, 50]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, k in enumerate(top_k_values):
            if k > n_experts:
                continue
                
            layer_top_k_usage = []
            for layer in moe_layers_list:
                layer_scores = scores[layer-1]
                top_k_indices = np.argsort(layer_scores)[-k:]
                top_k_usage = np.sum(layer_scores[top_k_indices])
                layer_top_k_usage.append(top_k_usage)
            
            axes[i].plot(moe_layers_list, layer_top_k_usage, 'o-', linewidth=2, markersize=4)
            axes[i].set_xlabel('Layer')
            axes[i].set_ylabel(f'Top-{k} {score_type.capitalize()} Score Sum')
            axes[i].set_title(f'Top-{k} Expert Usage Across Layers')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_ylim(0, 1)
        
        plt.suptitle(f'Top-K Expert Usage Analysis ({score_type.capitalize()} Scores)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'expert_{score_type}_topk_usage.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {score_type} score visualizations to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize expert score distributions")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--expert_scores_dir", type=str, required=True, help="Directory containing expert scores")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for visualizations")
    parser.add_argument("--score_types", type=str, nargs="+", default=["gate", "token"], 
                       choices=["gate", "token"], help="Types of scores to visualize")
    parser.add_argument("--top_p", type=float, default=0.2)

    args = parser.parse_args()

    # Get MoE config from model
    n_layers, n_experts, top_k = get_moe_config(args.model_name_or_path)

    # Walk inside expert_scores_dir and get file names
    file_names = []
    for rank in [i for i in os.listdir(args.expert_scores_dir) if 'rank' in i]:
        for file in os.listdir(os.path.join(args.expert_scores_dir, rank)):
            file_names.append([rank, file])

    # Load expert scores
    gate_scores, token_scores, moe_layers = get_summary(args.expert_scores_dir, file_names, top_k, n_experts, n_layers)

    # Generate visualizations for each score type
    for score_type in args.score_types:
        print(f"\nGenerating {score_type} score visualizations...")
        visualize_expert_distribution(gate_scores, token_scores, moe_layers, n_experts, args.output_dir, score_type, args.top_p)

    print(f"\nAll visualizations saved to {args.output_dir}")