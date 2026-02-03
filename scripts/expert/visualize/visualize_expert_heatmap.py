import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from transformers import AutoConfig


def parse_line(line):
    """Parse expert routing log line."""
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


def load_expert_scores_for_task(expert_scores_dir, model_name, task_name, n_layers, n_experts, top_k):
    """
    Load expert routing logs for a specific task and return gate scores.
    
    Returns:
        gate_scores: numpy array of shape (n_layers, n_experts)
        moe_layers: set of layer indices (1-indexed) that have MoE data
    """
    task_dir = os.path.join(expert_scores_dir, task_name, model_name)
    
    if not os.path.exists(task_dir):
        raise ValueError(f"Task directory not found: {task_dir}")
    
    gate_scores = np.zeros((n_layers, n_experts))
    token_scores = np.zeros((n_layers, n_experts))
    moe_layers = set()
    
    # Find all rank directories
    rank_dirs = [d for d in os.listdir(task_dir) if d.startswith('rank_')]
    
    if not rank_dirs:
        raise ValueError(f"No rank directories found in {task_dir}")
    
    print(f"Loading task '{task_name}' for model '{model_name}'...")
    
    for rank_dir in rank_dirs:
        rank_path = os.path.join(task_dir, rank_dir)
        
        # Get all expert weight files
        files = [f for f in os.listdir(rank_path) if f.startswith('expert_weights_') and f.endswith('.txt')]
        
        for file in files:
            # File format: expert_weights_{layer_id}.txt (1-indexed)
            layer_id_1indexed = int(file.split(".")[0].split("_")[2])
            layer_id = layer_id_1indexed - 1  # Convert to 0-indexed for array
            moe_layers.add(layer_id_1indexed)  # Track as 1-indexed
            
            with open(os.path.join(rank_path, file)) as f:
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
        raise ValueError(f"No expert scores found for task {task_name}. Check if log files exist and are properly formatted.")
    
    gate_scores = gate_scores / total
    
    print(f"  Found {len(moe_layers)} MoE layers: {sorted(moe_layers)[:3]}...{sorted(moe_layers)[-3:]}" 
          if len(moe_layers) > 6 else f"  Found {len(moe_layers)} MoE layers: {sorted(moe_layers)}")
    
    return gate_scores, moe_layers


def get_top_k_experts_per_layer(gate_scores, moe_layers, top_k=6):
    """
    Get top-k experts for each MoE layer.
    
    Returns:
        dict: {layer_id (1-indexed): set of top-k expert indices}
    """
    top_experts_per_layer = {}
    
    for layer_1idx in moe_layers:
        layer_scores = gate_scores[layer_1idx - 1]  # Convert to 0-indexed
        top_k_indices = np.argsort(layer_scores)[-top_k:]  # Get top-k expert indices
        top_experts_per_layer[layer_1idx] = set(top_k_indices)
    
    return top_experts_per_layer


def calculate_overlap_matrix(task_top_experts, tasks, moe_layers, top_k=6):
    """
    Calculate overlap matrix between tasks.
    
    Returns:
        overlap_matrix: numpy array of shape (n_tasks, n_tasks, n_layers)
        layer_avg_overlap: numpy array of shape (n_tasks, n_tasks) - average across layers
    """
    n_tasks = len(tasks)
    n_layers = len(moe_layers)
    moe_layers_list = sorted(list(moe_layers))
    
    # Initialize matrices
    overlap_matrix = np.zeros((n_tasks, n_tasks, n_layers))
    layer_avg_overlap = np.zeros((n_tasks, n_tasks))
    
    for i, task1 in enumerate(tasks):
        for j, task2 in enumerate(tasks):
            layer_overlaps = []
            
            for k, layer_1idx in enumerate(moe_layers_list):
                if layer_1idx in task_top_experts[task1] and layer_1idx in task_top_experts[task2]:
                    experts1 = task_top_experts[task1][layer_1idx]
                    experts2 = task_top_experts[task2][layer_1idx]
                    
                    # Calculate overlap (intersection over union would be another option)
                    overlap = len(experts1.intersection(experts2))
                    overlap_ratio = overlap  # Keep as absolute number (0-6 for top_k=6)
                    
                    overlap_matrix[i, j, k] = overlap_ratio
                    layer_overlaps.append(overlap_ratio)
            
            # Average overlap across all layers
            if layer_overlaps:
                layer_avg_overlap[i, j] = np.mean(layer_overlaps)
    
    return overlap_matrix, layer_avg_overlap, moe_layers_list


def plot_overlap_heatmaps(overlap_matrix, layer_avg_overlap, tasks, moe_layers_list, output_dir, model_name, top_k=6):
    """Plot overlap heatmaps."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Average overlap across all layers (4x4 heatmap)
    plt.figure(figsize=(10, 8))
    
    # Create mask for upper triangle (since matrix is symmetric)
    # mask = np.triu(np.ones_like(layer_avg_overlap, dtype=bool), k=1)  # Removed mask to show full matrix
    
    sns.heatmap(layer_avg_overlap, 
                annot=True, 
                fmt='.3f',
                cmap='Blues',
                xticklabels=tasks,
                yticklabels=tasks,
                square=True,
                cbar_kws={'label': f'Top-{top_k} Expert Overlap Count (0-{top_k})'},
                mask=None)
    
    plt.title(f'Average Top-{top_k} Expert Overlap Across All Layers\nModel: {model_name}')
    plt.xlabel('Task')
    plt.ylabel('Task')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_expert_overlap_average.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Layer-wise overlap heatmaps (show first 16 layers)
    max_layers_to_show = min(16, len(moe_layers_list))
    layers_to_show = moe_layers_list[:max_layers_to_show]
    
    n_rows = int(np.ceil(max_layers_to_show / 4))
    n_cols = min(4, max_layers_to_show)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    if max_layers_to_show == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for i, layer_1idx in enumerate(layers_to_show):
        layer_idx = moe_layers_list.index(layer_1idx)  # Get index in the overlap_matrix
        layer_overlap = overlap_matrix[:, :, layer_idx]
        
        # Create mask for upper triangle
        # mask = np.triu(np.ones_like(layer_overlap, dtype=bool), k=1)  # Removed mask to show full matrix
        
        sns.heatmap(layer_overlap,
                   annot=True,
                   fmt='.2f',
                   cmap='Blues',
                   xticklabels=tasks,
                   yticklabels=tasks,
                   square=True,
                   cbar=False,
                   mask=None,
                   ax=axes[i])
        
        axes[i].set_title(f'Layer {layer_1idx}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
        # Only show labels on bottom and left edges
        if i >= (n_rows - 1) * n_cols:  # Bottom row
            axes[i].set_xlabel('Task')
            axes[i].tick_params(axis='x', rotation=45)
        else:
            axes[i].set_xticklabels([])
            
        if i % n_cols == 0:  # Left column
            axes[i].set_ylabel('Task')
        else:
            axes[i].set_yticklabels([])
    
    # Hide unused subplots
    for i in range(max_layers_to_show, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=0, vmax=top_k))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'Top-{top_k} Expert Overlap Count (0-{top_k})')
    
    plt.suptitle(f'Layer-wise Top-{top_k} Expert Overlap\nModel: {model_name}')
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(os.path.join(output_dir, f'{model_name}_expert_overlap_by_layer.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved overlap heatmaps to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert overlap between tasks")
    parser.add_argument("--model_name_or_path", type=str, required=True, 
                       help="HuggingFace model name or path")
    parser.add_argument("--model_short_name", type=str, required=True,
                       help="Short model name used in directory structure (e.g., qwen3-30b-coder)")
    parser.add_argument("--expert_scores_dir", type=str, default="outputs/expert_scores/cybersecurity",
                       help="Directory containing expert scores")
    parser.add_argument("--tasks", type=str, nargs="*", default=None,
                       help="List of task names. If not provided, will use all tasks in the directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for visualizations")
    parser.add_argument("--top_k", type=int, default=6,
                       help="Number of top experts to consider for overlap analysis")
    
    args = parser.parse_args()
    
    # Get MoE config from model
    n_layers, n_experts, model_top_k = get_moe_config(args.model_name_or_path)
    
    # Get task list
    if args.tasks is None:
        # Auto-discover tasks from directory
        if not os.path.exists(args.expert_scores_dir):
            raise ValueError(f"Expert scores directory not found: {args.expert_scores_dir}")
        
        tasks = [d for d in os.listdir(args.expert_scores_dir) 
                if os.path.isdir(os.path.join(args.expert_scores_dir, d))]
        tasks.sort()
        print(f"Auto-discovered tasks: {tasks}")
    else:
        tasks = args.tasks
        print(f"Using specified tasks: {tasks}")
    
    if len(tasks) < 2:
        raise ValueError("Need at least 2 tasks for overlap analysis")
    
    # Load expert scores for each task
    task_gate_scores = {}
    task_moe_layers = {}
    task_top_experts = {}
    
    common_moe_layers = None
    
    for task in tasks:
        print(f"\nProcessing task: {task}")
        gate_scores, moe_layers = load_expert_scores_for_task(
            args.expert_scores_dir, args.model_short_name, task, n_layers, n_experts, model_top_k
        )
        
        task_gate_scores[task] = gate_scores
        task_moe_layers[task] = moe_layers
        
        # Get top-k experts for each layer
        task_top_experts[task] = get_top_k_experts_per_layer(gate_scores, moe_layers, args.top_k)
        
        # Find common MoE layers across all tasks
        if common_moe_layers is None:
            common_moe_layers = moe_layers.copy()
        else:
            common_moe_layers = common_moe_layers.intersection(moe_layers)
    
    print(f"\nCommon MoE layers across all tasks: {len(common_moe_layers)} layers")
    print(f"Layers: {sorted(list(common_moe_layers))[:5]}...{sorted(list(common_moe_layers))[-5:]}" 
          if len(common_moe_layers) > 10 else f"Layers: {sorted(list(common_moe_layers))}")
    
    # Calculate overlap matrix
    overlap_matrix, layer_avg_overlap, moe_layers_list = calculate_overlap_matrix(
        task_top_experts, tasks, common_moe_layers, args.top_k
    )
    
    # Plot heatmaps
    plot_overlap_heatmaps(overlap_matrix, layer_avg_overlap, tasks, moe_layers_list, 
                         args.output_dir, args.model_short_name, args.top_k)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Model: {args.model_short_name}")
    print(f"Tasks analyzed: {len(tasks)}")
    print(f"Common MoE layers: {len(common_moe_layers)}")
    print(f"Top-{args.top_k} expert overlap analysis")
    
    print(f"\nAverage overlap matrix (overlap count 0-{args.top_k}):")
    print("Tasks:", tasks)
    for i, task1 in enumerate(tasks):
        row_str = f"{task1:25s}: "
        for j, task2 in enumerate(tasks):
            row_str += f"{layer_avg_overlap[i, j]:.1f}  "
        print(row_str)


if __name__ == "__main__":
    main()