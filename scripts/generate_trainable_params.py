#!/usr/bin/env python3
"""
Generate trainable parameter list from ESFT expert_config.json for ms-swift training.
Used with --trainable_parameters (combined with --freeze_parameters_ratio 1.0)

This script can auto-detect the parameter naming pattern from the model if --model is provided.

Expert ID Conversion:
- When using Expert Parallel (EP), experts are distributed across EP ranks in round-robin fashion
- Global expert ID to local expert ID conversion: local_id = global_id // ep_size
- EP rank assignment: ep_rank = global_id % ep_size

Examples:
  EP=1: Global ID 79 -> Local ID 79 (on EP rank 0)
  EP=2: Global ID 79 -> Local ID 39 (on EP rank 1)
  EP=8: Global ID 79 -> Local ID 9  (on EP rank 7)
"""
import json
import sys
import argparse
import os


def get_model_config(model_path):
    """
    Get model configuration from HuggingFace.
    
    Args:
        model_path: Path or name of the model
    
    Returns:
        Model config dict, or None if failed
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return config
    except Exception as e:
        print(f"[WARNING] Failed to load model config: {e}", file=sys.stderr)
        return None


def detect_param_pattern(model_path):
    """
    Auto-detect parameter naming pattern from model structure.
    Fast method: reads safetensors index without loading weights.
    
    Args:
        model_path: Path or name of the model
    
    Returns:
        Detected parameter pattern string, or None if detection fails
    """
    try:
        from pathlib import Path
        
        print(f"[INFO] Detecting parameter pattern from: {model_path}", file=sys.stderr)
        
        # Try to find model index file
        if os.path.isdir(model_path):
            model_dir = Path(model_path)
        else:
            # Download from HF hub
            from huggingface_hub import snapshot_download
            print(f"[INFO] Downloading model index from HuggingFace...", file=sys.stderr)
            model_dir = Path(snapshot_download(
                model_path,
                allow_patterns=["*.json", "*.index.json"],
                ignore_patterns=["*.safetensors", "*.bin", "*.pth"]
            ))
        
        # Try safetensors index first (fastest)
        index_file = model_dir / "model.safetensors.index.json"
        if not index_file.exists():
            # Try pytorch index
            index_file = model_dir / "pytorch_model.bin.index.json"
        
        if index_file.exists():
            print(f"[INFO] Reading parameter names from: {index_file.name}", file=sys.stderr)
            with open(index_file, 'r') as f:
                index_data = json.load(f)
            param_names = list(index_data.get('weight_map', {}).keys())
        else:
            # Fallback: read from first safetensors file
            safetensors_files = list(model_dir.glob("*.safetensors"))
            if safetensors_files:
                print(f"[INFO] Reading parameter names from: {safetensors_files[0].name}", file=sys.stderr)
                from safetensors import safe_open
                with safe_open(safetensors_files[0], framework="pt") as f:
                    param_names = list(f.keys())
            else:
                print("[WARNING] No model index or safetensors files found", file=sys.stderr)
                return None
        
        # Find expert parameters
        expert_params = [name for name in param_names 
                        if 'expert' in name.lower() and 'shared' not in name.lower()]
        
        if not expert_params:
            print("[WARNING] No expert parameters found in model", file=sys.stderr)
            return None
        
        # Analyze pattern from first expert parameter
        sample = expert_params[0]
        print(f"[INFO] Sample expert parameter: {sample}", file=sys.stderr)
        
        # Detect pattern
        if '.mlp.experts.' in sample:
            parts = sample.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = i + 1
                if part == 'experts' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    expert_idx = i + 1
                    pattern_parts = parts[:expert_idx + 1]
                    pattern_parts[layer_idx] = '{layer}'
                    pattern_parts[expert_idx] = '{expert}'
                    pattern = '.'.join(pattern_parts)
                    print(f"[INFO] Detected pattern: {pattern}", file=sys.stderr)
                    return pattern
        elif '.experts.' in sample:
            parts = sample.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    layer_idx = i + 1
                if part == 'experts' and i + 1 < len(parts) and parts[i + 1].isdigit():
                    expert_idx = i + 1
                    pattern_parts = parts[:expert_idx + 1]
                    pattern_parts[layer_idx] = '{layer}'
                    pattern_parts[expert_idx] = '{expert}'
                    pattern = '.'.join(pattern_parts)
                    print(f"[INFO] Detected pattern: {pattern}", file=sys.stderr)
                    return pattern
        
        print(f"[WARNING] Could not detect pattern from: {sample}", file=sys.stderr)
        return None
        
    except Exception as e:
        print(f"[WARNING] Failed to detect pattern: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return None


def generate_trainable_params(expert_config_path, param_pattern='model.layers.{layer}.mlp.experts.{expert}', model_path=None, use_megatron_format=False, ep_size=1, total_experts=None):
    """
    Generate trainable parameter list from expert_config.json
    
    Args:
        expert_config_path: Path to expert_config.json
        param_pattern: Parameter naming pattern. If 'auto', will detect from model.
            Supported patterns:
            - 'auto' (auto-detect from model, requires --model)
            - 'model.layers.{layer}.mlp.experts.{expert}' (Qwen3-MoE, DeepSeek-V2, default)
            - 'model.layers.{layer}.experts.{expert}' (some other MoE models)
            - Custom pattern with {layer} and {expert} placeholders
        model_path: Path to model (required if param_pattern='auto')
        use_megatron_format: Convert to Megatron naming format
        ep_size: Expert parallel size for local expert ID conversion
        total_experts: Total number of experts per layer
    
    Returns:
        List of trainable parameter prefixes
    """
    # Auto-detect pattern if requested
    if param_pattern == 'auto':
        if not model_path:
            raise ValueError("--model is required when using --pattern auto")
        detected_pattern = detect_param_pattern(model_path)
        if detected_pattern:
            param_pattern = detected_pattern
            print(f"[INFO] Using detected pattern: {param_pattern}", file=sys.stderr)
        else:
            print("[WARNING] Auto-detection failed, using default pattern", file=sys.stderr)
            param_pattern = 'model.layers.{layer}.mlp.experts.{expert}'
    with open(expert_config_path, 'r') as f:
        config = json.load(f)
    
    experts = config['experts']
    shared_experts = config.get('shared_experts', False)
    non_expert_modules = config.get('non_expert_modules', False)
    
    # Auto-detect total_experts from model config if not provided
    if total_experts is None and model_path:
        model_config = get_model_config(model_path)
        if model_config:
            # Try different config keys for MoE models
            total_experts = getattr(model_config, 'num_experts', None)
            if total_experts is None:
                total_experts = getattr(model_config, 'num_experts_per_tok', None)
            if total_experts is None:
                total_experts = getattr(model_config, 'n_routed_experts', None)
            
            if total_experts:
                print(f"[INFO] Auto-detected total_experts={total_experts} from model config", file=sys.stderr)
    
    # Fallback to default
    if total_experts is None:
        total_experts = 128
        print(f"[WARNING] Could not detect total_experts, using default: {total_experts}", file=sys.stderr)
    
    trainable_params = []
    num_local_experts = total_experts // ep_size
    
    # 1. Add trainable experts
    for layer_idx, expert_ids in experts.items():
        for global_expert_id in expert_ids:
            # Determine which EP rank this expert belongs to
            ep_rank = global_expert_id % ep_size
            
            # Convert global expert ID to local expert ID
            # Formula: local_id = global_id // ep_size
            # This works because experts are distributed in round-robin fashion:
            # EP rank 0: experts [0, ep_size, 2*ep_size, ...]
            # EP rank 1: experts [1, ep_size+1, 2*ep_size+1, ...]
            local_expert_id = global_expert_id // ep_size
            
            # Use local_expert_id for Megatron format, global_expert_id for HF format
            expert_id_to_use = local_expert_id if use_megatron_format else global_expert_id
            
            # Generate parameter prefix using the pattern
            param_prefix = param_pattern.format(layer=layer_idx, expert=expert_id_to_use)
            
            # Convert to Megatron format if needed
            if use_megatron_format:
                # Megatron uses: decoder.layers.X.mlp.experts.local_experts.Y
                # HF uses: model.layers.X.mlp.experts.Y
                param_prefix = param_prefix.replace('model.layers', 'decoder.layers')
                param_prefix = param_prefix.replace('.mlp.experts.', '.mlp.experts.local_experts.')
            
            trainable_params.append(param_prefix)
    
    # 2. Add shared_experts if needed
    if shared_experts:
        for layer_idx in experts.keys():
            # Extract shared expert pattern from param_pattern
            # Replace .experts.{expert} with .shared_experts
            if '.mlp.experts.' in param_pattern:
                shared_pattern = param_pattern.rsplit('.experts.', 1)[0] + '.shared_experts'
            elif '.experts.' in param_pattern:
                shared_pattern = param_pattern.rsplit('.experts.', 1)[0] + '.shared_experts'
            else:
                # Fallback for custom patterns
                shared_pattern = f'model.layers.{layer_idx}.mlp.shared_experts'
            
            shared_prefix = shared_pattern.format(layer=layer_idx)
            trainable_params.append(shared_prefix)
    
    # 3. Add non_expert_modules if needed
    if non_expert_modules:
        print("Warning: non_expert_modules=true is not recommended with trainable_parameters method.", 
              file=sys.stderr)
        print("Consider using freeze_parameters_regex method instead.", file=sys.stderr)
        
        # Add common non-expert modules
        # Note: This list may be incomplete depending on specific model architecture
        common_modules = [
            'model.embed_tokens',
            'model.norm',
            'lm_head',
        ]
        trainable_params.extend(common_modules)
        
        # Add attention and layer_norm for each layer
        for layer_idx in experts.keys():
            trainable_params.extend([
                f'model.layers.{layer_idx}.self_attn',
                f'model.layers.{layer_idx}.input_layernorm',
                f'model.layers.{layer_idx}.post_attention_layernorm',
            ])
    
    return trainable_params


def generate_trainable_params_regex(expert_config_path, param_pattern='model.layers.{layer}.mlp.experts.{expert}', model_path=None, use_megatron_format=False, ep_size=1, total_experts=None):
    """
    Generate trainable parameter regex pattern from expert_config.json
    
    This function generates a single regex pattern that matches all trainable parameters,
    which is more efficient than using a list of prefixes.
    
    Args:
        expert_config_path: Path to expert_config.json
        param_pattern: Parameter naming pattern
        model_path: Path to model (required if param_pattern='auto')
        use_megatron_format: Convert to Megatron naming format
        ep_size: Expert parallel size for local expert ID conversion
        total_experts: Total number of experts per layer
    
    Returns:
        Regex pattern string that matches all trainable parameters
    """
    # Auto-detect pattern if requested
    if param_pattern == 'auto':
        if not model_path:
            raise ValueError("--model is required when using --pattern auto")
        detected_pattern = detect_param_pattern(model_path)
        if detected_pattern:
            param_pattern = detected_pattern
            print(f"[INFO] Using detected pattern: {param_pattern}", file=sys.stderr)
        else:
            print("[WARNING] Auto-detection failed, using default pattern", file=sys.stderr)
            param_pattern = 'model.layers.{layer}.mlp.experts.{expert}'
    
    with open(expert_config_path, 'r') as f:
        config = json.load(f)
    
    experts = config['experts']
    shared_experts = config.get('shared_experts', False)
    non_expert_modules = config.get('non_expert_modules', False)
    
    # Auto-detect total_experts from model config if not provided
    if total_experts is None and model_path:
        model_config = get_model_config(model_path)
        if model_config:
            total_experts = getattr(model_config, 'num_experts', None)
            if total_experts is None:
                total_experts = getattr(model_config, 'num_experts_per_tok', None)
            if total_experts is None:
                total_experts = getattr(model_config, 'n_routed_experts', None)
            
            if total_experts:
                print(f"[INFO] Auto-detected total_experts={total_experts} from model config", file=sys.stderr)
    
    if total_experts is None:
        total_experts = 128
        print(f"[WARNING] Could not detect total_experts, using default: {total_experts}", file=sys.stderr)
    
    # Group experts by layer
    layer_experts = {}
    for layer_idx, expert_ids in experts.items():
        local_expert_ids = []
        for global_expert_id in expert_ids:
            local_expert_id = global_expert_id // ep_size if use_megatron_format else global_expert_id
            local_expert_ids.append(local_expert_id)
        layer_experts[layer_idx] = sorted(local_expert_ids)
    
    # Build regex pattern
    regex_parts = []
    
    # 1. Expert parameters
    # Escape special regex characters in the pattern
    base_pattern = param_pattern
    
    # Convert to Megatron format if needed
    if use_megatron_format:
        base_pattern = base_pattern.replace('model.layers', 'decoder.layers')
        base_pattern = base_pattern.replace('.mlp.experts.', '.mlp.experts.local_experts.')
    
    # Escape dots for regex
    base_pattern_escaped = base_pattern.replace('.', '\\.')
    
    # Group by expert IDs to create more compact regex
    # For each layer, create a pattern like: decoder\.layers\.1\.mlp\.experts\.local_experts\.(9|79)
    for layer_idx, expert_ids in layer_experts.items():
        expert_ids_str = '|'.join(map(str, expert_ids))
        layer_pattern = base_pattern_escaped.replace('{layer}', str(layer_idx))
        layer_pattern = layer_pattern.replace('{expert}', f'({expert_ids_str})')
        regex_parts.append(layer_pattern)
    
    # 2. Shared experts if needed
    if shared_experts:
        for layer_idx in experts.keys():
            if '.mlp.experts.' in param_pattern:
                shared_pattern = param_pattern.rsplit('.experts.', 1)[0] + '.shared_experts'
            elif '.experts.' in param_pattern:
                shared_pattern = param_pattern.rsplit('.experts.', 1)[0] + '.shared_experts'
            else:
                shared_pattern = f'model.layers.{layer_idx}.mlp.shared_experts'
            
            if use_megatron_format:
                shared_pattern = shared_pattern.replace('model.layers', 'decoder.layers')
            
            shared_pattern_escaped = shared_pattern.replace('.', '\\.')
            shared_pattern_escaped = shared_pattern_escaped.replace('{layer}', str(layer_idx))
            regex_parts.append(shared_pattern_escaped)
    
    # 3. Non-expert modules if needed
    if non_expert_modules:
        print("Warning: non_expert_modules=true is not recommended with regex method.", 
              file=sys.stderr)
        print("Consider using freeze_parameters_regex method instead.", file=sys.stderr)
        
        # Add common non-expert modules
        prefix = 'decoder' if use_megatron_format else 'model'
        common_modules = [
            f'{prefix}\\.embed_tokens',
            f'{prefix}\\.norm',
            'lm_head',
        ]
        regex_parts.extend(common_modules)
        
        # Add attention and layer_norm for each layer
        for layer_idx in experts.keys():
            regex_parts.extend([
                f'{prefix}\\.layers\\.{layer_idx}\\.self_attn',
                f'{prefix}\\.layers\\.{layer_idx}\\.input_layernorm',
                f'{prefix}\\.layers\\.{layer_idx}\\.post_attention_layernorm',
            ])
    
    # Combine all patterns with OR
    final_regex = '|'.join(f'({part})' for part in regex_parts)
    
    return final_regex


def main():
    parser = argparse.ArgumentParser(
        description='Generate trainable_parameters list for ESFT training with ms-swift Megatron',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameter Pattern Examples:
  Auto-detect (recommended):
    python generate_trainable_params.py expert_config.json --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" --pattern auto
  
  Default (Qwen3-MoE, DeepSeek-V2):
    model.layers.{layer}.mlp.experts.{expert}
  
  Alternative pattern (some MoE models):
    model.layers.{layer}.experts.{expert}
  
  Custom pattern:
    Use {layer} and {expert} as placeholders

Examples:
  # Generate regex pattern (recommended for ms-swift)
  python generate_trainable_params.py expert_config.json --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" --pattern auto --format regex --megatron --ep-size 8
  
  # Auto-detect pattern from model
  python generate_trainable_params.py expert_config.json --model "Qwen/Qwen3-Coder-30B-A3B-Instruct" --pattern auto
  
  # Generate space-separated list with default pattern
  python generate_trainable_params.py expert_config.json
  
  # Generate line-separated list
  python generate_trainable_params.py expert_config.json --format lines
  
  # Use custom pattern
  python generate_trainable_params.py expert_config.json --pattern "model.layers.{layer}.experts.{expert}"
  
  # Save to file
  python generate_trainable_params.py expert_config.json --format regex --output params.txt
        """
    )
    parser.add_argument(
        'expert_config',
        type=str,
        help='Path to expert_config.json'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model path or name (required for --pattern auto)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['space', 'json', 'lines', 'regex'],
        default='space',
        help='Output format: space-separated (default), json array, newline-separated, or regex pattern'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='model.layers.{layer}.mlp.experts.{expert}',
        help='Parameter naming pattern with {layer} and {expert} placeholders. Use "auto" to auto-detect from model (requires --model)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file to save the parameters (optional, prints to stdout by default)'
    )
    parser.add_argument(
        '--megatron',
        action='store_true',
        help='Use Megatron parameter naming format (decoder.layers.X.mlp.experts.local_experts.Y)'
    )
    parser.add_argument(
        '--ep-size',
        type=int,
        default=1,
        help='Expert parallel size (default: 1). Used to convert global expert IDs to local IDs.'
    )
    parser.add_argument(
        '--total-experts',
        type=int,
        default=None,
        help='Total number of experts per layer (auto-detected from model config if --model is provided)'
    )
    
    args = parser.parse_args()
    
    # Generate regex pattern or parameter list based on format
    if args.format == 'regex':
        output = generate_trainable_params_regex(
            args.expert_config, args.pattern, args.model, 
            args.megatron, args.ep_size, args.total_experts
        )
        param_count = "N/A (regex pattern)"
    else:
        params = generate_trainable_params(
            args.expert_config, args.pattern, args.model, 
            args.megatron, args.ep_size, args.total_experts
        )
        param_count = len(params)
        
        # Format output
        if args.format == 'space':
            output = ' '.join(params)
        elif args.format == 'json':
            output = json.dumps(params, indent=2)
        else:  # lines
            output = '\n'.join(params)
    
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"[INFO] Output saved to {args.output}", file=sys.stderr)
        if args.format == 'regex':
            print(f"[INFO] Generated regex pattern", file=sys.stderr)
        else:
            print(f"[INFO] Total trainable parameter prefixes: {param_count}", file=sys.stderr)
        if args.pattern != 'auto':
            print(f"[INFO] Pattern used: {args.pattern}", file=sys.stderr)
    else:
        print(output)


if __name__ == '__main__':
    main()
