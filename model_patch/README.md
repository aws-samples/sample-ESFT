# MoE Model Patching Module

This module provides monkey patches for various MoE (Mixture of Experts) models to enable expert weight logging during inference.

## Supported Models

- **Qwen2 MoE** (`patch_qwen2_moe.py`)
  - Models: Qwen/Qwen2-57B-A14B-Instruct, Qwen/Qwen2.5-MoE-A2.7B-Instruct
  - Features: Includes shared expert support
  
- **Qwen3 MoE** (`patch_qwen3_moe.py`)
  - Models: Qwen/Qwen3-Coder-30B-A3B-Instruct
  - Features: No shared expert (different from Qwen2)

- **GLM-4 MoE** (`patch_glm4_moe.py`)
  - Models: zai-org/GLM-4.5-Air
  - Features: Includes shared expert, sigmoid-based routing with group selection

- **GPT-OSS** (`patch_gpt_oss.py`)
  - Models: openai/gpt-oss-20b
  - Features: No shared expert, softmax-based top-k routing

## Usage

### Basic Usage

Simply import the module before loading your model:

```python
import model_patch  # Auto-applies all patches

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")

# Enable expert logging
model.config.log_expert_weights = True
model.config.expert_log_dir = "./expert_logs"

# Use model normally - expert weights will be logged automatically
```

### Selective Patching

If you only want to patch specific models:

```python
from model_patch import patch_qwen3_moe

patch_qwen3_moe()  # Only patch Qwen3 MoE

# Load and use your model...
```

## Log Format

Each MoE layer generates a file: `expert_weights_{layer_idx}.txt`

Format:
```
expert_idx1  expert_idx2  ...  \t\t  weight1  weight2  ...
```

Each line represents one forward pass with selected expert indices and their routing weights.

Example:
```
0  5  12  \t\t  0.4521  0.3201  0.2278
3  7  15  \t\t  0.5123  0.2987  0.1890
```

## Adding Support for New Models

Follow these steps to add support for a new MoE model:

### Step 1: Create a New Patch File

Create a new file `model_patch/patch_<model_name>.py`:

```python
"""
Patch for <ModelName> MoE models to enable expert weight logging.
"""
import os
import torch


def patch_<model_name>_moe():
    """Patch <ModelName>MoE to log expert weights."""
    try:
        from transformers.models.<model_name>.modeling_<model_name> import (
            <ModelName>MoeBlock,
            <ModelName>DecoderLayer
        )
        import torch.nn.functional as F
    except ImportError:
        print("Warning: <ModelName>MoeBlock not found, skipping patch")
        return
    
    # Your patch implementation here
    # See existing patches for reference
    
    print("✓ <ModelName>MoeBlock patched successfully")
```

### Step 2: Identify Key Components

You need to identify these components in the model:

1. **MoE Block Class**: The class that implements the MoE layer
   - Example: `Qwen3MoeSparseMoeBlock`, `DeepseekV2MoE`

2. **Decoder Layer Class**: The class that contains the MoE block
   - Example: `Qwen3MoeDecoderLayer`

3. **Router/Gate**: Where expert selection happens
   - Usually: `self.gate(hidden_states)`

4. **Expert Selection Variables**: The variables containing selected experts and weights
   - Common names: `selected_experts`, `topk_idx`, `routing_weights`, `topk_weight`

### Step 3: Implement the Patch

Your patch should:

1. **Add layer_idx to MoE blocks** (if not already present):
```python
original_decoder_init = DecoderLayer.__init__

def patched_decoder_init(self, config, layer_idx):
    original_decoder_init(self, config, layer_idx)
    if isinstance(self.mlp, MoeBlock):
        self.mlp.layer_idx = layer_idx
        self.mlp.config = config

DecoderLayer.__init__ = patched_decoder_init
```

2. **Patch the forward method** to log expert weights:
```python
original_forward = MoeBlock.forward

def patched_forward(self, hidden_states):
    # ... original forward logic ...
    
    # After computing selected_experts and routing_weights:
    if hasattr(self, 'config') and hasattr(self.config, 'log_expert_weights') and self.config.log_expert_weights:
        assert hasattr(self.config, 'expert_log_dir'), "Please specify expert_log_dir"
        log_dir = self.config.expert_log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        layer_idx = getattr(self, 'layer_idx', 'unknown')
        file_path = os.path.join(log_dir, f"expert_weights_{layer_idx}.txt")
        
        with open(file_path, "a") as f:
            tk_idx_list = selected_experts.view(-1).tolist()
            tk_weight_list = routing_weights.view(-1).tolist()
            f.write("\t".join([str(i) for i in tk_idx_list]) + "\t\t" + 
                   "\t".join([str(round(i, 4)) for i in tk_weight_list]) + "\n")
    
    # ... continue with original forward logic ...
    return outputs

MoeBlock.forward = patched_forward
```

### Step 4: Register the Patch

Add your patch to `model_patch/__init__.py`:

```python
from .patch_<model_name> import patch_<model_name>_moe

def apply_all_patches():
    """Apply patches to all supported MoE models."""
    print("Applying MoE logging patches...")
    patch_qwen2_moe()
    patch_qwen3_moe()
    patch_glm4_moe()
    patch_gpt_oss()
    patch_<model_name>_moe()  # Add your patch here
    print("MoE logging patches applied!")

__all__ = [
    'patch_qwen2_moe',
    'patch_qwen3_moe',
    'patch_glm4_moe',
    'patch_gpt_oss',
    'patch_<model_name>_moe',  # Add to exports
    'apply_all_patches',
]
```

### Step 5: Test Your Patch

Create a test script:

```python
import model_patch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("your-model-name")
tokenizer = AutoTokenizer.from_pretrained("your-model-name")

# Enable logging
model.config.log_expert_weights = True
model.config.expert_log_dir = "./test_logs"

# Run inference
inputs = tokenizer("Test input", return_tensors="pt")
outputs = model(**inputs)

# Check if logs were created
import os
log_files = os.listdir("./test_logs")
print(f"Created {len(log_files)} log files")
```

## Tips and Best Practices

1. **Preserve Original Behavior**: Your patch should not change the model's output, only add logging.

2. **Handle Edge Cases**: Check if the model has shared experts, auxiliary losses, or other special components.

3. **Use Descriptive Names**: Variable names should match the original implementation when possible.

4. **Add Documentation**: Include docstrings explaining model-specific details.

5. **Test Thoroughly**: Verify that:
   - Logs are created correctly
   - Model outputs remain unchanged
   - Multi-GPU setups work properly

## Debugging

If your patch doesn't work:

1. **Check imports**: Ensure the model classes are imported correctly
2. **Inspect the forward method**: Use `inspect.getsource()` to see the original implementation
3. **Verify variable names**: Expert selection variables may have different names
4. **Check return values**: Some models return tuples, others return single tensors
5. **Test with small inputs**: Use a simple test case to isolate issues

## Common Issues

### Issue: No logs are generated
- Check if `model.config.log_expert_weights = True` is set
- Verify `model.config.expert_log_dir` is set correctly
- Ensure the model actually uses MoE layers (not all layers may be MoE)

### Issue: Wrong number of log files
- Check `layer_idx` assignment in the decoder layer patch
- Verify which layers are actually MoE layers (some may be dense)

### Issue: Incorrect expert indices
- Verify the variable names for expert selection
- Check if the model uses 0-based or 1-based indexing

## Contributing

When contributing a new patch:

1. Follow the existing code style
2. Add comprehensive documentation
3. Include example usage
4. Test with actual model inference
5. Update this README with the new model
