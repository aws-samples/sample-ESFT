"""
Patch for Qwen2 MoE models to enable expert weight logging.

This patch modifies the Qwen2MoeSparseMoeBlock to log expert selection
and routing weights during forward passes.
"""
import os
import torch


def patch_qwen2_moe():
    """Patch Qwen2MoeSparseMoeBlock to log expert weights."""
    try:
        from transformers.models.qwen2_moe.modeling_qwen2_moe import (
            Qwen2MoeSparseMoeBlock,
            Qwen2MoeDecoderLayer
        )
        import torch.nn.functional as F
    except ImportError:
        print("Warning: Qwen2MoeSparseMoeBlock not found, skipping Qwen2 MoE patch")
        return
    
    # Patch DecoderLayer __init__ to add layer_idx to MoE block
    original_decoder_init = Qwen2MoeDecoderLayer.__init__
    
    def patched_decoder_init(self, config, layer_idx):
        original_decoder_init(self, config, layer_idx)
        # Add layer_idx to MoE block if it exists
        if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
            self.mlp.layer_idx = layer_idx
            self.mlp.config = config
    
    Qwen2MoeDecoderLayer.__init__ = patched_decoder_init
    
    # Patch forward to add logging
    original_forward = Qwen2MoeSparseMoeBlock.forward
    
    def patched_forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # Router logits and expert selection
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Log expert weights if configured
        if hasattr(self, 'config') and hasattr(self.config, 'log_expert_weights') and self.config.log_expert_weights:
            assert hasattr(self.config, 'expert_log_dir'), "Please specify expert_log_dir in the config to log the expert weights"
            log_dir = self.config.expert_log_dir
            os.makedirs(log_dir, exist_ok=True)
            
            # Get layer index
            layer_idx = getattr(self, 'layer_idx', 'unknown')
            file_path = os.path.join(log_dir, f"expert_weights_{layer_idx}.txt")
            
            with open(file_path, "a") as f:
                tk_idx_list = selected_experts.view(-1).tolist()
                tk_weight_list = routing_weights.view(-1).tolist()
                f.write("\t".join([str(i) for i in tk_idx_list]) + "\t\t" + 
                       "\t".join([str(round(i, 4)) for i in tk_weight_list]) + "\n")
        
        # Continue with original forward logic
        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        # Add shared expert
        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output
        final_hidden_states = final_hidden_states + shared_expert_output
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits
    
    Qwen2MoeSparseMoeBlock.forward = patched_forward
    print("✓ Qwen2MoeSparseMoeBlock patched successfully")
