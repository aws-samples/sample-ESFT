"""
Patch for GPT-OSS MoE models to enable expert weight logging.

This patch modifies the GptOssMLP to log expert selection
and routing weights during forward passes.

Note: GPT-OSS does NOT have shared experts.
Supported models: openai/gpt-oss-20b
"""
import os
import torch


def patch_gpt_oss():
    """Patch GptOssMLP to log expert weights."""
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import (
            GptOssMLP,
            GptOssDecoderLayer
        )
    except ImportError:
        print("Warning: GptOssMLP not found, skipping GPT-OSS patch")
        return
    
    # Patch DecoderLayer __init__ to add layer_idx to MoE block
    original_decoder_init = GptOssDecoderLayer.__init__
    
    def patched_decoder_init(self, config, layer_idx):
        original_decoder_init(self, config, layer_idx)
        # Add layer_idx to MoE block
        self.mlp.layer_idx = layer_idx
        self.mlp.config = config
    
    GptOssDecoderLayer.__init__ = patched_decoder_init
    
    # Patch forward to add logging
    original_forward = GptOssMLP.forward
    
    def patched_forward(self, hidden_states):
        # Router returns (router_scores, router_indices) in 4.57.x
        router_output = self.router(hidden_states)
        if len(router_output) == 3:
            # Older version: (router_logits, router_scores, router_indices)
            _, router_scores, router_indices = router_output
        else:
            # 4.57.x version: (router_scores, router_indices)
            router_scores, router_indices = router_output
        
        # Log expert weights if configured
        if hasattr(self, 'config') and hasattr(self.config, 'log_expert_weights') and self.config.log_expert_weights:
            assert hasattr(self.config, 'expert_log_dir'), "Please specify expert_log_dir in the config to log the expert weights"
            log_dir = self.config.expert_log_dir
            os.makedirs(log_dir, exist_ok=True)
            
            # Get layer index
            layer_idx = getattr(self, 'layer_idx', 'unknown')
            file_path = os.path.join(log_dir, f"expert_weights_{layer_idx}.txt")
            
            with open(file_path, "a") as f:
                tk_idx_list = router_indices.view(-1).tolist()
                tk_weight_list = router_scores.view(-1).tolist()
                f.write("\t".join([str(i) for i in tk_idx_list]) + "\t\t" + 
                       "\t".join([str(round(i, 4)) for i in tk_weight_list]) + "\n")
        
        # Continue with original forward logic
        routed_out = self.experts(hidden_states, router_indices, router_scores)
        return routed_out, router_scores
    
    GptOssMLP.forward = patched_forward
    print("✓ GptOssMLP patched successfully")
