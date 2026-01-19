"""
Patch for GLM-4 MoE models to enable expert weight logging.

This patch modifies the Glm4MoeMoE to log expert selection
and routing weights during forward passes.

Note: GLM-4 MoE has shared experts.
Supported models: zai-org/GLM-4.5-Air
"""
import os
import torch


def patch_glm4_moe():
    """Patch Glm4MoeMoE to log expert weights."""
    try:
        from transformers.models.glm4_moe.modeling_glm4_moe import (
            Glm4MoeMoE,
            Glm4MoeDecoderLayer,
            Glm4MoeTopkRouter
        )
    except ImportError:
        print("Warning: Glm4MoeMoE not found, skipping GLM-4 MoE patch")
        return

    # Patch DecoderLayer __init__ to add layer_idx to MoE block
    original_decoder_init = Glm4MoeDecoderLayer.__init__

    def patched_decoder_init(self, config, layer_idx):
        original_decoder_init(self, config, layer_idx)
        # Add layer_idx to MoE block if it exists
        if isinstance(self.mlp, Glm4MoeMoE):
            self.mlp.layer_idx = layer_idx
            self.mlp.config = config

    Glm4MoeDecoderLayer.__init__ = patched_decoder_init

    # Detect version by checking if route_tokens_to_experts exists
    has_route_method = hasattr(Glm4MoeMoE, "route_tokens_to_experts")
    has_moe_method = hasattr(Glm4MoeMoE, "moe")

    # Patch the router to store indices for logging
    original_router_forward = Glm4MoeTopkRouter.forward

    def patched_router_forward(self, hidden_states):
        # Call original forward
        topk_indices, topk_weights = original_router_forward(self, hidden_states)
        # Store indices on the router for later retrieval by MoE
        self._last_topk_indices = topk_indices.clone()
        self._last_topk_weights = topk_weights.clone()
        return topk_indices, topk_weights

    Glm4MoeTopkRouter.forward = patched_router_forward

    # Patch forward to add logging
    original_forward = Glm4MoeMoE.forward

    def patched_forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Handle different API versions
        if has_route_method:
            # Main branch: gate returns router_logits, need to call route_tokens_to_experts
            router_logits = self.gate(hidden_states)
            topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        else:
            # 4.57.x: gate directly returns (topk_indices, topk_weights)
            topk_indices, topk_weights = self.gate(hidden_states)

        # Log expert weights if configured
        if (
            hasattr(self, "config")
            and hasattr(self.config, "log_expert_weights")
            and self.config.log_expert_weights
        ):
            assert hasattr(
                self.config, "expert_log_dir"
            ), "Please specify expert_log_dir in the config to log the expert weights"
            log_dir = self.config.expert_log_dir
            os.makedirs(log_dir, exist_ok=True)

            # Get layer index
            layer_idx = getattr(self, "layer_idx", "unknown")
            file_path = os.path.join(log_dir, f"expert_weights_{layer_idx}.txt")

            with open(file_path, "a") as f:
                # topk_indices should be integers (expert IDs)
                # topk_weights should be floats (routing weights)
                tk_idx_list = topk_indices.view(-1).tolist()
                tk_weight_list = topk_weights.view(-1).tolist()
                # Format: indices as integers, weights as floats
                f.write(
                    "\t".join([str(int(i)) for i in tk_idx_list])
                    + "\t\t"
                    + "\t".join([str(round(w, 4)) for w in tk_weight_list])
                    + "\n"
                )

        # Continue with original forward logic
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if has_moe_method:
            # 4.57.x: use self.moe()
            hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        else:
            # Main branch: use self.experts()
            hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states

    Glm4MoeMoE.forward = patched_forward
    print("✓ Glm4MoeMoE patched successfully")
