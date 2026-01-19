"""
MoE Model Patching Module

This module provides monkey patches for various MoE (Mixture of Experts) models
to enable expert weight logging during inference.

Supported models:
- Qwen2 MoE
- Qwen3 MoE
- GLM-4 MoE
- GPT-OSS
"""

from .patch_qwen2_moe import patch_qwen2_moe
from .patch_qwen3_moe import patch_qwen3_moe
from .patch_glm4_moe import patch_glm4_moe
from .patch_gpt_oss import patch_gpt_oss


def apply_all_patches():
    """Apply patches to all supported MoE models."""
    print("Applying MoE logging patches...")
    patch_qwen2_moe()
    patch_qwen3_moe()
    patch_glm4_moe()
    patch_gpt_oss()
    print("MoE logging patches applied!")


# Auto-apply patches when imported
apply_all_patches()


__all__ = [
    'patch_qwen2_moe',
    'patch_qwen3_moe',
    'patch_glm4_moe',
    'patch_gpt_oss',
    'apply_all_patches',
]
