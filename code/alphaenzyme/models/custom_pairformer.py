"""
自定义 PairformerStack 用于酶分类模型

这是一个独立的 PairformerStack 实现，支持自定义 no_heads_pair 参数，
不会影响 protenix 原始的 PairformerStack。
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from functools import partial

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_party', 'protenix_runtime'))

from protenix.model.modules.pairformer import PairformerBlock
from protenix.model.utils import checkpoint_blocks


class CustomPairformerStack(nn.Module):
    """
    自定义 PairformerStack，支持配置 no_heads_pair 参数

    专门用于酶分类/动力学预测模型，不影响其他模块。
    """

    def __init__(
        self,
        n_blocks: int = 48,
        n_heads: int = 16,
        c_z: int = 128,
        c_s: int = 384,
        dropout: float = 0.25,
        blocks_per_ckpt: Optional[int] = None,
        no_heads_pair: int = 4,
    ) -> None:
        """
        Args:
            n_blocks (int, optional): number of blocks. Defaults to 48.
            n_heads (int, optional): number of heads for single attention. Defaults to 16.
            c_z (int, optional): hidden dim for pair embedding. Defaults to 128.
            c_s (int, optional): hidden dim for single embedding. Defaults to 384.
            dropout (float, optional): dropout ratio. Defaults to 0.25.
            blocks_per_ckpt: number of blocks in each activation checkpoint. Defaults to None.
            no_heads_pair (int, optional): number of heads for pair attention. Defaults to 4.
        """
        super(CustomPairformerStack, self).__init__()
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.no_heads_pair = no_heads_pair
        self.blocks_per_ckpt = blocks_per_ckpt
        self.blocks = nn.ModuleList()

        for _ in range(n_blocks):
            block = PairformerBlock(
                n_heads=n_heads,
                c_z=c_z,
                c_s=c_s,
                dropout=dropout,
                no_heads_pair=no_heads_pair
            )
            self.blocks.append(block)

    def _prep_blocks(
        self,
        pair_mask: Optional[torch.Tensor],
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        blocks = [
            partial(
                b,
                pair_mask=pair_mask,
                use_memory_efficient_kernel=use_memory_efficient_kernel,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
            for b in self.blocks
        ]

        def clear_cache(b, *args, **kwargs):
            torch.cuda.empty_cache()
            return b(*args, **kwargs)

        if clear_cache_between_blocks:
            blocks = [partial(clear_cache, b) for b in blocks]
        return blocks

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        single_mask: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        use_memory_efficient_kernel: bool = False,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
        clear_cache_between_blocks: bool = False,
    ):
        """
        Args:
            s: [B, N_token, C_s] single embedding
            z: [B, N_token, N_token, C_z] pair embedding
            single_mask: [B, N_token] single mask (not used by PairformerBlock)
            pair_mask: [B, N_token, N_token] pair mask

        Returns:
            s: [B, N_token, C_s] updated single embedding
            z: [B, N_token, N_token, C_z] updated pair embedding
        """
        # Auto-enable clear_cache for large sequences during inference
        if z.shape[-2] > 2000 and (not self.training):
            clear_cache_between_blocks = True

        blocks = self._prep_blocks(
            pair_mask=pair_mask,
            use_memory_efficient_kernel=use_memory_efficient_kernel,
            use_deepspeed_evo_attention=use_deepspeed_evo_attention,
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
            clear_cache_between_blocks=clear_cache_between_blocks,
        )

        # Use checkpoint_blocks for gradient checkpointing support
        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None

        s, z = checkpoint_blocks(
            blocks,
            args=(s, z),
            blocks_per_ckpt=blocks_per_ckpt,
        )

        return s, z
