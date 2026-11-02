"""
酶分类预测头模型 - 完整特征版 (非压缩)。

该预测头在不改变外部接口的前提下，沿着以下流水线完成二分类打分：
Pairformer refinement -> single token projection -> pair-to-token readout ->
token fusion -> sample pooling -> logits。

与压缩版 (enzyme_classification_compressed.py) 的区别：
1. 输入特征：本版使用完整 2D pair embedding z [B, N, N, C_z]，经 Pairformer 精炼后
   通过门控池化压缩为 token-level；压缩版直接接受已压缩的 z [B, N, C_z]，跳过 Pairformer。
2. 核心模块：本版使用 CustomPairformerStack 做特征精炼；压缩版使用 Transformer Encoder。
3. 特征融合：本版分别用 s_mlp / z_mlp 提取后拼接再经 token_fusion_mlp 融合；
   压缩版直接 concat(s, z) 后用一个 input_proj 投影。
4. 计算量：本版因 Pairformer 的 O(N²) pair attention 而更重；压缩版仅 O(N) 复杂度。
"""

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn

# 兼容直接脚本执行方式
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from alphaenzyme.core import core
from alphaenzyme.core.core import Registry as R
from alphaenzyme.models.custom_pairformer import CustomPairformerStack


@R.register("EnzymeClassificationHead")
class EnzymeClassificationHead(core.Configurable, nn.Module):
    """
    酶分类预测头。

    该 head 使用 Pairformer 对输入 embedding 做一次轻量精炼，然后将
    single/pair 特征都读出为 token-level 表示，经过 token 融合与 sample-level
    门控池化后输出二分类 logits。

    Args:
        configs: 配置字典，包含:
            - c_s: single embedding 维度 (default: 384)
            - c_z: pair embedding 维度 (default: 128)
            - hidden_size: 隐藏层维度 (default: 512)
            - dropout: dropout 比例 (default: 0.1)
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.c_s = configs.get("c_s", 384)
        self.c_z = configs.get("c_z", 128)
        self.hidden_size = configs.get("hidden_size", 512)
        self.dropout = configs.get("dropout", 0.1)
        self.pool_eps = configs.get("pool_eps", 1e-6)

        self.pairformer_blocks = configs.get("pairformer_blocks", 2)
        self.n_heads = configs.get("n_heads", 16)
        self.n_heads_pair = configs.get("n_heads_pair", 4)
        self.blocks_per_ckpt = configs.get("blocks_per_ckpt", 1)

        self.pairformer = CustomPairformerStack(
            n_blocks=self.pairformer_blocks,
            n_heads=self.n_heads,
            no_heads_pair=self.n_heads_pair,
            c_z=self.c_z,
            c_s=self.c_s,
            dropout=self.dropout,
            blocks_per_ckpt=self.blocks_per_ckpt,
        )

        self.gate_ln = nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        self.gate_linear = nn.Linear(self.c_s, 1, bias=True)
        self.ln = nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)

        self.pre_pairformer_z_ln = nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_ln = nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_linear = nn.Linear(self.c_z, 1, bias=True)
        self.z_ln = nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)

        self.s_mlp = nn.Sequential(
            nn.Linear(self.c_s, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.z_mlp = nn.Sequential(
            nn.Linear(self.c_z, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.token_fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1),
        )

        self.use_drfp = configs.get("use_drfp", False)
        self.drfp_dim = configs.get("drfp_dim", 2048)
        self.drfp_injection = configs.get("drfp_injection", "z_pair_pre_pairformer_global")
        self.drfp_pair_scale_init = configs.get("drfp_pair_scale_init", 1.0)

        if self.use_drfp:
            if self.drfp_injection != "z_pair_pre_pairformer_global":
                raise ValueError(
                    f"Unsupported drfp_injection: {self.drfp_injection}"
                )

            self.drfp_pair_ln = nn.LayerNorm(self.drfp_dim, elementwise_affine=False, bias=False)
            self.drfp_pair_mlp = nn.Sequential(
                nn.Linear(self.drfp_dim, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, self.c_z),
            )
            self.drfp_pair_scale = nn.Parameter(torch.tensor(float(self.drfp_pair_scale_init)))

    def _build_masks(
        self,
        single_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """构造与输入 embedding dtype 对齐的 single/pair mask。"""
        single_mask = single_mask.to(dtype=dtype)
        pair_mask = single_mask[..., None] * single_mask[..., None, :]
        return single_mask, pair_mask

    def _inject_drfp_if_enabled(
        self,
        z_input: torch.Tensor,
        pair_mask: torch.Tensor,
        drfp: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        将 DRFP 投影到 pair 通道后，在 Pairformer 之前广播加到所有有效 token pair 上。
        """
        if not self.use_drfp or drfp is None:
            return z_input

        drfp_pair = self.drfp_pair_mlp(self.drfp_pair_ln(drfp))
        drfp_pair_bias = drfp_pair[:, None, None, :]
        return z_input + self.drfp_pair_scale * drfp_pair_bias * pair_mask[..., None]

    def _run_pairformer(
        self,
        s: torch.Tensor,
        z_input: torch.Tensor,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """运行 Pairformer，并保持与原实现一致的残差更新语义。"""
        s_update, z_update = self.pairformer(
            s,
            z_input,
            pair_mask=pair_mask,
        )
        return s + s_update, z + z_update

    def _weighted_mean(
        self,
        values: torch.Tensor,
        weights: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        """执行带权平均，并保持 clamp_min(self.pool_eps) 的数值路径不变。"""
        numerator = (weights * values).sum(dim=dim)
        denominator = weights.sum(dim=dim).clamp_min(self.pool_eps)
        return numerator / denominator

    def _project_single_tokens(
        self,
        s: torch.Tensor,
        single_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 single token 特征，以及最终 sample pooling 使用的 token gate。"""
        single_token_feat = self.s_mlp(self.ln(s)) * single_mask[..., None]
        token_pool_gate = self.gate_linear(self.gate_ln(s)).sigmoid() * single_mask[..., None]
        return single_token_feat, token_pool_gate

    def _project_pair_tokens(
        self,
        z: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """将 pair-level 表示门控聚合为 token-level pair 特征。"""
        pair_feature_proj = self.z_mlp(self.z_ln(z))
        pair_pool_gate = self.z_gate_linear(self.z_gate_ln(z)).sigmoid() * pair_mask[..., None]
        pair_token_feat = self._weighted_mean(pair_feature_proj, pair_pool_gate, dim=2)
        return pair_token_feat

    def _fuse_token_features(
        self,
        single_token_feat: torch.Tensor,
        pair_token_feat: torch.Tensor,
        single_mask: torch.Tensor,
    ) -> torch.Tensor:
        """拼接 single/pair token 特征并映射到统一的 token-level hidden space。"""
        fused_token_input = torch.cat([single_token_feat, pair_token_feat], dim=-1)
        fused_token_feat = self.token_fusion_mlp(fused_token_input) * single_mask[..., None]
        return fused_token_feat

    def _pool_tokens_to_sample(
        self,
        fused_token_feat: torch.Tensor,
        token_pool_gate: torch.Tensor,
    ) -> torch.Tensor:
        """对 token-level 特征做门控池化，得到 sample-level 表示。"""
        return self._weighted_mean(fused_token_feat, token_pool_gate, dim=1)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        is_ligand: torch.Tensor,
        single_mask: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        drfp: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            s: [B, N_token, C_s] single embedding
            z: [B, N_token, N_token, C_z] pair embedding
            is_ligand: [B, N_token] 为兼容上游调用保留，当前 head 不直接使用
            single_mask: [B, N_token] 有效 token mask
            conditions: 为兼容上游调用保留，当前 head 不直接使用
            drfp: [B, drfp_dim] DRFP 特征。若启用，会先投影到 c_z，再作为全局 pair bias
                广播到所有有效 token pair，并在 Pairformer 之前注入 z。

        Returns:
            dict: 保持框架兼容的输出格式，包含 'logits'
        """
        # 当前读出逻辑完全基于 single_mask / pair_mask，不做 ligand-aware pooling。
        del is_ligand, conditions

        # 1) 构造 mask。
        single_mask, pair_mask = self._build_masks(single_mask, dtype=s.dtype)

        # 2) 在 Pairformer 前对 z 做可选的 DRFP 注入。
        pairformer_input_z = self.pre_pairformer_z_ln(z)
        pairformer_input_z = self._inject_drfp_if_enabled(pairformer_input_z, pair_mask, drfp)

        # 3) Pairformer 残差更新。
        s, z = self._run_pairformer(s, pairformer_input_z, z, pair_mask)

        # 4) 将 single / pair 特征都投影为 token-level 表示。
        single_token_feat, token_pool_gate = self._project_single_tokens(s, single_mask)
        pair_token_feat = self._project_pair_tokens(z, pair_mask)

        # 5) 在 token 维融合 single/pair 特征。
        fused_token_feat = self._fuse_token_features(single_token_feat, pair_token_feat, single_mask)

        # 6) 将 token-level 特征池化为 sample-level 表示并输出 logits。
        sample_feature = self._pool_tokens_to_sample(fused_token_feat, token_pool_gate)
        logits = self.output_mlp(sample_feature)

        return {
            'logits': logits.squeeze(-1),
            'sample_feature': sample_feature,
        }

    def predict(
        self,
        s,
        z,
        is_ligand,
        single_mask,
        conditions=None,
        drfp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用 logits > 0 的阈值规则输出二分类标签。"""
        output = self.forward(s, z, is_ligand, single_mask, conditions, drfp=drfp)
        logits = output['logits']
        return (logits > 0).long()

    def predict_proba(
        self,
        s,
        z,
        is_ligand,
        single_mask,
        conditions=None,
        drfp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """输出二分类概率，顺序为 [P(negative), P(positive)]。"""
        output = self.forward(s, z, is_ligand, single_mask, conditions, drfp=drfp)
        logits = output['logits']
        probs = torch.sigmoid(logits)
        return torch.stack([1 - probs, probs], dim=-1)
