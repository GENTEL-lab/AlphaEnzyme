"""
Compressed enzyme classification head.

The input Pairformer pair embedding is compressed to a token-level vector
(`z`: [N, C_z] instead of [N, N, C_z]). A Transformer encoder fuses the
single and pair token features, and gated pooling produces the binary
enzyme-reaction logit used by the AlphaEnzyme classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class EnzymeClassificationHeadCompressed(nn.Module):
    """
    酶分类预测头 - 压缩特征版本

    使用 Transformer Encoder 处理融合后的 s 和 z token-level 特征，
    然后通过门控池化提取配体特征进行二分类。

    Args:
        configs: 配置字典，包含:
            - c_s: single embedding 维度 (default: 384)
            - c_z: pair embedding 维度 (default: 128)
            - hidden_size: 隐藏层维度 (default: 512)
            - dropout: dropout 比例 (default: 0.1)
            - n_transformer_layers: Transformer 层数 (default: 4)
            - n_heads: 注意力头数 (default: 8)
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # 维度配置
        self.c_s = configs.get("c_s", 384)
        self.c_z = configs.get("c_z", 128)
        self.hidden_size = configs.get("hidden_size", 512)
        self.dropout = configs.get("dropout", 0.1)
        self.pooling_mode = configs.get("pooling_mode", "ligand_only")
        self.global_pool_scale_init = float(configs.get("global_pool_scale_init", 0.5))
        self.main_input_mode = str(configs.get("main_input_mode", "sz")).lower()
        if self.main_input_mode not in {"sz", "s_only", "z_only", "none"}:
            raise ValueError(f"Unsupported main_input_mode: {self.main_input_mode}")

        # Transformer 配置
        self.n_transformer_layers = configs.get("n_transformer_layers", 4)
        self.n_heads = configs.get("n_heads_single", configs.get("n_heads", 8))

        # s 和 z 的融合维度
        self.fused_dim = self.c_s + self.c_z  # 384 + 128 = 512

        # 投影层：将融合特征投影到 hidden_size
        self.input_proj = nn.Linear(self.fused_dim, self.hidden_size)
        self.input_ln = nn.LayerNorm(self.hidden_size)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_transformer_layers,
            enable_nested_tensor=False,
        )

        # 门控池化层
        self.gate_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, bias=False)
        self.gate_linear = nn.Linear(self.hidden_size, 1, bias=True)
        self.pool_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, bias=False)

        # 分类 MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        if self.pooling_mode == "ligand_plus_global":
            self.global_pool_scale = nn.Parameter(torch.tensor(self.global_pool_scale_init))
        elif self.pooling_mode != "ligand_only":
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")

        # DRFP 融合配置
        self.use_drfp = configs.get("use_drfp", False)
        self.drfp_dim = configs.get("drfp_dim", 2048)
        self.use_extra_features = bool(configs.get("use_extra_features", False))
        self.extra_feature_dim = int(configs.get("extra_feature_dim", 0))
        self.extra_hidden_size = int(configs.get("extra_hidden_size", self.hidden_size))
        self.extra_fusion_mode = str(configs.get("extra_fusion_mode", "delta")).lower()
        self.extra_logit_scale_init = float(configs.get("extra_logit_scale_init", 0.5))
        self.extra_token_injection = bool(configs.get("extra_token_injection", False))
        self.extra_branch_scale_init = float(configs.get("extra_branch_scale_init", 1.0))
        raw_branch_dims = configs.get("extra_feature_branch_dims", [])
        self.extra_feature_branch_dims = [int(dim) for dim in (raw_branch_dims or []) if int(dim) > 0]
        if self.use_extra_features and not self.extra_feature_branch_dims and self.extra_feature_dim > 0:
            self.extra_feature_branch_dims = [self.extra_feature_dim]
        self.use_extra_feature_branches = len(self.extra_feature_branch_dims) > 1

        if self.use_drfp:
            self.drfp_ln = nn.LayerNorm(self.drfp_dim, elementwise_affine=False, bias=False)
            self.drfp_mlp = nn.Sequential(
                nn.Linear(self.drfp_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.drfp_scale = nn.Parameter(torch.tensor(1.0))

        if self.use_extra_features:
            if self.extra_feature_dim <= 0:
                raise ValueError("use_extra_features=True but extra_feature_dim <= 0")
            if sum(self.extra_feature_branch_dims) != self.extra_feature_dim:
                raise ValueError(
                    "sum(extra_feature_branch_dims) must equal extra_feature_dim: "
                    f"{self.extra_feature_branch_dims} vs {self.extra_feature_dim}"
                )
            if self.use_extra_feature_branches:
                if self.extra_fusion_mode not in {"branch_gate_sum", "branch_concat_mlp", "delta"}:
                    raise ValueError(f"Unsupported extra_fusion_mode: {self.extra_fusion_mode}")
                self.extra_branch_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(branch_dim, elementwise_affine=False, bias=False),
                            nn.Linear(branch_dim, self.extra_hidden_size),
                            nn.GELU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(self.extra_hidden_size, self.hidden_size),
                        )
                        for branch_dim in self.extra_feature_branch_dims
                    ]
                )
                self.extra_branch_scale = nn.Parameter(
                    torch.full((len(self.extra_feature_branch_dims),), self.extra_branch_scale_init)
                )
                fusion_branch_count = 1 + len(self.extra_feature_branch_dims)
                fusion_input_dim = fusion_branch_count * self.hidden_size
                if self.extra_fusion_mode == "branch_gate_sum":
                    self.extra_branch_fusion = nn.Sequential(
                        nn.LayerNorm(fusion_input_dim),
                        nn.Linear(fusion_input_dim, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, fusion_branch_count),
                    )
                elif self.extra_fusion_mode == "branch_concat_mlp":
                    self.extra_branch_fusion = nn.Sequential(
                        nn.LayerNorm(fusion_input_dim),
                        nn.Linear(fusion_input_dim, self.hidden_size * 2),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                    )
                else:
                    self.extra_branch_fusion = None
                self.extra_ln = None
                self.extra_proj = None
            else:
                self.extra_ln = nn.LayerNorm(self.extra_feature_dim, elementwise_affine=False, bias=False)
                self.extra_proj = nn.Sequential(
                    nn.Linear(self.extra_feature_dim, self.extra_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.extra_hidden_size, self.hidden_size),
                )
                self.extra_branch_proj = None
                self.extra_branch_scale = None
                self.extra_branch_fusion = None
            if self.extra_fusion_mode not in {"delta", "gated_add", "branch_gate_sum", "branch_concat_mlp"}:
                raise ValueError(f"Unsupported extra_fusion_mode: {self.extra_fusion_mode}")
            if not self.use_extra_feature_branches:
                if self.extra_fusion_mode == "delta":
                    self.extra_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 4),
                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.GELU(),
                    )
                elif self.extra_fusion_mode == "gated_add":
                    self.extra_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                    )
                    self.extra_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
                elif self.extra_fusion_mode == "branch_gate_sum":
                    self.extra_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, 2),
                    )
                else:
                    self.extra_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                    )
            self.extra_score = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1),
            )
            self.extra_logit_scale = nn.Parameter(torch.tensor(self.extra_logit_scale_init))

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        is_ligand: torch.Tensor,
        single_mask: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        drfp: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_scores: Optional[torch.Tensor] = None,
        overall_corr: Optional[torch.Tensor] = None,
        **unused_kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            s: [B, N_token, C_s] single embedding
            z: [B, N_token, C_z] compressed pair embedding (mean-pooled)
            is_ligand: [B, N_token] 配体 mask (1 表示配体 token)
            single_mask: [B, N_token] 有效 token mask
            conditions: 保留参数接口兼容性，但不使用
            drfp: [B, 2048] DRFP 特征 (可选)。若启用，会投影到 hidden_size，
                作为全局 token 偏置在 Transformer Encoder 之前注入到所有 token 上。

        Returns:
            dict: 包含 'logits' 预测值
        """
        batch_size, seq_len, _ = s.shape

        # 1. 融合 s 和 z: [B, N, C_s] + [B, N, C_z] -> [B, N, C_s + C_z]
        if self.main_input_mode == "s_only":
            z = torch.zeros_like(z)
        elif self.main_input_mode == "z_only":
            s = torch.zeros_like(s)
        elif self.main_input_mode == "none":
            s = torch.zeros_like(s)
            z = torch.zeros_like(z)
        fused = torch.cat([s, z], dim=-1)

        # 2. 投影到 hidden_size
        h = self.input_proj(fused)  # [B, N, hidden_size]
        h = self.input_ln(h)

        # 3. 构建 attention mask: True 表示该位置应被忽略
        # single_mask: 1 = valid, 0 = padding
        # TransformerEncoder src_key_padding_mask: True = ignore
        padding_mask = ~single_mask.bool()  # [B, N]

        # 4. 注入 DRFP 作为全局 token 偏置（DRFP 经投影后加到所有 token 上，再通过 Transformer 传播）
        drfp_hidden = None
        if self.use_drfp and drfp is not None:
            drfp_hidden = self.drfp_mlp(self.drfp_ln(drfp))  # [B, hidden_size]
            drfp_bias = self.drfp_scale * drfp_hidden[:, None, :]  # [B, 1, hidden_size]
            h = h + drfp_bias

        extra_hidden = None
        extra_branch_hidden_list = []
        if self.use_extra_features and extra_features is not None:
            if self.use_extra_feature_branches:
                extra_chunks = torch.split(extra_features, self.extra_feature_branch_dims, dim=-1)
                extra_branch_hidden_list = [
                    scale * projector(chunk)
                    for scale, projector, chunk in zip(
                        self.extra_branch_scale,
                        self.extra_branch_proj,
                        extra_chunks,
                    )
                ]
                if extra_branch_hidden_list:
                    extra_hidden = torch.stack(extra_branch_hidden_list, dim=0).mean(dim=0)
            else:
                extra_hidden = self.extra_proj(self.extra_ln(extra_features))
            if self.extra_token_injection and extra_hidden is not None:
                h = h + extra_hidden[:, None, :]

        # 5. Transformer Encoder
        h = self.transformer(h, src_key_padding_mask=padding_mask)  # [B, N, hidden_size]

        def pool_token_subset(token_repr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            gate = self.gate_linear(self.gate_ln(token_repr)).sigmoid()
            pooled_feature = (gate * token_repr).sum(dim=-2) / gate.sum(dim=-2).clamp_min(1e-6)
            pooled_score = (gate * self.classifier(self.pool_ln(token_repr))).sum(dim=-2)
            return pooled_feature, pooled_score

        # 6. 门控池化提取配体特征
        batch_logits = []
        batch_features = []
        batch_ligand_features = []
        batch_global_features = []

        for i in range(batch_size):
            ligand_mask = is_ligand[i].bool()
            valid_mask = single_mask[i].bool()

            h_i = h[i]  # [N, hidden_size]

            if ligand_mask.any():
                h_lig = h_i[ligand_mask]  # [N_ligand, hidden_size]
            else:
                h_lig = h_i[valid_mask]  # fallback to all valid tokens

            ligand_feature, ligand_score = pool_token_subset(h_lig)
            global_feature, global_score = pool_token_subset(h_i[valid_mask])
            if self.pooling_mode == "ligand_plus_global":
                sample_feature = ligand_feature + self.global_pool_scale * global_feature
                score = ligand_score + self.global_pool_scale * global_score
            else:
                sample_feature = ligand_feature
                score = ligand_score

            batch_ligand_features.append(ligand_feature)
            batch_global_features.append(global_feature)
            batch_features.append(sample_feature)
            batch_logits.append(score)

        logits = torch.stack(batch_logits, dim=0)  # [B, 1]
        sample_feature = torch.stack(batch_features, dim=0)  # [B, H]
        ligand_feature = torch.stack(batch_ligand_features, dim=0)  # [B, H]
        global_feature = torch.stack(batch_global_features, dim=0)  # [B, H]

        extra_delta = None
        extra_branch_weights = None
        if self.use_extra_features and extra_features is not None:
            if self.use_extra_feature_branches and extra_branch_hidden_list:
                branch_features = [sample_feature] + extra_branch_hidden_list
                if self.extra_fusion_mode == "branch_gate_sum":
                    branch_input = torch.cat(branch_features, dim=-1)
                    extra_branch_weights = torch.softmax(self.extra_branch_fusion(branch_input), dim=-1)
                    stacked_branch_features = torch.stack(branch_features, dim=1)
                    fused_hidden = (extra_branch_weights.unsqueeze(-1) * stacked_branch_features).sum(dim=1)
                elif self.extra_fusion_mode == "branch_concat_mlp":
                    fused_hidden = self.extra_branch_fusion(torch.cat(branch_features, dim=-1))
                elif self.extra_fusion_mode == "delta":
                    merged_extra_hidden = extra_hidden
                    fusion_input = torch.cat(
                        [
                            sample_feature,
                            merged_extra_hidden,
                            sample_feature - merged_extra_hidden,
                            sample_feature * merged_extra_hidden,
                        ],
                        dim=-1,
                    )
                    fused_hidden = self.extra_fusion(fusion_input)
                else:
                    gate_input = torch.cat([sample_feature, extra_hidden], dim=-1)
                    gate = torch.sigmoid(self.extra_gate(gate_input))
                    fused_hidden = sample_feature + gate * self.extra_fusion(gate_input)
            else:
                if self.extra_fusion_mode == "delta":
                    fusion_input = torch.cat(
                        [
                            sample_feature,
                            extra_hidden,
                            sample_feature - extra_hidden,
                            sample_feature * extra_hidden,
                        ],
                        dim=-1,
                    )
                    fused_hidden = self.extra_fusion(fusion_input)
                elif self.extra_fusion_mode == "branch_gate_sum":
                    branch_input = torch.cat([sample_feature, extra_hidden], dim=-1)
                    extra_branch_weights = torch.softmax(self.extra_fusion(branch_input), dim=-1)
                    stacked_branch_features = torch.stack([sample_feature, extra_hidden], dim=1)
                    fused_hidden = (extra_branch_weights.unsqueeze(-1) * stacked_branch_features).sum(dim=1)
                elif self.extra_fusion_mode == "branch_concat_mlp":
                    fused_hidden = self.extra_fusion(torch.cat([sample_feature, extra_hidden], dim=-1))
                else:
                    gate_input = torch.cat([sample_feature, extra_hidden], dim=-1)
                    gate = torch.sigmoid(self.extra_gate(gate_input))
                    fused_hidden = sample_feature + gate * self.extra_fusion(gate_input)
            extra_delta = self.extra_score(fused_hidden).squeeze(-1)
            logits = logits.squeeze(-1) + self.extra_logit_scale * extra_delta
        else:
            logits = logits.squeeze(-1)

        output = {
            'logits': logits,  # [B]
            'sample_feature': sample_feature,
            'ligand_feature': ligand_feature,
            'global_feature': global_feature,
        }
        drfp_repr = drfp_hidden if drfp_hidden is not None else torch.zeros_like(sample_feature)
        extra_repr = extra_hidden if extra_hidden is not None else torch.zeros_like(sample_feature)
        output['alpha_repr'] = torch.cat([sample_feature, drfp_repr, extra_repr], dim=-1)
        if drfp_hidden is not None:
            output['drfp_feature'] = drfp_hidden
        if extra_hidden is not None:
            output['extra_feature'] = extra_hidden
        if extra_delta is not None:
            output['extra_delta'] = extra_delta
        if extra_branch_weights is not None:
            output['extra_branch_weights'] = extra_branch_weights
        return output

    def predict(self, s, z, is_ligand, single_mask, conditions=None, drfp=None, extra_features=None) -> torch.Tensor:
        output = self.forward(s, z, is_ligand, single_mask, conditions, drfp=drfp, extra_features=extra_features)
        return (output['logits'] > 0).long()

    def predict_proba(self, s, z, is_ligand, single_mask, conditions=None, drfp=None, extra_features=None) -> torch.Tensor:
        output = self.forward(s, z, is_ligand, single_mask, conditions, drfp=drfp, extra_features=extra_features)
        probs = torch.sigmoid(output['logits'])
        return torch.stack([1 - probs, probs], dim=-1)
