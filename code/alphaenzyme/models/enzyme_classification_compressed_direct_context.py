import torch
import torch.nn as nn
from typing import Dict, Optional


class EnzymeClassificationHeadCompressedDirectContext(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.c_s = configs.get("c_s", 384)
        self.c_z = configs.get("c_z", 128)
        self.hidden_size = configs.get("hidden_size", 512)
        self.dropout = configs.get("dropout", 0.1)
        self.pooling_mode = configs.get("pooling_mode", "ligand_only")
        self.global_pool_scale_init = float(configs.get("global_pool_scale_init", 0.5))
        self.main_input_mode = str(configs.get("main_input_mode", "sz")).lower()
        if self.main_input_mode not in {"sz", "s_only", "z_only", "none"}:
            raise ValueError(f"Unsupported main_input_mode: {self.main_input_mode}")

        self.n_transformer_layers = configs.get("n_transformer_layers", 4)
        self.n_heads = configs.get("n_heads_single", configs.get("n_heads", 8))
        self.fused_dim = self.c_s + self.c_z

        self.input_proj = nn.Linear(self.fused_dim, self.hidden_size)
        self.input_ln = nn.LayerNorm(self.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=self.hidden_size * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_transformer_layers,
            enable_nested_tensor=False,
        )

        self.gate_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, bias=False)
        self.gate_linear = nn.Linear(self.hidden_size, 1, bias=True)
        self.pool_ln = nn.LayerNorm(self.hidden_size, elementwise_affine=False, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        if self.pooling_mode == "ligand_plus_global":
            self.global_pool_scale = nn.Parameter(torch.tensor(self.global_pool_scale_init))
        elif self.pooling_mode != "ligand_only":
            raise ValueError(f"Unsupported pooling_mode: {self.pooling_mode}")

        self.use_drfp = bool(configs.get("use_drfp", False))
        self.drfp_dim = int(configs.get("drfp_dim", 2048))
        if self.use_drfp:
            self.drfp_ln = nn.LayerNorm(self.drfp_dim, elementwise_affine=False, bias=False)
            self.drfp_mlp = nn.Sequential(
                nn.Linear(self.drfp_dim, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.drfp_scale = nn.Parameter(torch.tensor(1.0))

        self.sample_external_dim = int(configs.get("sample_external_dim", configs.get("extra_feature_dim", 0)))
        self.sample_external_hidden_size = int(
            configs.get("extra_hidden_size", configs.get("sample_external_hidden_size", self.hidden_size))
        )
        self.sample_external_fusion_mode = str(
            configs.get("extra_fusion_mode", configs.get("sample_external_fusion_mode", "delta"))
        ).lower()
        self.sample_external_scale_init = float(
            configs.get("extra_logit_scale_init", configs.get("sample_external_scale_init", 0.5))
        )
        self.sample_external_token_injection = bool(
            configs.get("extra_token_injection", configs.get("sample_external_token_injection", False))
        )
        self.sample_external_branch_scale_init = float(
            configs.get("extra_branch_scale_init", configs.get("sample_external_branch_scale_init", 1.0))
        )
        raw_branch_dims = configs.get("extra_feature_branch_dims", configs.get("sample_external_branch_dims", []))
        self.sample_external_branch_dims = [int(dim) for dim in (raw_branch_dims or []) if int(dim) > 0]
        if self.sample_external_dim > 0 and not self.sample_external_branch_dims:
            self.sample_external_branch_dims = [self.sample_external_dim]
        self.use_sample_external_branches = len(self.sample_external_branch_dims) > 1

        self.sample_external_proj = None
        self.sample_external_ln = None
        self.sample_external_branch_proj = None
        self.sample_external_branch_scale = None
        self.sample_external_branch_fusion = None
        self.sample_external_fusion = None
        self.sample_external_gate = None
        self.sample_external_score = None
        self.sample_external_scale = None
        if self.sample_external_dim > 0:
            if sum(self.sample_external_branch_dims) != self.sample_external_dim:
                raise ValueError(
                    "sum(sample_external_branch_dims) must equal sample_external_dim: "
                    f"{self.sample_external_branch_dims} vs {self.sample_external_dim}"
                )
            if self.use_sample_external_branches:
                if self.sample_external_fusion_mode not in {"branch_gate_sum", "branch_concat_mlp", "delta"}:
                    raise ValueError(
                        f"Unsupported sample external fusion mode with branches: {self.sample_external_fusion_mode}"
                    )
                self.sample_external_branch_proj = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(branch_dim, elementwise_affine=False, bias=False),
                            nn.Linear(branch_dim, self.sample_external_hidden_size),
                            nn.GELU(),
                            nn.Dropout(self.dropout),
                            nn.Linear(self.sample_external_hidden_size, self.hidden_size),
                        )
                        for branch_dim in self.sample_external_branch_dims
                    ]
                )
                self.sample_external_branch_scale = nn.Parameter(
                    torch.full((len(self.sample_external_branch_dims),), self.sample_external_branch_scale_init)
                )
                fusion_branch_count = 1 + len(self.sample_external_branch_dims)
                fusion_input_dim = fusion_branch_count * self.hidden_size
                if self.sample_external_fusion_mode == "branch_gate_sum":
                    self.sample_external_branch_fusion = nn.Sequential(
                        nn.LayerNorm(fusion_input_dim),
                        nn.Linear(fusion_input_dim, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, fusion_branch_count),
                    )
                elif self.sample_external_fusion_mode == "branch_concat_mlp":
                    self.sample_external_branch_fusion = nn.Sequential(
                        nn.LayerNorm(fusion_input_dim),
                        nn.Linear(fusion_input_dim, self.hidden_size * 2),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                    )
            else:
                self.sample_external_ln = nn.LayerNorm(self.sample_external_dim, elementwise_affine=False, bias=False)
                self.sample_external_proj = nn.Sequential(
                    nn.Linear(self.sample_external_dim, self.sample_external_hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.sample_external_hidden_size, self.hidden_size),
                )
                if self.sample_external_fusion_mode == "delta":
                    self.sample_external_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 4),
                        nn.Linear(self.hidden_size * 4, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, self.hidden_size),
                        nn.GELU(),
                    )
                elif self.sample_external_fusion_mode == "gated_add":
                    self.sample_external_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                    )
                    self.sample_external_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
                elif self.sample_external_fusion_mode == "branch_gate_sum":
                    self.sample_external_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size, 2),
                    )
                elif self.sample_external_fusion_mode == "branch_concat_mlp":
                    self.sample_external_fusion = nn.Sequential(
                        nn.LayerNorm(self.hidden_size * 2),
                        nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        nn.Linear(self.hidden_size * 2, self.hidden_size),
                        nn.GELU(),
                    )
                else:
                    raise ValueError(f"Unsupported sample external fusion mode: {self.sample_external_fusion_mode}")
            self.sample_external_score = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1),
            )
            self.sample_external_scale = nn.Parameter(torch.tensor(self.sample_external_scale_init))

        self.context_dim = int(configs.get("context_dim", 0))
        self.context_scale_init = float(configs.get("context_scale_init", 1.0))
        self.context_score_scale_init = float(configs.get("context_score_scale_init", 0.25))
        self.use_context_score_bias = bool(configs.get("use_context_score_bias", True))
        if self.context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            if self.use_context_score_bias:
                self.context_score_proj = nn.Linear(1, self.hidden_size)
                self.context_score_scale = nn.Parameter(torch.tensor(self.context_score_scale_init))
            else:
                self.context_score_proj = None
                self.context_score_scale = None
            self.context_scale = nn.Parameter(torch.tensor(self.context_scale_init))
            self.main_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
            self.context_token_embedding = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        else:
            self.context_proj = None
            self.context_score_proj = None
            self.context_score_scale = None

        self.overall_corr_dim = int(configs.get("overall_corr_dim", 0))
        self.overall_corr_scale_init = float(configs.get("overall_corr_scale_init", 0.5))
        self.use_overall_corr = bool(configs.get("use_overall_corr", True))
        if self.overall_corr_dim > 0 and self.use_overall_corr:
            self.overall_corr_proj = nn.Sequential(
                nn.LayerNorm(self.overall_corr_dim),
                nn.Linear(self.overall_corr_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.hidden_size),
            )
            self.overall_corr_scale = nn.Parameter(torch.tensor(self.overall_corr_scale_init))
        else:
            self.overall_corr_proj = None
            self.overall_corr_scale = None

        fusion_input_dim = self.hidden_size
        self.has_extra_feature_branches = False
        if self.sample_external_dim > 0:
            fusion_input_dim += self.hidden_size * 3
            self.has_extra_feature_branches = True
        if self.context_dim > 0:
            fusion_input_dim += self.hidden_size * 3
            self.has_extra_feature_branches = True
        if self.overall_corr_proj is not None:
            fusion_input_dim += self.hidden_size
            self.has_extra_feature_branches = True

        if self.has_extra_feature_branches:
            self.feature_delta_head = nn.Sequential(
                nn.LayerNorm(fusion_input_dim),
                nn.Linear(fusion_input_dim, self.hidden_size),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, 1),
            )
        else:
            self.feature_delta_head = None

    def _pool_token_subset(self, token_repr: torch.Tensor):
        gate = self.gate_linear(self.gate_ln(token_repr)).sigmoid()
        pooled_feature = (gate * token_repr).sum(dim=-2) / gate.sum(dim=-2).clamp_min(1e-6)
        pooled_score = (gate * self.classifier(self.pool_ln(token_repr))).sum(dim=-2)
        return pooled_feature, pooled_score

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        is_ligand: torch.Tensor,
        single_mask: torch.Tensor,
        conditions: Optional[torch.Tensor] = None,
        drfp: Optional[torch.Tensor] = None,
        extra_features: Optional[torch.Tensor] = None,
        external_features: Optional[torch.Tensor] = None,
        context_features: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        context_scores: Optional[torch.Tensor] = None,
        overall_corr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del conditions
        if external_features is None and extra_features is not None:
            external_features = extra_features

        batch_size, seq_len, _ = s.shape
        if self.main_input_mode == "s_only":
            z = torch.zeros_like(z)
        elif self.main_input_mode == "z_only":
            s = torch.zeros_like(s)
        elif self.main_input_mode == "none":
            s = torch.zeros_like(s)
            z = torch.zeros_like(z)
        fused = torch.cat([s, z], dim=-1)
        h = self.input_ln(self.input_proj(fused))

        drfp_hidden = None
        if self.use_drfp and drfp is not None:
            drfp_hidden = self.drfp_mlp(self.drfp_ln(drfp))
            h = h + self.drfp_scale * drfp_hidden[:, None, :]

        sample_external_hidden = None
        sample_external_branch_hidden_list = []
        if self.sample_external_dim > 0 and external_features is not None:
            if self.use_sample_external_branches:
                external_chunks = torch.split(external_features, self.sample_external_branch_dims, dim=-1)
                sample_external_branch_hidden_list = [
                    scale * projector(chunk)
                    for scale, projector, chunk in zip(
                        self.sample_external_branch_scale,
                        self.sample_external_branch_proj,
                        external_chunks,
                    )
                ]
                if sample_external_branch_hidden_list:
                    sample_external_hidden = torch.stack(sample_external_branch_hidden_list, dim=0).mean(dim=0)
            else:
                sample_external_hidden = self.sample_external_proj(self.sample_external_ln(external_features))
            if self.sample_external_token_injection and sample_external_hidden is not None:
                h = h + sample_external_hidden[:, None, :]

        padding_mask = ~single_mask.bool()
        context_hidden = None
        if self.context_proj is not None and context_features is not None and context_mask is not None:
            context_hidden = self.context_proj(context_features)
            if self.context_score_proj is not None and context_scores is not None:
                score_bias = self.context_score_proj(context_scores.unsqueeze(-1))
                context_hidden = context_hidden + self.context_score_scale * score_bias
            context_hidden = context_hidden + self.context_token_embedding
            context_hidden = self.context_scale * context_hidden * context_mask.unsqueeze(-1)
            h = h + self.main_token_embedding
            h = torch.cat([h, context_hidden], dim=1)
            padding_mask = torch.cat([padding_mask, ~context_mask.bool()], dim=1)

        h = self.transformer(h, src_key_padding_mask=padding_mask)
        main_h = h[:, :seq_len, :]
        encoded_context_hidden = h[:, seq_len:] if context_hidden is not None else None

        batch_logits = []
        batch_features = []
        batch_ligand_features = []
        batch_global_features = []
        batch_context_pooled = []
        batch_external_hidden = []
        batch_corr_hidden = []

        for i in range(batch_size):
            ligand_mask = is_ligand[i].bool()
            valid_mask = single_mask[i].bool()
            h_i = main_h[i]

            if ligand_mask.any():
                h_lig = h_i[ligand_mask]
            else:
                h_lig = h_i[valid_mask]

            ligand_feature, ligand_score = self._pool_token_subset(h_lig)
            global_feature, global_score = self._pool_token_subset(h_i[valid_mask])
            if self.pooling_mode == "ligand_plus_global":
                sample_feature = ligand_feature + self.global_pool_scale * global_feature
                base_score = ligand_score + self.global_pool_scale * global_score
            else:
                sample_feature = ligand_feature
                base_score = ligand_score

            fusion_parts = [sample_feature]

            ext_hidden_i = None
            if self.sample_external_dim > 0 and external_features is not None:
                if self.use_sample_external_branches and sample_external_branch_hidden_list:
                    branch_hidden_i_list = [branch_hidden[i] for branch_hidden in sample_external_branch_hidden_list]
                    if self.sample_external_fusion_mode == "branch_gate_sum":
                        branch_features = [sample_feature] + branch_hidden_i_list
                        branch_input = torch.cat(branch_features, dim=-1)
                        branch_weights = torch.softmax(self.sample_external_branch_fusion(branch_input), dim=-1)
                        stacked_branch_features = torch.stack(branch_features, dim=0)
                        ext_hidden_i = (branch_weights.unsqueeze(-1) * stacked_branch_features).sum(dim=0)
                    elif self.sample_external_fusion_mode == "branch_concat_mlp":
                        ext_hidden_i = self.sample_external_branch_fusion(
                            torch.cat([sample_feature] + branch_hidden_i_list, dim=-1)
                        )
                    elif self.sample_external_fusion_mode == "delta":
                        merged_ext_hidden = sample_external_hidden[i]
                        ext_input = torch.cat(
                            [
                                sample_feature,
                                merged_ext_hidden,
                                sample_feature - merged_ext_hidden,
                                sample_feature * merged_ext_hidden,
                            ],
                            dim=-1,
                        )
                        ext_hidden_i = self.sample_external_fusion(ext_input)
                    else:
                        raise ValueError(
                            f"Unsupported sample external fusion mode in branch path: {self.sample_external_fusion_mode}"
                        )
                else:
                    raw_ext_hidden = sample_external_hidden[i]
                    if self.sample_external_fusion_mode == "delta":
                        ext_input = torch.cat(
                            [
                                sample_feature,
                                raw_ext_hidden,
                                sample_feature - raw_ext_hidden,
                                sample_feature * raw_ext_hidden,
                            ],
                            dim=-1,
                        )
                        ext_hidden_i = self.sample_external_fusion(ext_input)
                    elif self.sample_external_fusion_mode == "branch_gate_sum":
                        branch_input = torch.cat([sample_feature, raw_ext_hidden], dim=-1)
                        branch_weights = torch.softmax(self.sample_external_fusion(branch_input), dim=-1)
                        stacked_branch_features = torch.stack([sample_feature, raw_ext_hidden], dim=0)
                        ext_hidden_i = (branch_weights.unsqueeze(-1) * stacked_branch_features).sum(dim=0)
                    elif self.sample_external_fusion_mode == "branch_concat_mlp":
                        ext_hidden_i = self.sample_external_fusion(torch.cat([sample_feature, raw_ext_hidden], dim=-1))
                    else:
                        gate_input = torch.cat([sample_feature, raw_ext_hidden], dim=-1)
                        gate = torch.sigmoid(self.sample_external_gate(gate_input))
                        ext_hidden_i = sample_feature + gate * self.sample_external_fusion(gate_input)
                base_score = base_score + self.sample_external_scale * self.sample_external_score(ext_hidden_i)
                fusion_parts.extend([ext_hidden_i, sample_feature - ext_hidden_i, sample_feature * ext_hidden_i])

            ctx_pooled_i = None
            if encoded_context_hidden is not None and context_mask is not None:
                mask_i = context_mask[i].bool()
                if mask_i.any():
                    ctx_tokens = encoded_context_hidden[i][mask_i]
                    ctx_pooled_i = ctx_tokens.mean(dim=0)
                else:
                    ctx_pooled_i = torch.zeros_like(sample_feature)
                fusion_parts.extend([ctx_pooled_i, sample_feature - ctx_pooled_i, sample_feature * ctx_pooled_i])

            corr_hidden_i = None
            if self.overall_corr_proj is not None and overall_corr is not None:
                corr_hidden_i = self.overall_corr_scale * self.overall_corr_proj(overall_corr[i])
                fusion_parts.append(corr_hidden_i)

            if self.feature_delta_head is not None:
                feature_delta = self.feature_delta_head(torch.cat(fusion_parts, dim=-1))
                score = base_score + feature_delta
            else:
                score = base_score

            batch_ligand_features.append(ligand_feature)
            batch_global_features.append(global_feature)
            batch_features.append(sample_feature)
            batch_logits.append(score)
            batch_context_pooled.append(ctx_pooled_i if ctx_pooled_i is not None else torch.zeros_like(sample_feature))
            batch_external_hidden.append(ext_hidden_i if ext_hidden_i is not None else torch.zeros_like(sample_feature))
            batch_corr_hidden.append(corr_hidden_i if corr_hidden_i is not None else torch.zeros_like(sample_feature))

        logits = torch.stack(batch_logits, dim=0).squeeze(-1)
        sample_feature = torch.stack(batch_features, dim=0)
        ligand_feature = torch.stack(batch_ligand_features, dim=0)
        global_feature = torch.stack(batch_global_features, dim=0)
        context_pooled = torch.stack(batch_context_pooled, dim=0)
        external_hidden = torch.stack(batch_external_hidden, dim=0)
        corr_hidden = torch.stack(batch_corr_hidden, dim=0)

        output = {
            "logits": logits,
            "sample_feature": sample_feature,
            "ligand_feature": ligand_feature,
            "global_feature": global_feature,
            "context_pooled_feature": context_pooled,
            "external_feature": external_hidden,
            "overall_corr_feature": corr_hidden,
        }
        drfp_repr = drfp_hidden if drfp_hidden is not None else torch.zeros_like(sample_feature)
        output["alpha_repr"] = torch.cat([sample_feature, drfp_repr, external_hidden], dim=-1)
        if drfp_hidden is not None:
            output["drfp_feature"] = drfp_hidden
        return output

    def predict(self, s, z, is_ligand, single_mask, conditions=None, **kwargs) -> torch.Tensor:
        output = self.forward(s, z, is_ligand, single_mask, conditions=conditions, **kwargs)
        return (output["logits"] > 0).long()

    def predict_proba(self, s, z, is_ligand, single_mask, conditions=None, **kwargs) -> torch.Tensor:
        output = self.forward(s, z, is_ligand, single_mask, conditions=conditions, **kwargs)
        probs = torch.sigmoid(output["logits"])
        return torch.stack([1 - probs, probs], dim=-1)
