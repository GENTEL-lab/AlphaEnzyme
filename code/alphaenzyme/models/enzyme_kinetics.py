"""
酶动力学预测头模型

基于预计算的 Pairformer embedding，预测 Km、Kcat 和 Kcat/Km。
参考 alphafold-ln-0212/src/model.py 的 LigandAffinityHead 实现。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_party', 'protenix_runtime'))

from alphaenzyme.core import core
from alphaenzyme.core.core import Registry as R

# 导入 PairformerStack (使用 external/protenix 版本)
from protenix.model.modules.pairformer import PairformerStack


@R.register("EnzymeKineticsHead")
class EnzymeKineticsHead(core.Configurable, nn.Module):
    """
    酶动力学预测头

    多任务预测头，用于预测 Km、Kcat 和 Kcat/Km。
    使用门控池化机制聚合蛋白质和配体特征。

    Args:
        configs: 配置字典，包含:
            - c_s: single embedding 维度 (default: 384)
            - c_z: pair embedding 维度 (default: 128)
            - hidden_size: 隐藏层维度 (default: 512)
            - dropout: dropout 比例 (default: 0.1)
            - use_pairformer: 是否使用额外的 Pairformer 层 (default: False)
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # 维度配置
        self.c_s = configs.get("c_s", 384)
        self.c_z = configs.get("c_z", 128)
        self.hidden_size = configs.get("hidden_size", 512)
        self.dropout = configs.get("dropout", 0.1)

        # 可选: Pairformer 精炼层
        self.use_pairformer = configs.get("use_pairformer", False)
        if self.use_pairformer:
            # 使用 external/protenix 的 PairformerStack
            self.pairformer_stack = PairformerStack(
                n_blocks=configs.get("pairformer_blocks", 2),
                n_heads=configs.get("pairformer_heads", 8),
                c_s=self.c_s,
                c_z=self.c_z,
                dropout=self.dropout,
            )

        # 门控池化层 - 用于 single embedding
        self.gate_ln = nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)
        self.gate_linear = nn.Linear(self.c_s, 1, bias=True)
        self.ln = nn.LayerNorm(self.c_s, elementwise_affine=False, bias=False)

        # 门控池化层 - 用于 pair embedding
        self.z_gate_ln = nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)
        self.z_gate_linear = nn.Linear(self.c_z, 1, bias=True)
        self.z_ln = nn.LayerNorm(self.c_z, elementwise_affine=False, bias=False)

        # 条件嵌入 (pH, 温度)
        self.condition_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_size)
        )

        # 特征融合维度
        fusion_dim = self.c_s + self.c_z + self.hidden_size

        # 三个独立的预测头
        self.km_head = self._make_head(fusion_dim)
        self.kcat_head = self._make_head(fusion_dim)
        self.kcat_km_head = self._make_head(fusion_dim)

    def _make_head(self, in_dim: int) -> nn.Module:
        """创建预测头 MLP"""
        return nn.Sequential(
            nn.Linear(in_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        is_ligand: torch.Tensor,
        single_mask: torch.Tensor,
        conditions: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            s: [B, N_token, C_s] single embedding
            z: [B, N_token, N_token, C_z] pair embedding
            is_ligand: [B, N_token] 配体 mask (1 表示配体 token)
            single_mask: [B, N_token] 有效 token mask
            conditions: [B, 2] 条件特征 (pH, 温度)

        Returns:
            dict: 包含 log_km, log_kcat, log_kcat_km 预测值
        """
        batch_size = s.shape[0]

        # 如果启用 Pairformer，先处理 embedding
        if self.use_pairformer:
            # 创建 pair_mask
            pair_mask = single_mask.unsqueeze(-1) * single_mask.unsqueeze(-2)
            # 通过 Pairformer Stack
            s_update, z_update = self.pairformer_stack(
                s,
                z,
                pair_mask=pair_mask,
            )
            # 残差连接
            s = s + s_update
            z = z + z_update

        # 门控池化提取配体特征
        # 参考 LigandAffinityHead 的实现
        s_features = []
        z_features = []

        for i in range(batch_size):
            # 获取当前样本的 mask
            ligand_mask = is_ligand[i].bool()
            valid_mask = single_mask[i].bool()

            # 配体 token 的 single embedding
            s_i = s[i]  # [N_token, C_s]

            if ligand_mask.any():
                # 有配体 token
                s_ligand = s_i[ligand_mask]  # [N_ligand, C_s]

                # 门控池化
                gate = self.gate_linear(self.gate_ln(s_ligand)).sigmoid()  # [N_ligand, 1]
                s_feat = (gate * self.ln(s_ligand)).sum(dim=0)  # [C_s]
            else:
                # 没有配体 token，使用全局池化
                s_valid = s_i[valid_mask]
                gate = self.gate_linear(self.gate_ln(s_valid)).sigmoid()
                s_feat = (gate * self.ln(s_valid)).sum(dim=0)

            s_features.append(s_feat)

            # 配体-配体 pair embedding
            z_i = z[i]  # [N_token, N_token, C_z]

            if ligand_mask.any():
                # 提取配体-配体 pair
                z_ligand_mask = ligand_mask.unsqueeze(-1) & ligand_mask.unsqueeze(-2)  # [N_token, N_token]
                z_ligand = z_i[z_ligand_mask]  # [N_ligand^2, C_z]

                if z_ligand.numel() > 0:
                    num_ligands = ligand_mask.sum()
                    z_ligand = z_ligand.reshape(num_ligands, num_ligands, self.c_z)

                    # 门控池化
                    z_gate = self.z_gate_linear(self.z_gate_ln(z_ligand)).sigmoid()
                    z_feat = (z_gate * self.z_ln(z_ligand)).mean(dim=0).sum(dim=0)  # [C_z]
                else:
                    z_feat = torch.zeros(self.c_z, device=z.device, dtype=z.dtype)
            else:
                # 没有配体，使用全局 pair 特征
                z_valid_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
                z_valid = z_i[z_valid_mask]
                if z_valid.numel() > 0:
                    num_valid = valid_mask.sum()
                    z_valid = z_valid.reshape(num_valid, num_valid, self.c_z)
                    z_gate = self.z_gate_linear(self.z_gate_ln(z_valid)).sigmoid()
                    z_feat = (z_gate * self.z_ln(z_valid)).mean(dim=0).sum(dim=0)
                else:
                    z_feat = torch.zeros(self.c_z, device=z.device, dtype=z.dtype)

            z_features.append(z_feat)

        # Stack features
        s_features = torch.stack(s_features, dim=0)  # [B, C_s]
        z_features = torch.stack(z_features, dim=0)  # [B, C_z]

        # 条件嵌入
        cond_features = self.condition_embed(conditions)  # [B, hidden_size]

        # 特征融合
        combined = torch.cat([s_features, z_features, cond_features], dim=-1)  # [B, fusion_dim]

        # 多任务预测
        log_km = self.km_head(combined)  # [B, 1]
        log_kcat = self.kcat_head(combined)  # [B, 1]
        log_kcat_km = self.kcat_km_head(combined)  # [B, 1]

        return {
            'log_km': log_km.squeeze(-1),  # [B]
            'log_kcat': log_kcat.squeeze(-1),  # [B]
            'log_kcat_km': log_kcat_km.squeeze(-1),  # [B]
        }


@R.register("EnzymeKineticsModel")
class EnzymeKineticsModel(core.Configurable, nn.Module):
    """
    完整的酶动力学预测模型

    包装 EnzymeKineticsHead，提供统一的接口。
    """

    def __init__(self, kinetics_head, configs):
        super().__init__()
        self.configs = configs
        self.kinetics_head = kinetics_head

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            batch: 包含 input_features 的字典

        Returns:
            dict: 预测结果
        """
        input_features = batch.get('input_features', batch)

        s = input_features['s']
        z = input_features['z']
        is_ligand = input_features['is_ligand']
        single_mask = input_features['single_mask']
        conditions = input_features['conditions']

        return self.kinetics_head(s, z, is_ligand, single_mask, conditions)
