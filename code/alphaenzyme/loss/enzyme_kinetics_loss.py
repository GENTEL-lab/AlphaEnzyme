"""
酶动力学损失函数

支持部分标签缺失的多任务损失函数，结合 MSE Loss 和 ListMLE Loss。
参考 alphafold-ln-0212/openfold/utils/loss.py 的 ListMLE 实现。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from alphaenzyme.core import core
from alphaenzyme.core.core import Registry as R


def listMLE(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    带掩码的 ListMLE 损失

    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".

    防止数值塌缩的手段:
    1. max_pred_values 减法防止 exp 溢出
    2. eps 防止 log(0)

    Args:
        y_pred: [B] 预测值
        y_true: [B] 真实值
        mask: [B] 有效样本掩码
        eps: 数值稳定性常数

    Returns:
        标量损失值
    """
    # 只选择有效样本
    valid_indices = mask.bool()
    if valid_indices.sum() < 2:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    valid_pred = y_pred[valid_indices]
    valid_true = y_true[valid_indices]

    # 随机打乱以解决平局
    random_indices = torch.randperm(valid_pred.shape[0], device=y_pred.device)
    y_pred_shuffled = valid_pred[random_indices]
    y_true_shuffled = valid_true[random_indices]

    # 按真实值降序排序
    _, indices = y_true_shuffled.sort(descending=True)
    preds_sorted_by_true = y_pred_shuffled[indices]

    # 数值稳定性: 减去最大值
    max_pred_values = preds_sorted_by_true.max()
    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    # 计算累积和 (从后往前)
    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[0]), dim=0).flip(dims=[0])

    # 计算 ListMLE 损失
    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    return observation_loss.mean()


@R.register("loss.MaskedEnzymeKineticsLoss")
class MaskedEnzymeKineticsLoss(core.Configurable, nn.Module):
    """
    支持部分标签缺失的多任务酶动力学损失函数

    组合损失:
    - Masked MSE Loss (点估计)
    - Masked ListMLE Loss (排序学习)
    - 一致性损失: log_kcat - log_km ≈ log_kcat_km

    Args:
        configs: 配置字典，包含:
            - km_weight: Km 损失权重 (default: 1.0)
            - kcat_weight: Kcat 损失权重 (default: 1.0)
            - kcat_km_weight: Kcat/Km 损失权重 (default: 1.0)
            - mse_weight: MSE 损失权重 (default: 0.5)
            - listmle_weight: ListMLE 损失权重 (default: 0.5)
            - consistency_weight: 一致性损失权重 (default: 0.1)
            - eps: 数值稳定性常数 (default: 1e-8)
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # 任务权重
        self.km_weight = configs.get("km_weight", 1.0)
        self.kcat_weight = configs.get("kcat_weight", 1.0)
        self.kcat_km_weight = configs.get("kcat_km_weight", 1.0)

        # 损失类型权重
        self.mse_weight = configs.get("mse_weight", 0.5)
        self.listmle_weight = configs.get("listmle_weight", 0.5)

        # 一致性损失权重
        self.consistency_weight = configs.get("consistency_weight", 0.1)

        # 数值稳定性
        self.eps = configs.get("eps", 1e-8)

        # 损失名称
        self.name = configs.get("name", "MaskedEnzymeKineticsLoss")

    def masked_mse_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算带掩码的 MSE 损失

        Args:
            pred: [B] 预测值
            target: [B] 真实值
            mask: [B] 有效样本掩码

        Returns:
            标量损失值
        """
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = ((pred - target) ** 2) * mask
        return loss.sum() / (mask.sum() + self.eps)

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        per_sample_weight: Optional[torch.Tensor] = None,
        mode: str = 'train',
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算总损失

        Args:
            pred: 预测字典，包含 'log_km', 'log_kcat', 'log_kcat_km'
            target: 目标字典，包含 'log_km', 'log_kcat', 'log_kcat_km',
                    'has_km', 'has_kcat', 'has_kcat_km'
            per_sample_weight: 可选的样本权重
            mode: 'train' 或 'eval'

        Returns:
            (loss, weighted_loss) 元组
        """
        total_loss = torch.tensor(0.0, device=pred['log_km'].device, requires_grad=True)
        loss_breakdown = {}

        # Km Loss
        if target['has_km'].sum() > 0:
            km_mse = self.masked_mse_loss(
                pred['log_km'],
                target['log_km'],
                target['has_km']
            )
            km_listmle = listMLE(
                pred['log_km'],
                target['log_km'],
                target['has_km'],
                self.eps
            )
            km_loss = self.mse_weight * km_mse + self.listmle_weight * km_listmle
            total_loss = total_loss + self.km_weight * km_loss
            loss_breakdown['km_mse'] = km_mse.detach()
            loss_breakdown['km_listmle'] = km_listmle.detach()

        # Kcat Loss
        if target['has_kcat'].sum() > 0:
            kcat_mse = self.masked_mse_loss(
                pred['log_kcat'],
                target['log_kcat'],
                target['has_kcat']
            )
            kcat_listmle = listMLE(
                pred['log_kcat'],
                target['log_kcat'],
                target['has_kcat'],
                self.eps
            )
            kcat_loss = self.mse_weight * kcat_mse + self.listmle_weight * kcat_listmle
            total_loss = total_loss + self.kcat_weight * kcat_loss
            loss_breakdown['kcat_mse'] = kcat_mse.detach()
            loss_breakdown['kcat_listmle'] = kcat_listmle.detach()

        # Kcat/Km Loss
        if target['has_kcat_km'].sum() > 0:
            kcat_km_mse = self.masked_mse_loss(
                pred['log_kcat_km'],
                target['log_kcat_km'],
                target['has_kcat_km']
            )
            kcat_km_listmle = listMLE(
                pred['log_kcat_km'],
                target['log_kcat_km'],
                target['has_kcat_km'],
                self.eps
            )
            kcat_km_loss = self.mse_weight * kcat_km_mse + self.listmle_weight * kcat_km_listmle
            total_loss = total_loss + self.kcat_km_weight * kcat_km_loss
            loss_breakdown['kcat_km_mse'] = kcat_km_mse.detach()
            loss_breakdown['kcat_km_listmle'] = kcat_km_listmle.detach()

        # 一致性损失: log(kcat/km) ≈ log(kcat) - log(km)
        consistency_mask = target['has_km'] * target['has_kcat'] * target['has_kcat_km']
        if consistency_mask.sum() > 0 and self.consistency_weight > 0:
            pred_kcat_km_from_ratio = pred['log_kcat'] - pred['log_km']
            consistency_loss = self.masked_mse_loss(
                pred_kcat_km_from_ratio,
                pred['log_kcat_km'],
                consistency_mask
            )
            total_loss = total_loss + self.consistency_weight * consistency_loss
            loss_breakdown['consistency'] = consistency_loss.detach()

        # 应用样本权重
        weighted_loss = total_loss
        if per_sample_weight is not None:
            weighted_loss = total_loss * per_sample_weight.mean()

        return total_loss, weighted_loss

    def __call__(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        per_sample_weight: Optional[torch.Tensor] = None,
        mode: str = 'train',
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """调用 forward"""
        return self.forward(pred, target, per_sample_weight, mode, **kwargs)


@R.register("loss.BinaryClassificationLoss")
class BinaryClassificationLoss(nn.Module):
    """
    二分类损失函数

    使用 Binary Cross Entropy with Logits 损失。

    Args:
        pos_weight: 正样本权重，用于处理类别不平衡 (default: 1.0)
    """

    def __init__(self, configs=None):
        super().__init__()
        if configs is None:
            configs = {}
        self.pos_weight = configs.get("pos_weight", 1.0)
        self.label_smoothing = float(configs.get("label_smoothing", 0.0))

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算二分类损失

        Args:
            pred: [B] 预测 logits
            target: [B] 真实标签 (0 或 1)

        Returns:
            标量损失值
        """
        # target 应该是 0 或 1 的浮点数
        if target.dtype == torch.long:
            target = target.float()

        if self.label_smoothing > 0:
            target = target * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        return F.binary_cross_entropy_with_logits(
            pred,
            target,
            pos_weight=torch.tensor(self.pos_weight, device=pred.device) if self.pos_weight != 1.0 else None
        )

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.forward(pred, target)


@R.register("loss.PairwiseRankingLoss")
class PairwiseRankingLoss(nn.Module):
    """Pairwise Ranking Loss for reaction-level ranking.

    对每个 batch 内（属于同一 reaction）的所有 (positive, negative) 对计算：
        log(1 + exp(-(score_pos - score_neg)))

    要求每个 batch 同时包含正样本和负样本。
    """

    def __init__(self, configs=None):
        super().__init__()

    def forward(self, pred, target, pair_weights=None):
        """
        Args:
            pred: [B] 预测 logits
            target: [B] 真实标签 (0 或 1)
            pair_weights: [P, N] 的正负样本对权重，可选
        Returns:
            标量 loss 值
        """
        pos_mask = (target > 0.5)
        neg_mask = ~pos_mask

        pos_scores = pred[pos_mask]
        neg_scores = pred[neg_mask]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            # 不应该发生，因为按 reaction 分组 batch 必定有正负
            return pred.new_zeros(())

        # [P, 1] - [1, N] -> [P, N]
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [P, N]
        pairwise = F.softplus(-diff)  # [P, N]
        if pair_weights is not None:
            pair_weights = pair_weights.to(pairwise.device, dtype=pairwise.dtype)
            if pair_weights.shape != pairwise.shape:
                raise ValueError(
                    f"pair_weights shape mismatch: expected {pairwise.shape}, got {pair_weights.shape}"
                )
            return (pairwise * pair_weights).sum() / pair_weights.sum().clamp_min(1e-6)
        return pairwise.mean()


@R.register("loss.ReactionListNetLoss")
class ReactionListNetLoss(nn.Module):
    """Listwise ranking loss for one reaction batch.

    将同一个 reaction 内的所有候选视为一个列表，目标分布在正样本上均匀分布。
    这样可以直接优化“把正样本整体推向前列”的目标，而不是只看局部正负对。
    """

    def __init__(self, configs=None):
        super().__init__()
        configs = configs or {}
        self.temperature = float(configs.get("temperature", 1.0))

    def forward(self, pred, target):
        pos_mask = target > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return pred.new_zeros(())

        logits = pred / max(self.temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=0)
        target_probs = pos_mask.float()
        target_probs = target_probs / target_probs.sum().clamp_min(1.0)
        return -(target_probs * log_probs).sum()
