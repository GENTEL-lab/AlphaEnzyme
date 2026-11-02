"""Final mutation residual head and within-assay ranking objective."""
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sanitize(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=20.0, neginf=-20.0)

class CleanHead(nn.Module):
    def __init__(self, input_dim: int, alpha_dim: int, raw_dim: int, head_type: str, hidden_dim: int, bottleneck_dim: int, dropout: float):
        super().__init__()
        self.head_type = head_type
        self.alpha_dim = alpha_dim
        self.raw_dim = raw_dim
        if head_type == "linear":
            self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
        elif head_type == "residual_mlp":
            self.base = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 1))
            self.residual = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )
            self.scale = nn.Parameter(torch.tensor(-2.0))
        else:
            self.alpha_proj = nn.Sequential(nn.LayerNorm(alpha_dim), nn.Linear(alpha_dim, hidden_dim), nn.GELU())
            self.raw_proj = nn.Sequential(nn.LayerNorm(raw_dim), nn.Linear(raw_dim, hidden_dim), nn.GELU())
            self.gate = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 2))
            self.score = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type in {"linear", "mlp"}:
            return self.net(x).squeeze(-1)
        if self.head_type == "residual_mlp":
            return (self.base(x) + torch.sigmoid(self.scale) * self.residual(x)).squeeze(-1)
        alpha = self.alpha_proj(x[:, : self.alpha_dim])
        raw = self.raw_proj(x[:, self.alpha_dim : self.alpha_dim + self.raw_dim])
        weights = torch.softmax(self.gate(torch.cat([alpha, raw], dim=-1)), dim=-1)
        h = weights[:, :1] * alpha + weights[:, 1:] * raw
        return self.score(h).squeeze(-1)
def pairwise_loss(pred: torch.Tensor, target: torch.Tensor, assays: list[str], max_pairs_per_assay: int = 512) -> torch.Tensor:
    losses = []
    device = pred.device
    for assay in sorted(set(assays)):
        idx = [i for i, a in enumerate(assays) if a == assay]
        if len(idx) < 2:
            continue
        ii = torch.tensor(idx, device=device, dtype=torch.long)
        p = pred[ii]
        y = target[ii]
        diff = y[:, None] - y[None, :]
        pos = (diff > 1e-6).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        if pos.shape[0] > max_pairs_per_assay:
            pos = pos[torch.randperm(pos.shape[0], device=device)[:max_pairs_per_assay]]
        margin = p[pos[:, 1]] - p[pos[:, 0]]
        losses.append(F.softplus(margin).mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
