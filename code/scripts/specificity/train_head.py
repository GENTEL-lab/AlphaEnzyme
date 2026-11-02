#!/usr/bin/env python
"""Train an alpha_repr specificity head from a precomputed frozen representation cache."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MAX_EPOCHS = 3

from alphaenzyme.loss.enzyme_kinetics_loss import PairwiseRankingLoss  # noqa: E402


class AlphaReprSpecificityHead(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, alpha_repr: torch.Tensor) -> torch.Tensor:
        return self.net(alpha_repr).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--repr_cache", type=Path, nargs="+", default=None)
    parser.add_argument(
        "--repr_cache_list",
        type=Path,
        default=None,
        help="Optional text file with one cache path per line. Useful when many cache parts are used.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.04)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight_decay", type=float, default=2e-3)
    parser.add_argument("--pairwise_weight", type=float, default=4.0)
    parser.add_argument("--bce_weight", type=float, default=0.0)
    parser.add_argument("--distill_weight", type=float, default=1.0)
    parser.add_argument(
        "--residual_scale_l2",
        type=float,
        default=0.0,
        help="L2 penalty that keeps the frozen-base residual scale close to residual_scale_target.",
    )
    parser.add_argument(
        "--residual_scale_target",
        type=float,
        default=float("nan"),
        help="Target value for residual scale L2. Defaults to base_logit_residual_init.",
    )
    parser.add_argument(
        "--head_logit_l2",
        type=float,
        default=0.0,
        help="L2 penalty on the learned alpha head logits, useful for small residual corrections.",
    )
    parser.add_argument(
        "--base_logit_residual_init",
        type=float,
        default=0.0,
        help="Score as alpha_head(alpha_repr) + scale * frozen base logit.",
    )
    parser.add_argument("--train_base_logit_residual", action="store_true")
    parser.add_argument("--seed", type=int, default=62340)
    parser.add_argument("--init_from_base_alpha_head", action="store_true")
    parser.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Optionally save numbered intermediate checkpoints; the final checkpoint is always saved.",
    )
    return parser.parse_args()


def load_alpha_head_from_payload(payload: dict, head: nn.Module) -> None:
    state = payload.get("specificity_alpha_head")
    if not state:
        return
    first = state.get("net.1.weight")
    if first is None or tuple(first.shape) != tuple(head.state_dict()["net.1.weight"].shape):
        return
    missing, unexpected = head.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"Warning: partial alpha head init missing={missing} unexpected={unexpected}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cache_paths = list(args.repr_cache or [])
    if args.repr_cache_list is not None:
        listed = [
            (args.repr_cache_list.resolve().parent / line.strip()).resolve()
            for line in args.repr_cache_list.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        cache_paths.extend(listed)
    if not cache_paths:
        raise ValueError("Provide at least one --repr_cache path or --repr_cache_list file.")

    base_payload = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    cache_payloads = [torch.load(path, map_location="cpu", weights_only=True) for path in cache_paths]
    batches = []
    for payload in cache_payloads:
        batches.extend(payload["batches"])
    if not batches:
        raise ValueError(f"empty repr cache: {args.repr_cache}")

    head = AlphaReprSpecificityHead(1152, args.hidden_size, args.layers, args.dropout).to(device)
    if args.base_logit_residual_init != 0:
        for module in reversed(list(head.net.modules())):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                break
    if args.init_from_base_alpha_head and isinstance(base_payload, dict):
        load_alpha_head_from_payload(base_payload, head)

    residual_scale = nn.Parameter(torch.tensor(float(args.base_logit_residual_init), device=device))
    residual_scale.requires_grad_(bool(args.train_base_logit_residual))
    optim_params = list(head.parameters())
    if residual_scale.requires_grad:
        optim_params.append(residual_scale)
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
    pairwise_loss = PairwiseRankingLoss({})
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else len(batches)
    metrics_history = []

    run_meta = {"base_classifier_frozen": True, "base_classifier_source": "released_specificity_classifier"}
    run_meta.update(
        {
            "specificity_head_mode": "alpha_repr",
            "specificity_alpha_repr_dim": 1152,
            "specificity_alpha_hidden_size": args.hidden_size,
            "specificity_alpha_layers": args.layers,
            "specificity_alpha_dropout": args.dropout,
            "specificity_alpha_head_lr": args.lr,
            "alpha_repr_cache": [str(path) for path in cache_paths],
            "alpha_repr_cache_meta": [payload.get("meta", {}) for payload in cache_payloads],
            "alpha_head_train_script": str(Path(__file__).resolve()),
            "pairwise_weight": args.pairwise_weight,
            "bce_weight": args.bce_weight,
            "distill_weight": args.distill_weight,
            "specificity_alpha_base_logit_residual_init": args.base_logit_residual_init,
            "specificity_alpha_base_logit_residual_trainable": bool(args.train_base_logit_residual),
            "max_epochs": args.epochs,
            "seed": args.seed,
        }
    )
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = list(range(len(batches)))
        random.shuffle(order)
        total = {"loss": 0.0, "pairwise": 0.0, "bce": 0.0, "distill": 0.0}
        pbar = tqdm(range(steps_per_epoch), desc=f"alpha-head epoch {epoch}/{args.epochs}")
        for step_idx in pbar:
            batch = batches[order[step_idx % len(order)]]
            alpha_repr = batch["alpha_repr"].to(device=device, dtype=torch.float32, non_blocking=True)
            labels = batch["labels"].to(device=device, dtype=torch.float32, non_blocking=True)
            teacher_logits = batch["teacher_logits"].to(device=device, dtype=torch.float32, non_blocking=True)

            head_logits = head(alpha_repr)
            logits = head_logits + residual_scale * teacher_logits
            pair_loss = pairwise_loss(logits, labels)
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            distill_loss = F.mse_loss(logits, teacher_logits)
            loss = args.pairwise_weight * pair_loss + args.bce_weight * bce_loss + args.distill_weight * distill_loss
            if args.residual_scale_l2 > 0:
                residual_target = (
                    float(args.base_logit_residual_init)
                    if math.isnan(float(args.residual_scale_target))
                    else float(args.residual_scale_target)
                )
                target_tensor = torch.tensor(residual_target, device=device, dtype=residual_scale.dtype)
                loss = loss + args.residual_scale_l2 * (residual_scale - target_tensor).pow(2)
            if args.head_logit_l2 > 0:
                loss = loss + args.head_logit_l2 * head_logits.pow(2).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total["loss"] += float(loss.detach().cpu())
            total["pairwise"] += float(pair_loss.detach().cpu())
            total["bce"] += float(bce_loss.detach().cpu())
            total["distill"] += float(distill_loss.detach().cpu())
            if (step_idx + 1) % 50 == 0:
                pbar.set_postfix({k: v / (step_idx + 1) for k, v in total.items()})

        metrics = {f"train_{k}": v / max(steps_per_epoch, 1) for k, v in total.items()}
        metrics["epoch"] = epoch
        metrics["base_logit_residual_scale"] = float(residual_scale.detach().cpu())
        metrics_history.append(metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        save_intermediate = (
            args.save_every > 0
            and epoch < args.epochs
            and epoch % args.save_every == 0
        )
        save_final = epoch == args.epochs
        if save_intermediate or save_final:
            out_payload = {"classifier": base_payload["classifier"]}
            out_payload["specificity_alpha_head"] = head.state_dict()
            out_payload["specificity_alpha_base_logit_residual_scale"] = residual_scale.detach().cpu()
            out_payload["run_meta"] = copy.deepcopy(run_meta)
            out_payload["metrics"] = metrics
            out_payload["optimizer"] = optimizer.state_dict()
            checkpoint_name = (
                "specificity_final.pt"
                if save_final
                else f"specificity_checkpoint_{epoch}.pt"
            )
            torch.save(out_payload, args.output_dir / checkpoint_name)

        (args.output_dir / "metrics_history.json").write_text(
            json.dumps(metrics_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
