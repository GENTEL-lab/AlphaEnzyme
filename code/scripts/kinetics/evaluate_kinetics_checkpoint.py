#!/usr/bin/env python
"""Evaluate a multitask kinetics checkpoint on a held-out CSV split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from train_multitask_with_replay import (  # noqa: E402
    DEFAULT_EXTRA_SPLIT_DIM,
    KineticsDataset,
    KineticsHeads,
    apply_drfp_input_policy,
    TASK_COLUMNS,
    apply_ratio_prediction_mode,
    build_classifier,
    denormalize,
    forward_kinetics_heads,
    is_source_specific_stats,
    kinetics_collate,
    load_full_feature_stats,
    load_task_branch_weight_map,
    load_source_specs_json,
    move_batch,
)


def safe_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(pearsonr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")


def safe_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(spearmanr(y_true, y_pred).statistic)
    except Exception:
        return float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan"),
        "pearson": safe_pearson(y_true, y_pred),
        "spearman": safe_spearman(y_true, y_pred),
        "p1mag_percent": float(np.mean(abs_err < 1.0) * 100.0),
        "p0_5mag_percent": float(np.mean(abs_err < 0.5) * 100.0),
    }


def load_checkpoint(path: Path) -> tuple[dict, dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if "classifier" not in payload or "kinetics_heads" not in payload:
        raise ValueError(f"{path} is not a multitask checkpoint with classifier/kinetics_heads")
    run_meta = payload.get("run_meta") or {}
    stats = run_meta.get("label_stats")
    if not stats:
        label_stats_path = path.parent / "label_stats.json"
        if not label_stats_path.exists():
            raise FileNotFoundError(f"Missing label stats in checkpoint and {label_stats_path}")
        stats = json.loads(label_stats_path.read_text(encoding="utf-8"))
    return payload, run_meta, stats


def apply_posthoc_raw_affine(pred: np.ndarray, run_meta: dict) -> np.ndarray:
    """Apply optional per-task affine calibration in raw label space."""
    affine = run_meta.get("kin_posthoc_raw_affine")
    if not affine:
        return pred
    out = pred.copy()
    for idx, name in enumerate(TASK_COLUMNS):
        cfg = affine.get(name) if isinstance(affine, dict) else None
        if not cfg:
            continue
        scale = float(cfg.get("scale", 1.0))
        bias = float(cfg.get("bias", 0.0))
        out[:, idx] = out[:, idx] * scale + bias
    return out


def apply_posthoc_raw_piecewise(pred: np.ndarray, run_meta: dict) -> np.ndarray:
    """Apply optional valid-fit piecewise calibration in raw label space."""
    piecewise = run_meta.get("kin_posthoc_raw_piecewise")
    if not piecewise:
        return pred
    out = pred.copy()
    for idx, name in enumerate(TASK_COLUMNS):
        cfg = piecewise.get(name) if isinstance(piecewise, dict) else None
        if not cfg:
            continue
        x_knots = np.asarray(cfg.get("x_knots", []), dtype=np.float64)
        y_knots = np.asarray(cfg.get("y_knots", []), dtype=np.float64)
        if x_knots.ndim != 1 or y_knots.ndim != 1 or len(x_knots) != len(y_knots) or len(x_knots) < 2:
            continue
        order = np.argsort(x_knots)
        x_knots = x_knots[order]
        y_knots = y_knots[order]
        calibrated = np.interp(out[:, idx], x_knots, y_knots, left=y_knots[0], right=y_knots[-1])
        blend = float(cfg.get("blend", 1.0))
        out[:, idx] = (1.0 - blend) * out[:, idx] + blend * calibrated
    return out


def apply_posthoc_raw_poly(pred: np.ndarray, run_meta: dict) -> np.ndarray:
    """Apply optional valid-fit polynomial calibration in raw label space."""
    poly = run_meta.get("kin_posthoc_raw_poly")
    if not poly:
        return pred
    out = pred.copy()
    for idx, name in enumerate(TASK_COLUMNS):
        cfg = poly.get(name) if isinstance(poly, dict) else None
        if not cfg:
            continue
        coef = np.asarray(cfg.get("coef", []), dtype=np.float64)
        if coef.ndim != 1 or len(coef) < 2:
            continue
        calibrated = np.polyval(coef, out[:, idx])
        blend = float(cfg.get("blend", 1.0))
        out[:, idx] = (1.0 - blend) * out[:, idx] + blend * calibrated
    return out


def _build_posthoc_basis(pred: np.ndarray, terms: list[str]) -> np.ndarray:
    columns = []
    x0 = pred[:, 0]
    x1 = pred[:, 1]
    x2 = pred[:, 2]
    values = {
        "1": np.ones(len(pred), dtype=np.float64),
        "kcat": x0,
        "km": x1,
        "ratio": x2,
        "kcat2": x0 * x0,
        "km2": x1 * x1,
        "ratio2": x2 * x2,
        "kcat_km": x0 * x1,
        "kcat_ratio": x0 * x2,
        "km_ratio": x1 * x2,
    }
    for term in terms:
        if term not in values:
            raise ValueError(f"Unknown posthoc basis term: {term}")
        columns.append(values[term])
    return np.stack(columns, axis=1)


def apply_posthoc_raw_basis(pred: np.ndarray, run_meta: dict) -> np.ndarray:
    """Apply optional valid-fit basis calibration in raw label space."""
    basis_cfg = run_meta.get("kin_posthoc_raw_basis")
    if not basis_cfg:
        return pred
    terms = basis_cfg.get("terms") if isinstance(basis_cfg, dict) else None
    weights = basis_cfg.get("weights") if isinstance(basis_cfg, dict) else None
    blend = basis_cfg.get("blend", 1.0) if isinstance(basis_cfg, dict) else 1.0
    if not terms or not weights:
        return pred
    basis = _build_posthoc_basis(pred, [str(term) for term in terms])
    weight_matrix = np.asarray(weights, dtype=np.float64)
    if weight_matrix.ndim != 2 or weight_matrix.shape[0] != basis.shape[1] or weight_matrix.shape[1] != pred.shape[1]:
        return pred
    calibrated = basis @ weight_matrix
    if isinstance(blend, (list, tuple)):
        blend_array = np.asarray(blend, dtype=np.float64)
        if blend_array.shape != (pred.shape[1],):
            return pred
    else:
        blend_array = np.full((pred.shape[1],), float(blend), dtype=np.float64)
    return (1.0 - blend_array[None, :]) * pred + blend_array[None, :] * calibrated


@torch.no_grad()
def predict(args, payload: dict, run_meta: dict, stats: dict) -> tuple[pd.DataFrame, dict]:
    head_mode = str(run_meta.get("kin_head_mode", args.kin_head_mode))
    dataset = KineticsDataset(
        args.csv,
        args.embedding_dir,
        args.drfp_path,
        args.bank,
        substrate_fp_path=args.substrate_fp_path,
        row_extra_features_path=args.row_extra_features_npz,
        aux_row_extra_features_path=args.aux_row_extra_features_npz,
        source_specs=load_source_specs_json(args.source_specs_json),
        source_column=args.source_column,
        default_source=args.source_name,
        skip_structural_features=head_mode == "row_extra_only",
        target_columns=run_meta.get("kin_target_columns"),
    )
    run_source_names = [str(name) for name in (run_meta.get("source_names") or [])]
    if run_source_names:
        missing_sources = sorted(set(dataset.feature_sources) - set(run_source_names))
        if missing_sources:
            raise ValueError(
                f"Evaluation feature sources {missing_sources} are not present in checkpoint source_names={run_source_names}"
            )
        dataset.source_names = run_source_names
        dataset.source_name_to_id = {name: idx for idx, name in enumerate(run_source_names)}
        dataset.feature_source_ids = [dataset.source_name_to_id[name] for name in dataset.feature_sources]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=kinetics_collate,
        persistent_workers=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    amp_dtype = torch.bfloat16

    classifier_state = payload.get("kinetics_classifier", payload["classifier"])
    dropout = float(run_meta.get("dropout", args.dropout))
    classifier = build_classifier(dropout, classifier_state).to(device)
    missing, unexpected = classifier.load_state_dict(classifier_state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Classifier checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    row_extra_mode = str(run_meta.get("kin_row_extra_mode", "branch"))
    head_branch_dims = None
    if head_mode == "row_extra_only":
        row_extra_dim = int(run_meta.get("row_extra_feature_dim", dataset.row_extra_feature_dim))
        if row_extra_dim <= 0:
            raise ValueError("Checkpoint uses kin_head_mode=row_extra_only but no row-extra features were provided")
        head_branch_dims = {"row_extra": row_extra_dim}
    elif head_mode == "full_feature_mlp":
        head_branch_dims = {
            "full_feature": (
                384
                + 384
                + 384
                + 2 * DEFAULT_EXTRA_SPLIT_DIM
                + 2048
                + int(run_meta.get("substrate_fp_dim", dataset.substrate_fp_dim))
                + int(run_meta.get("row_extra_feature_dim", dataset.row_extra_feature_dim))
            )
        }
    elif head_mode in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
        head_branch_dims = {
            "sample": 384,
            "reaction": 384,
            "protein": DEFAULT_EXTRA_SPLIT_DIM,
            "pocket": DEFAULT_EXTRA_SPLIT_DIM,
            "substrate": int(run_meta.get("substrate_fp_dim", dataset.substrate_fp_dim)),
        }
        row_extra_dim = int(run_meta.get("row_extra_feature_dim", dataset.row_extra_feature_dim))
        if row_extra_dim > 0 and row_extra_mode == "branch":
            head_branch_dims["row_extra"] = row_extra_dim
    row_extra_residual_dim = int(run_meta.get("row_extra_residual_dim", 0))
    stats_path = run_meta.get("kin_full_feature_stats_checkpoint")
    full_feature_mean, full_feature_std = load_full_feature_stats(Path(stats_path)) if stats_path else (None, None)
    heads = KineticsHeads(
        hidden_size=int(run_meta.get("kin_head_hidden_size", args.kin_head_hidden_size)),
        num_layers=int(run_meta.get("kin_head_layers", args.kin_head_layers)),
        dropout=dropout,
        mode=head_mode,
        source_names=list(run_meta.get("source_names") or dataset.source_names),
        source_mode=str(run_meta.get("kin_source_head_mode", "none")),
        branch_dims=head_branch_dims,
        task_branch_weight_map=dict(run_meta.get("task_branch_weight_map") or {}),
        row_extra_residual_dim=row_extra_residual_dim,
        row_extra_residual_mode=str(run_meta.get("kin_row_extra_residual_mode", "plain")),
        aux_row_extra_residual_dim=int(run_meta.get("aux_row_extra_residual_dim", 0)),
        row_extra_network=str(run_meta.get("kin_row_extra_network", "mlp")),
        aux_row_extra_network=str(run_meta.get("kin_aux_row_extra_network", "mlp")),
        aux_row_extra_residual_mode=str(run_meta.get("kin_aux_row_extra_residual_mode", "plain")),
        aux_row_extra_blend_weights=list(run_meta.get("kin_aux_row_extra_blend_weights", [0.0, 0.0, 0.0])),
        extra_dropout=run_meta.get("kin_extra_dropout", None),
        aux_row_extra_hidden_size=int(run_meta.get("kin_aux_row_extra_hidden_size", run_meta.get("kin_head_hidden_size", args.kin_head_hidden_size))),
        aux_row_extra_layers=int(run_meta.get("kin_aux_row_extra_layers", run_meta.get("kin_head_layers", args.kin_head_layers))),
        moe_num_experts=int(run_meta.get("kin_moe_num_experts", 4)),
        moe_residual_num_experts=int(run_meta.get("kin_moe_residual_num_experts", 0)),
        structured_residual_mode=str(run_meta.get("kin_structured_residual_mode", "none")),
        structured_residual_network=str(run_meta.get("kin_structured_residual_network", "mlp")),
        structured_residual_hidden_size=int(
            run_meta.get("kin_structured_residual_hidden_size", run_meta.get("kin_head_hidden_size", args.kin_head_hidden_size))
        ),
        structured_residual_layers=int(
            run_meta.get("kin_structured_residual_layers", run_meta.get("kin_head_layers", args.kin_head_layers))
        ),
        structured_residual_output_weights=list(run_meta.get("kin_structured_residual_output_weights", [0.0, 1.0, 0.0])),
        full_feature_mean=full_feature_mean,
        full_feature_std=full_feature_std,
    ).to(device)
    heads.load_state_dict(payload["kinetics_heads"], strict=False)
    if hasattr(heads, "aux_row_extra_blend_weights"):
        blend = heads.aux_row_extra_blend_weights
        weights = torch.tensor(
            list(run_meta.get("kin_aux_row_extra_blend_weights", [0.0, 0.0, 0.0])),
            device=blend.device,
            dtype=blend.dtype,
        )
        if tuple(weights.shape) == tuple(blend.shape):
            blend.copy_(weights)
    classifier.eval()
    heads.eval()

    ratio_mode = str(run_meta.get("ratio_prediction_mode", args.ratio_prediction_mode))
    pred_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    sample_ids: list[str] = []
    feature_sources: list[str] = []

    for batch in loader:
        batch = move_batch(batch, device)
        batch = apply_drfp_input_policy(batch, zero_drfp=args.zero_drfp_input)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            if getattr(heads, "mode", None) == "row_extra_only":
                out = None
            else:
                out = classifier(
                    batch["s"],
                    batch["z"],
                    batch["is_ligand"],
                    batch["single_mask"],
                    drfp=batch["drfp"],
                    extra_features=batch["extra_features"],
                )
            preds = apply_ratio_prediction_mode(
                forward_kinetics_heads(heads, batch, out),
                stats,
                ratio_mode,
                feature_sources=batch["feature_sources"],
            )
        pred_rows.append(preds.float().cpu().numpy())
        target_rows.append(batch["targets"].float().cpu().numpy())
        mask_rows.append(batch["masks"].float().cpu().numpy())
        sample_ids.extend(batch["sample_ids"])
        feature_sources.extend(batch["feature_sources"])

    pred_norm = np.concatenate(pred_rows, axis=0)
    pred = denormalize(pred_norm, stats, feature_sources=feature_sources if is_source_specific_stats(stats) else None)
    if not args.disable_posthoc_raw:
        pred = apply_posthoc_raw_affine(pred, run_meta)
        pred = apply_posthoc_raw_poly(pred, run_meta)
        pred = apply_posthoc_raw_piecewise(pred, run_meta)
        pred = apply_posthoc_raw_basis(pred, run_meta)
    target = np.concatenate(target_rows, axis=0)
    masks = np.concatenate(mask_rows, axis=0)

    base_df = dataset.df.reset_index(drop=True).copy()
    base_df = base_df.loc[base_df["id"].astype(str).isin(sample_ids)].copy()
    if len(base_df) != len(sample_ids):
        order = {sample_id: idx for idx, sample_id in enumerate(sample_ids)}
        base_df["_eval_order"] = base_df["id"].astype(str).map(order)
        base_df = base_df.sort_values("_eval_order").drop(columns=["_eval_order"]).reset_index(drop=True)

    for idx, name in enumerate(TASK_COLUMNS):
        base_df[f"pred_{name}"] = pred[:, idx]
        base_df[f"target_{name}"] = target[:, idx]
        base_df[f"mask_{name}"] = masks[:, idx]

    metrics: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "csv": str(args.csv),
        "input_rows": int(len(pd.read_csv(args.csv))),
        "evaluated_rows": int(len(base_df)),
        "ratio_prediction_mode": ratio_mode,
        "overall": {},
        "by_dataset": {},
    }
    for idx, name in enumerate(TASK_COLUMNS):
        mask = masks[:, idx] > 0.5
        if mask.sum() >= 3:
            metrics["overall"][name] = regression_metrics(target[mask, idx], pred[mask, idx])

    if "dataset" in base_df.columns:
        for dataset_name, sub_df in base_df.groupby("dataset"):
            sub_metrics = {}
            sub_index = sub_df.index.to_numpy()
            for idx, name in enumerate(TASK_COLUMNS):
                mask = masks[sub_index, idx] > 0.5
                if mask.sum() >= 3:
                    sub_metrics[name] = regression_metrics(target[sub_index[mask], idx], pred[sub_index[mask], idx])
            if sub_metrics:
                metrics["by_dataset"][str(dataset_name)] = sub_metrics

    return base_df, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--drfp_path", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--substrate_fp_path", type=Path, default=None)
    parser.add_argument("--row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--aux_row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--source_specs_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--source_name", type=str, default="default")
    parser.add_argument("--source_column", type=str, default="feature_source")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--kin_head_hidden_size", type=int, default=384)
    parser.add_argument("--kin_head_layers", type=int, default=2)
    parser.add_argument("--kin_head_mode", type=str, default="shared")
    parser.add_argument("--ratio_prediction_mode", type=str, default="independent")
    parser.add_argument("--zero_drfp_input", action="store_true")
    parser.add_argument("--disable_posthoc_raw", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, run_meta, stats = load_checkpoint(args.checkpoint)
    pred_df, metrics = predict(args, payload, run_meta, stats)
    pred_df.to_csv(args.output_dir / "predictions.csv", index=False)
    (args.output_dir / "kinetics_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
