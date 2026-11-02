#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np


def load_bank(path: str):
    data = np.load(path, allow_pickle=True)
    sample_ids = data["sample_ids"].astype(str)
    s_mean = data["s_mean"].astype(np.float32, copy=False)
    z_mean = data["z_mean"].astype(np.float32, copy=False)
    return {
        "sample_ids": sample_ids,
        "s_mean": s_mean,
        "z_mean": z_mean,
        "id_to_idx": {sid: idx for idx, sid in enumerate(sample_ids.tolist())},
    }


def select_context_indices(scores_row, valid_row, threshold, max_selected):
    keep = valid_row & (scores_row >= threshold)
    if not np.any(keep):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    selected = np.flatnonzero(keep)
    selected_scores = scores_row[selected].astype(np.float32, copy=False)
    order = np.argsort(selected_scores)[::-1]
    if max_selected and max_selected > 0:
        order = order[: int(max_selected)]
    selected = selected[order]
    selected_scores = selected_scores[order]
    return selected.astype(np.int64, copy=False), selected_scores.astype(np.float32, copy=False)


def weighted_average(features: np.ndarray, scores: np.ndarray, mode: str):
    if features.shape[0] == 0:
        raise ValueError("features must be non-empty")
    mode = str(mode).lower()
    if mode == "mean" or features.shape[0] == 1:
        return features.mean(axis=0).astype(np.float32, copy=False)
    if mode != "score_weighted":
        raise ValueError(f"Unsupported context_pool_mode: {mode}")
    weights = np.clip(scores.astype(np.float32, copy=False), a_min=0.0, a_max=None)
    denom = float(weights.sum())
    if denom <= 0:
        return features.mean(axis=0).astype(np.float32, copy=False)
    weights = weights / denom
    return (features * weights[:, None]).sum(axis=0).astype(np.float32, copy=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--positive_context_bank_path", type=str, required=True,
                        help="Context bank with positive sample_ids in the same local order as context_indices")
    parser.add_argument("--positive_source_bank_path", type=str, required=True,
                        help="Protein/pocket source bank for positive train samples")
    parser.add_argument("--query_self_bank_path", type=str, required=True,
                        help="Protein/pocket source bank for current split")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--self_weight", type=float, default=0.9)
    parser.add_argument("--score_threshold", type=float, default=0.5)
    parser.add_argument("--max_selected", type=int, default=0,
                        help="0 means no cap after thresholding")
    parser.add_argument("--context_pool_mode", type=str, default="mean", choices=["mean", "score_weighted"])
    parser.add_argument("--fallback_mode", type=str, default="self_only", choices=["self_only", "top1"])
    args = parser.parse_args()

    cache = np.load(args.cache_path, allow_pickle=True)
    query_ids = cache["sample_ids"].astype(str)
    context_indices = cache["context_indices"].astype(np.int32, copy=False)
    context_mask = cache["context_mask"].astype(bool, copy=False)
    pair_corr_scores = cache["pair_corr_scores"].astype(np.float32, copy=False)

    positive_context_bank = np.load(args.positive_context_bank_path, allow_pickle=True)
    positive_context_ids = positive_context_bank["sample_ids"].astype(str)

    positive_source_bank = load_bank(args.positive_source_bank_path)
    query_self_bank = load_bank(args.query_self_bank_path)

    positive_s = np.zeros((positive_context_ids.shape[0], positive_source_bank["s_mean"].shape[1]), dtype=np.float32)
    positive_z = np.zeros((positive_context_ids.shape[0], positive_source_bank["z_mean"].shape[1]), dtype=np.float32)
    for idx, sample_id in enumerate(positive_context_ids.tolist()):
        src_idx = positive_source_bank["id_to_idx"].get(sample_id)
        if src_idx is None:
            raise KeyError(f"Positive sample_id={sample_id} not found in source bank {args.positive_source_bank_path}")
        positive_s[idx] = positive_source_bank["s_mean"][src_idx]
        positive_z[idx] = positive_source_bank["z_mean"][src_idx]

    out_s = np.zeros((query_ids.shape[0], query_self_bank["s_mean"].shape[1]), dtype=np.float32)
    out_z = np.zeros((query_ids.shape[0], query_self_bank["z_mean"].shape[1]), dtype=np.float32)
    selected_count = np.zeros((query_ids.shape[0],), dtype=np.int32)
    selected_score_mean = np.zeros((query_ids.shape[0],), dtype=np.float32)
    selected_score_max = np.zeros((query_ids.shape[0],), dtype=np.float32)
    self_only_count = 0

    for row_idx, sample_id in enumerate(query_ids.tolist()):
        self_idx = query_self_bank["id_to_idx"].get(sample_id)
        if self_idx is None:
            raise KeyError(f"Query sample_id={sample_id} not found in self bank {args.query_self_bank_path}")
        self_s = query_self_bank["s_mean"][self_idx]
        self_z = query_self_bank["z_mean"][self_idx]

        selected_cols, selected_scores = select_context_indices(
            pair_corr_scores[row_idx],
            context_mask[row_idx] & (context_indices[row_idx] >= 0),
            threshold=float(args.score_threshold),
            max_selected=int(args.max_selected),
        )

        if selected_cols.size == 0 and args.fallback_mode == "top1":
            valid = context_mask[row_idx] & (context_indices[row_idx] >= 0)
            if np.any(valid):
                valid_cols = np.flatnonzero(valid)
                best_col = valid_cols[np.argmax(pair_corr_scores[row_idx, valid_cols])]
                selected_cols = np.asarray([best_col], dtype=np.int64)
                selected_scores = np.asarray([pair_corr_scores[row_idx, best_col]], dtype=np.float32)

        if selected_cols.size == 0:
            out_s[row_idx] = self_s
            out_z[row_idx] = self_z
            self_only_count += 1
            continue

        local_indices = context_indices[row_idx, selected_cols].astype(np.int64, copy=False)
        ctx_s = weighted_average(positive_s[local_indices], selected_scores, args.context_pool_mode)
        ctx_z = weighted_average(positive_z[local_indices], selected_scores, args.context_pool_mode)
        alpha = float(args.self_weight)
        out_s[row_idx] = alpha * self_s + (1.0 - alpha) * ctx_s
        out_z[row_idx] = alpha * self_z + (1.0 - alpha) * ctx_z
        selected_count[row_idx] = int(selected_cols.size)
        selected_score_mean[row_idx] = float(selected_scores.mean())
        selected_score_max[row_idx] = float(selected_scores.max())

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        sample_ids=query_ids,
        s_mean=out_s.astype(np.float16, copy=False),
        z_mean=out_z.astype(np.float16, copy=False),
        selected_context_count=selected_count,
        selected_score_mean=selected_score_mean,
        selected_score_max=selected_score_max,
    )

    meta = {
        "cache_path": str(args.cache_path),
        "positive_context_bank_path": str(args.positive_context_bank_path),
        "positive_source_bank_path": str(args.positive_source_bank_path),
        "query_self_bank_path": str(args.query_self_bank_path),
        "self_weight": float(args.self_weight),
        "score_threshold": float(args.score_threshold),
        "max_selected": int(args.max_selected),
        "context_pool_mode": str(args.context_pool_mode),
        "fallback_mode": str(args.fallback_mode),
        "num_samples": int(query_ids.shape[0]),
        "s_dim": int(out_s.shape[1]),
        "z_dim": int(out_z.shape[1]),
        "self_only_count": int(self_only_count),
        "self_only_ratio": float(self_only_count / max(1, query_ids.shape[0])),
        "mean_selected_context_count": float(selected_count.mean()),
        "median_selected_context_count": float(np.median(selected_count)),
        "p90_selected_context_count": float(np.quantile(selected_count, 0.9)),
    }
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
