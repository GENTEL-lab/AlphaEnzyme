#!/usr/bin/env python
"""Precompute frozen AlphaEnzyme alpha_repr batches for fast specificity-head training."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = ROOT / "code"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

from alphaenzyme.data.pairwise_dataset import PairwiseReactionDataset  # noqa: E402
from evaluate_ranking_single import load_model  # noqa: E402


def move_batch(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def classifier_forward(model: torch.nn.Module, batch: dict) -> dict:
    return model(
        batch["s"],
        batch["z"],
        batch["is_ligand"],
        batch["single_mask"],
        drfp=batch.get("drfp"),
        extra_features=batch.get("extra_features"),
        context_features=batch.get("context_features"),
        context_mask=batch.get("context_mask"),
        context_scores=batch.get("context_scores"),
        overall_corr=batch.get("overall_corr"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--json_dir", type=Path, required=True)
    parser.add_argument("--embedding_dir", type=Path, required=True)
    parser.add_argument("--csv_path", type=Path, required=True)
    parser.add_argument("--drfp_path", type=Path, required=True)
    parser.add_argument("--external_bank", type=Path, required=True)
    parser.add_argument("--metadata_cache_path", type=Path, default=None)
    parser.add_argument("--batch_cache_path", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--max_seq_length", type=int, default=700)
    parser.add_argument("--batching_strategy", type=str, default="reaction_sampled")
    parser.add_argument("--sampled_batches_per_reaction", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--start_batch", type=int, default=0)
    parser.add_argument("--end_batch", type=int, default=0)
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=0,
        help="If >0, write part files every N reaction batches instead of one large cache.",
    )
    return parser.parse_args()


def save_payload(path: Path, payload: dict) -> None:
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def build_payload(args: argparse.Namespace, batches: list[dict], start_batch: int, end_batch: int, part_index: int | None = None) -> dict:
    meta = {
        "checkpoint": str(args.checkpoint),
        "json_dir": str(args.json_dir),
        "embedding_dir": str(args.embedding_dir),
        "csv_path": str(args.csv_path),
        "drfp_path": str(args.drfp_path),
        "external_bank": str(args.external_bank),
        "dtype": args.dtype,
        "alpha_repr_dim": 1152,
        "start_batch": int(start_batch),
        "end_batch": int(end_batch),
        "n_batches": int(end_batch - start_batch),
        "n_samples": int(sum(int(b["labels"].numel()) for b in batches)),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if part_index is not None:
        meta["part_index"] = int(part_index)
    return {"meta": meta, "batches": batches}


def part_path_for(output_path: Path, part_index: int) -> Path:
    return output_path.parent / f"{output_path.stem}.part{part_index:04d}{output_path.suffix}"


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    drfp_data = None
    if args.drfp_path.exists():
        with args.drfp_path.open("rb") as handle:
            drfp_data = pickle.load(handle)

    dataset = PairwiseReactionDataset(
        json_dir=str(args.json_dir),
        embedding_dir=str(args.embedding_dir),
        csv_path=str(args.csv_path),
        drfp_path=str(args.drfp_path),
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        batching_strategy=args.batching_strategy,
        sampled_batches_per_reaction=args.sampled_batches_per_reaction,
        random_seed=42,
        external_feature_banks=[
            {"path": str(args.external_bank), "include_z": True, "name": "protein_pocket"}
        ],
        metadata_cache_path=str(args.metadata_cache_path) if args.metadata_cache_path else None,
        batch_cache_path=str(args.batch_cache_path) if args.batch_cache_path else None,
    )

    model = load_model(
        str(args.checkpoint),
        str(device),
        use_drfp=True,
        compressed=True,
        embedding_dir=str(args.embedding_dir),
        resolved_config=yaml.safe_load((ROOT/"configs/specificity_model.yaml").read_text()),
    )
    model.eval()

    start_batch = max(0, int(args.start_batch))
    end_batch = int(args.end_batch) if args.end_batch > 0 else len(dataset)
    end_batch = min(len(dataset), end_batch)
    if args.max_batches > 0:
        end_batch = min(end_batch, start_batch + args.max_batches)
    if start_batch >= end_batch:
        raise ValueError(f"empty batch range: start={start_batch}, end={end_batch}, len={len(dataset)}")
    n_batches = end_batch - start_batch
    batches = []
    chunk_paths = []
    chunk_start = start_batch
    part_index = 0
    out_dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    with torch.no_grad():
        for idx in tqdm(range(start_batch, end_batch), desc="precompute alpha_repr"):
            batch = move_batch(dataset[idx], device)
            out = classifier_forward(model, batch)
            alpha_repr = out["alpha_repr"].detach().cpu().to(out_dtype)
            teacher_logits = out["logits"].detach().cpu().to(torch.float32)
            labels = batch["labels"].detach().cpu().to(torch.float32)
            batches.append(
                {
                    "alpha_repr": alpha_repr,
                    "labels": labels,
                    "teacher_logits": teacher_logits,
                    "reaction": str(batch.get("reaction", "")),
                    "sample_ids": list(batch.get("sample_ids", [])),
                }
            )
            if args.chunk_size > 0 and len(batches) >= args.chunk_size:
                current_end = idx + 1
                payload = build_payload(args, batches, chunk_start, current_end, part_index=part_index)
                path = part_path_for(args.output_path, part_index)
                save_payload(path, payload)
                chunk_paths.append(str(path))
                print(json.dumps(payload["meta"], ensure_ascii=False), flush=True)
                batches = []
                chunk_start = current_end
                part_index += 1

    if args.chunk_size > 0:
        if batches:
            payload = build_payload(args, batches, chunk_start, end_batch, part_index=part_index)
            path = part_path_for(args.output_path, part_index)
            save_payload(path, payload)
            chunk_paths.append(str(path))
            print(json.dumps(payload["meta"], ensure_ascii=False), flush=True)
        manifest = {
            "checkpoint": str(args.checkpoint),
            "output_path": str(args.output_path),
            "chunk_size": int(args.chunk_size),
            "start_batch": int(start_batch),
            "end_batch": int(end_batch),
            "n_batches": int(n_batches),
            "parts": chunk_paths,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest_path = args.output_path.with_suffix(args.output_path.suffix + ".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        payload = build_payload(args, batches, start_batch, end_batch)
        save_payload(args.output_path, payload)
        print(json.dumps(payload["meta"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
