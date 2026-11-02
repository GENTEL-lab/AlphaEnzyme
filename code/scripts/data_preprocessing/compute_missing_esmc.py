#!/usr/bin/env python
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def seq_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]


def to_residue_node_feature(node_feature: np.ndarray, sequence: str) -> np.ndarray:
    """Convert ESM-C token embeddings to residue embeddings."""
    if node_feature.shape[0] == len(sequence) + 2:
        return node_feature[1:-1]
    return node_feature


def load_existing_sequences(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    with p.open("rb") as f:
        data = pickle.load(f)
    return set(map(str, data.keys()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_csv", required=True)
    parser.add_argument("--existing_seq2feature", default=None)
    parser.add_argument("--node_dir", required=True)
    parser.add_argument("--partial_seq2feature", required=True)
    parser.add_argument("--mapping_csv", required=True)
    parser.add_argument("--failed_csv", required=True)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_name", default="esmc_600m")
    parser.add_argument(
        "--ignore_existing_seq2feature",
        action="store_true",
        help="Compute/save sequence-hash node features even if protein-level seq2feature already has this sequence.",
    )
    args = parser.parse_args()

    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    node_dir = Path(args.node_dir)
    node_dir.mkdir(parents=True, exist_ok=True)
    Path(args.partial_seq2feature).parent.mkdir(parents=True, exist_ok=True)
    Path(args.mapping_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.failed_csv).parent.mkdir(parents=True, exist_ok=True)

    tasks = pd.read_csv(args.task_csv)
    tasks["sequence"] = tasks["sequence"].astype(str)
    seqs = (
        tasks[["sequence"]]
        .drop_duplicates()
        .sort_values("sequence")
        .reset_index(drop=True)
    )
    seqs.insert(0, "sequence_idx", range(len(seqs)))
    if args.end_idx == -1:
        args.end_idx = len(seqs)
    seqs = seqs.iloc[args.start_idx : args.end_idx].copy()

    existing_sequences = load_existing_sequences(args.existing_seq2feature)
    print(
        json.dumps(
            {
                "task_csv": args.task_csv,
                "total_unique_sequences": int(tasks["sequence"].nunique()),
                "start_idx": args.start_idx,
                "end_idx": args.end_idx,
                "num_sequences_this_shard": int(len(seqs)),
                "existing_sequences": int(len(existing_sequences)),
                "node_dir": str(node_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"Loading {args.model_name} on {args.device}...")
    model = ESMC.from_pretrained(args.model_name).to(args.device)
    model.eval()

    seq_to_feature = {}
    mapping_rows = []
    failed_rows = []

    with torch.no_grad():
        for row in tqdm(seqs.itertuples(index=False), total=len(seqs), desc="ESM-C"):
            seq = row.sequence
            h = seq_hash(seq)
            save_path = node_dir / f"{h}.npz"

            if seq in existing_sequences and not args.ignore_existing_seq2feature:
                continue

            if save_path.exists():
                try:
                    node_feature = np.load(save_path)["node_feature"]
                    node_feature = to_residue_node_feature(node_feature, seq)
                    seq_to_feature[seq] = node_feature.mean(axis=0).astype(np.float32)
                    mapping_rows.append(
                        {
                            "sequence_idx": int(row.sequence_idx),
                            "sequence_hash": h,
                            "sequence": seq,
                            "node_path": str(save_path),
                            "status": "exists",
                        }
                    )
                    continue
                except Exception:
                    pass

            try:
                protein = ESMProtein(sequence=seq)
                protein_tensor = model.encode(protein)
                logits_output = model.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                node_feature = logits_output.embeddings[0].detach().cpu().numpy().astype(np.float32)
                node_feature = to_residue_node_feature(node_feature, seq)
                np.savez_compressed(save_path, node_feature=node_feature)
                seq_to_feature[seq] = node_feature.mean(axis=0).astype(np.float32)
                mapping_rows.append(
                    {
                        "sequence_idx": int(row.sequence_idx),
                        "sequence_hash": h,
                        "sequence": seq,
                        "node_path": str(save_path),
                        "status": "computed",
                    }
                )
            except Exception as exc:
                failed_rows.append(
                    {
                        "sequence_idx": int(row.sequence_idx),
                        "sequence_hash": h,
                        "sequence": seq,
                        "error": repr(exc),
                    }
                )

    with Path(args.partial_seq2feature).open("wb") as f:
        pickle.dump(seq_to_feature, f)
    pd.DataFrame(mapping_rows).to_csv(args.mapping_csv, index=False)
    pd.DataFrame(failed_rows).to_csv(args.failed_csv, index=False)
    print(
        json.dumps(
            {
                "computed_or_loaded": int(len(seq_to_feature)),
                "failed": int(len(failed_rows)),
                "partial_seq2feature": args.partial_seq2feature,
                "mapping_csv": args.mapping_csv,
                "failed_csv": args.failed_csv,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
