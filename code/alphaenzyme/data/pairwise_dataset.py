"""
Pairwise Reaction Dataset - 按 reaction 分组 batch 的 Dataset
用于 Pairwise Ranking Loss 训练
"""

import os
import sys
import json
import glob
import pickle
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from alphaenzyme.data.direct_feature_support import DirectContextAssetStore
from alphaenzyme.data.external_feature_bank import load_external_feature_bank, normalize_bank_specs


def _to_f32(arr):
    """Convert bf16/uint16 to float32."""
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr).view(torch.float16).float()
    return torch.from_numpy(arr.astype(np.float32))


def _normalize_ec_string(value):
    value = str(value).strip()
    if not value or value.lower() == "nan":
        return ""
    return value


def _sample_id_to_tuple(sample_id: str):
    parts = str(sample_id).split("_")
    if len(parts) < 2:
        return None
    return parts[0], ".".join(parts[1:])


class PairwiseReactionDataset(Dataset):
    """按 reaction 分组的 Dataset。

    每个 batch 属于同一个 reaction，包含 1-N 个正样本和若干负样本，
    总样本数 ≤ batch_size。多个正样本可以合并到一个 batch。

    数据遍历：
    1. 按 CANO_RXN_SMILES 分组所有样本
    2. 每个 reaction 内打乱样本，按 batch_size 切分为多个 batch
    3. 打乱所有 batch 的顺序
    4. 每个 epoch 返回所有 batch
    """

    def __init__(
        self,
        json_dir,
        embedding_dir,
        csv_path=None,
        drfp_path=None,
        max_seq_length=None,
        batch_size=256,
        batching_strategy="balanced_chunks",
        sampled_batches_per_reaction=1,
        min_positive_fraction=0.15,
        max_positive_fraction=0.70,
        random_seed=42,
        num_workers=4,
        pin_memory=True,
        external_feature_banks=None,
        external_missing_policy="error",
        metadata_cache_path=None,
        batch_cache_path=None,
        context_cache_path=None,
        context_bank_path=None,
        context_missing_policy="error",
    ):
        self.json_dir = json_dir
        self.embedding_dir = embedding_dir
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.batching_strategy = str(batching_strategy).lower()
        self.sampled_batches_per_reaction = max(1, int(sampled_batches_per_reaction))
        self.min_positive_fraction = float(min_positive_fraction)
        self.max_positive_fraction = float(max_positive_fraction)
        self.random_seed = int(random_seed)
        self._rng = random.Random(self.random_seed)
        if self.max_positive_fraction < self.min_positive_fraction:
            self.max_positive_fraction = self.min_positive_fraction
        self.external_features = None
        self.external_banks = []
        self.external_feature_dim = 0
        self.external_feature_meta = []
        self.external_missing_policy = external_missing_policy
        self.context_store = None
        self.context_len = 0
        self.context_dim = 0
        self.metadata_cache_path = metadata_cache_path
        self.batch_cache_path = batch_cache_path

        # 加载 DRFP 数据
        self.drfp_data = {}
        self.rxn_mapping = {}
        self.rxn_mapping_by_tuple = {}
        if drfp_path and os.path.exists(drfp_path):
            with open(drfp_path, 'rb') as f:
                self.drfp_data = pickle.load(f)
            print(f"Loaded {len(self.drfp_data)} DRFP features from {drfp_path}")

            if csv_path and os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    sample_id = str(row['id']) if 'id' in row.index else str(idx)
                    if pd.notna(row.get('CANO_RXN_SMILES')):
                        rxn = str(row['CANO_RXN_SMILES'])
                        self.rxn_mapping[sample_id] = rxn
                        uniprot_id = str(row.get('UniprotID', '')).strip()
                        ec_number = _normalize_ec_string(row.get('EC number', ''))
                        if uniprot_id and ec_number:
                            self.rxn_mapping_by_tuple[(uniprot_id, ec_number)] = rxn
                print(f"Loaded {len(self.rxn_mapping)} reaction mappings from {csv_path}")
            else:
                print("No CSV mapping provided, will read CANO_RXN_SMILES directly from JSON")

        self.reaction_groups = self._load_or_build_reaction_groups()
        bank_specs = normalize_bank_specs(external_feature_banks)
        if bank_specs:
            self.external_banks = [
                load_external_feature_bank(
                    spec["path"],
                    include_z=bool(spec.get("include_z", False)),
                    feature_keys=spec.get("keys"),
                )
                for spec in bank_specs
            ]
            self.external_feature_meta = [
                {
                    "name": spec["name"],
                    "path": spec["path"],
                    "include_z": bool(spec.get("include_z", False)),
                    "keys": spec.get("keys"),
                    "dim": int(bank.feature_dim),
                }
                for spec, bank in zip(bank_specs, self.external_banks)
            ]
            self.external_feature_dim = int(sum(bank.feature_dim for bank in self.external_banks))
            print(
                f"Loaded external feature banks lazily: dim={self.external_feature_dim}, "
                f"banks={[m.get('name') for m in self.external_feature_meta]}"
            )
        if context_cache_path and context_bank_path:
            self.context_store = DirectContextAssetStore(
                cache_path=context_cache_path,
                bank_path=context_bank_path,
                missing_policy=context_missing_policy,
            )
            self.context_len = int(self.context_store.context_len)
            self.context_dim = int(self.context_store.context_dim)
            print(
                f"Loaded direct context assets: len={self.context_len}, "
                f"dim={self.context_dim}"
            )
        self.batches = []
        self.batch_metadata = []
        self._rebuild_batches()

    def _batch_cache_meta(self):
        return {
            "json_dir": os.path.abspath(self.json_dir),
            "embedding_dir": os.path.abspath(self.embedding_dir),
            "max_seq_length": self.max_seq_length,
            "batch_size": int(self.batch_size),
            "batching_strategy": str(self.batching_strategy),
            "sampled_batches_per_reaction": int(self.sampled_batches_per_reaction),
            "min_positive_fraction": float(self.min_positive_fraction),
            "max_positive_fraction": float(self.max_positive_fraction),
            "reaction_count": int(len(self.reaction_groups)),
            "sample_count": int(sum(len(v) for v in self.reaction_groups.values())),
        }

    def _load_batch_cache(self):
        if not self.batch_cache_path or not os.path.exists(self.batch_cache_path):
            return False
        try:
            with open(self.batch_cache_path, "rb") as handle:
                payload = pickle.load(handle)
            if payload.get("meta") != self._batch_cache_meta():
                return False
            self.batches = payload.get("batches", [])
            self.batch_metadata = payload.get("batch_metadata", [])
            if self.batches and self.batch_metadata:
                print(
                    f"Loaded replay batch cache from {self.batch_cache_path}: "
                    f"{len(self.batches)} batches"
                )
                return True
        except Exception as exc:
            print(f"Warning: failed to load replay batch cache {self.batch_cache_path}: {exc}")
        return False

    def _save_batch_cache(self):
        if not self.batch_cache_path:
            return
        try:
            cache_dir = os.path.dirname(self.batch_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            payload = {
                "meta": self._batch_cache_meta(),
                "batches": self.batches,
                "batch_metadata": self.batch_metadata,
            }
            tmp_path = f"{self.batch_cache_path}.tmp.{os.getpid()}"
            with open(tmp_path, "wb") as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, self.batch_cache_path)
            print(f"Saved replay batch cache to {self.batch_cache_path}")
        except Exception as exc:
            print(f"Warning: failed to save replay batch cache {self.batch_cache_path}: {exc}")

    def _metadata_cache_signature(self):
        return {
            "json_dir": os.path.abspath(self.json_dir),
            "embedding_dir": os.path.abspath(self.embedding_dir),
            "csv_path": os.path.abspath(self.rxn_mapping and next(iter(self.rxn_mapping.keys()), "") and "" or ""),
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
        }

    def _load_or_build_reaction_groups(self):
        cache_path = self.metadata_cache_path
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as handle:
                    payload = pickle.load(handle)
                meta = payload.get("meta", {})
                if (
                    meta.get("json_dir") == os.path.abspath(self.json_dir)
                    and meta.get("embedding_dir") == os.path.abspath(self.embedding_dir)
                    and meta.get("max_seq_length") == self.max_seq_length
                ):
                    reaction_groups = payload.get("reaction_groups", {})
                    print(
                        f"Loaded replay metadata cache from {cache_path}: "
                        f"{sum(len(v) for v in reaction_groups.values())} samples, "
                        f"{len(reaction_groups)} reactions"
                    )
                    return reaction_groups
            except Exception as exc:
                print(f"Warning: failed to load replay metadata cache {cache_path}: {exc}")

        reaction_groups = self._build_reaction_groups()
        if cache_path:
            try:
                cache_dir = os.path.dirname(cache_path)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                payload = {
                    "meta": {
                        "json_dir": os.path.abspath(self.json_dir),
                        "embedding_dir": os.path.abspath(self.embedding_dir),
                        "max_seq_length": self.max_seq_length,
                    },
                    "reaction_groups": reaction_groups,
                }
                with open(cache_path, "wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Saved replay metadata cache to {cache_path}")
            except Exception as exc:
                print(f"Warning: failed to save replay metadata cache {cache_path}: {exc}")
        return reaction_groups

    def _build_reaction_groups(self):
        json_files = sorted(glob.glob(os.path.join(self.json_dir, "*.json")))
        print(f"Found {len(json_files)} JSON files in {self.json_dir}")

        emb_files = glob.glob(os.path.join(self.embedding_dir, "*.npz"))
        available_emb_ids = set(os.path.splitext(os.path.basename(f))[0] for f in emb_files)
        print(f"Found {len(available_emb_ids)} embedding files")

        reaction_groups = defaultdict(list)
        skipped_by_length = 0
        skipped_no_emb = 0

        for json_path in json_files:
            sample_id = os.path.splitext(os.path.basename(json_path))[0]
            if sample_id not in available_emb_ids:
                skipped_no_emb += 1
                continue

            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                if self.max_seq_length is not None:
                    if 'sequences' in data and len(data['sequences']) > 0:
                        protein_seq = data['sequences'][0].get('proteinChain', {}).get('sequence', '')
                        if len(protein_seq) > self.max_seq_length:
                            skipped_by_length += 1
                            continue

                label = data.get('label', 0)

                if self.rxn_mapping:
                    rxn_smiles = self.rxn_mapping.get(sample_id, '')
                    if not rxn_smiles:
                        uniprot_id = str(data.get('uniprot_id', '')).strip()
                        ec_number = _normalize_ec_string(data.get('ec_number', ''))
                        if uniprot_id and ec_number:
                            rxn_smiles = self.rxn_mapping_by_tuple.get((uniprot_id, ec_number), '')
                    if not rxn_smiles:
                        sample_tuple = _sample_id_to_tuple(sample_id)
                        if sample_tuple is not None:
                            rxn_smiles = self.rxn_mapping_by_tuple.get(sample_tuple, '')
                else:
                    rxn_smiles = data.get('cano_rxn_smiles', '')

                if not rxn_smiles:
                    continue

                reaction_groups[rxn_smiles].append({
                    'sample_id': sample_id,
                    'json_path': json_path,
                    'label': label,
                })
            except Exception as e:
                print(f"Warning: Failed to load {json_path}: {e}")
                continue

        print(f"Loaded {sum(len(v) for v in reaction_groups.values())} samples "
              f"in {len(reaction_groups)} reactions")
        if skipped_no_emb > 0:
            print(f"Skipped {skipped_no_emb} samples (no embedding)")
        if skipped_by_length > 0:
            print(f"Skipped {skipped_by_length} samples (sequence too long)")
        return dict(reaction_groups)

    def _add_batch(self, rxn, batch_samples, batch_idx, tail_batch=False, sampled_batch=False):
        self._rng.shuffle(batch_samples)
        self.batches.append((rxn, batch_samples))
        self.batch_metadata.append({
            'reaction': rxn,
            'n_pos': sum(1 for s in batch_samples if s['label'] == 1),
            'n_neg': sum(1 for s in batch_samples if s['label'] == 0),
            'n_total': len(batch_samples),
            'batch_idx': batch_idx,
            'tail_batch': tail_batch,
            'sampled_batch': sampled_batch,
        })

    def _build_balanced_chunks_for_reaction(self, rxn, samples):
        pos_samples = [s for s in samples if s['label'] == 1]
        neg_samples = [s for s in samples if s['label'] == 0]
        if not pos_samples or not neg_samples:
            return

        i = 0
        j = 0
        batch_idx = 0
        while i < len(pos_samples) and j < len(neg_samples):
            remaining_pos = len(pos_samples) - i
            remaining_neg = len(neg_samples) - j
            remaining_total = remaining_pos + remaining_neg
            if remaining_total == 0:
                break

            size = min(self.batch_size, remaining_total)
            n_pos = min(remaining_pos, size // 2)
            n_neg = min(remaining_neg, size - n_pos)
            if n_pos == 0 and remaining_pos > 0:
                n_pos = 1
                n_neg = min(remaining_neg, size - 1)

            batch_samples = []
            batch_samples.extend(pos_samples[i:i + n_pos])
            batch_samples.extend(neg_samples[j:j + n_neg])
            i += n_pos
            j += n_neg
            self._add_batch(rxn, batch_samples, batch_idx=batch_idx)
            batch_idx += 1

        while j < len(neg_samples):
            remaining_neg = len(neg_samples) - j
            n_neg = min(max(self.batch_size - 1, 1), remaining_neg)
            n_pos = min(max(1, self.batch_size - n_neg), len(pos_samples))
            batch_samples = []
            batch_samples.extend(self._rng.sample(pos_samples, k=n_pos))
            batch_samples.extend(neg_samples[j:j + n_neg])
            j += n_neg
            self._add_batch(rxn, batch_samples, batch_idx=batch_idx, tail_batch=True)
            batch_idx += 1

        while i < len(pos_samples):
            remaining_pos = len(pos_samples) - i
            n_pos = min(max(self.batch_size - 1, 1), remaining_pos)
            n_neg = min(max(1, self.batch_size - n_pos), len(neg_samples))
            batch_samples = []
            batch_samples.extend(pos_samples[i:i + n_pos])
            batch_samples.extend(self._rng.sample(neg_samples, k=n_neg))
            i += n_pos
            self._add_batch(rxn, batch_samples, batch_idx=batch_idx, tail_batch=True)
            batch_idx += 1

    def _build_reaction_sampled_batches_for_reaction(self, rxn, samples):
        pos_samples = [s for s in samples if s['label'] == 1]
        neg_samples = [s for s in samples if s['label'] == 0]
        if not pos_samples or not neg_samples:
            return

        total_count = len(samples)
        if total_count <= self.batch_size:
            self._add_batch(rxn, list(samples), batch_idx=0, sampled_batch=False)
            return

        pos_ratio = len(pos_samples) / total_count
        target_pos_ratio = min(max(pos_ratio, self.min_positive_fraction), self.max_positive_fraction)

        for batch_idx in range(self.sampled_batches_per_reaction):
            target_pos = int(round(self.batch_size * target_pos_ratio))
            target_pos = max(1, min(len(pos_samples), target_pos))
            target_neg = max(1, self.batch_size - target_pos)

            if target_neg > len(neg_samples):
                deficit = target_neg - len(neg_samples)
                target_neg = len(neg_samples)
                target_pos = min(len(pos_samples), target_pos + deficit)
            if target_pos > len(pos_samples):
                deficit = target_pos - len(pos_samples)
                target_pos = len(pos_samples)
                target_neg = min(len(neg_samples), target_neg + deficit)
            if target_pos <= 0 or target_neg <= 0:
                continue

            batch_samples = []
            if target_pos < len(pos_samples):
                batch_samples.extend(self._rng.sample(pos_samples, k=target_pos))
            else:
                batch_samples.extend(pos_samples)
            if target_neg < len(neg_samples):
                batch_samples.extend(self._rng.sample(neg_samples, k=target_neg))
            else:
                batch_samples.extend(neg_samples)
            self._add_batch(rxn, batch_samples, batch_idx=batch_idx, sampled_batch=True)

    def _print_batch_summary(self):
        print(f"Constructed {len(self.batches)} batches")
        if self.batch_metadata:
            total_pos = sum(m['n_pos'] for m in self.batch_metadata)
            total_neg = sum(m['n_neg'] for m in self.batch_metadata)
            avg_batch_size = np.mean([m['n_total'] for m in self.batch_metadata])
            tail_batches = sum(1 for m in self.batch_metadata if m.get('tail_batch', False))
            sampled_batches = sum(1 for m in self.batch_metadata if m.get('sampled_batch', False))
            print(f"  Batching strategy: {self.batching_strategy}")
            print(f"  Total pos samples: {total_pos}, Total neg samples: {total_neg}")
            print(f"  Average batch size: {avg_batch_size:.1f}")
            print(f"  Tail batches (with resampled minority class): {tail_batches}")
            if sampled_batches > 0:
                print(f"  Sampled large-reaction batches: {sampled_batches}")

    def _rebuild_batches(self):
        self.batches = []
        self.batch_metadata = []
        if self._load_batch_cache():
            self._print_batch_summary()
            return

        reaction_ids = list(self.reaction_groups.keys())
        self._rng.shuffle(reaction_ids)
        for rxn in reaction_ids:
            samples = list(self.reaction_groups[rxn])
            self._rng.shuffle(samples)
            if self.batching_strategy == "reaction_sampled":
                self._build_reaction_sampled_batches_for_reaction(rxn, samples)
            else:
                self._build_balanced_chunks_for_reaction(rxn, samples)

        self._print_batch_summary()
        self._save_batch_cache()

    def shuffle_batches(self):
        """打乱所有 batch 的顺序。每个 epoch 开始时调用。"""
        if self.batching_strategy == "reaction_sampled" and not self.batch_cache_path:
            self._rebuild_batches()
            return
        perm = list(range(len(self.batches)))
        self._rng.shuffle(perm)
        self.batches = [self.batches[i] for i in perm]
        self.batch_metadata = [self.batch_metadata[i] for i in perm]

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        rxn, batch_samples = self.batches[idx]

        s_list = []
        z_list = []
        is_ligand_list = []
        single_mask_list = []
        labels = []
        sample_ids = []
        context_features = []
        context_masks = []
        context_scores = []
        overall_corrs = []

        # DRFP 扩展到 [B, 2048]（同一个 reaction 的所有样本共享 DRFP）
        for sample in batch_samples:
            sample_id = sample['sample_id']
            emb_path = os.path.join(self.embedding_dir, f"{sample_id}.npz")

            try:
                emb_data = np.load(emb_path)
            except Exception as e:
                print(f"Warning: Failed to load {emb_path}: {e}")
                # 返回零向量
                # 需要推断维度，先尝试加载一个样本
                continue

            s_list.append(_to_f32(emb_data['s']))
            z_list.append(_to_f32(emb_data['z']))
            is_ligand_list.append(torch.from_numpy(emb_data['is_ligand'].astype(np.float32)))
            single_mask_list.append(torch.from_numpy(emb_data['single_mask'].astype(np.float32)))
            labels.append(sample['label'])
            sample_ids.append(sample_id)
            if self.context_store is not None:
                ctx_feat, ctx_mask, ctx_score, overall_corr = self.context_store.get(sample_id)
                context_features.append(ctx_feat)
                context_masks.append(ctx_mask)
                context_scores.append(ctx_score)
                overall_corrs.append(overall_corr)

        if not s_list or len(set(labels)) < 2:
            next_idx = (idx + 1) % len(self.batches)
            print(f"Warning: Invalid pairwise batch at idx={idx}, falling back to idx={next_idx}")
            return self.__getitem__(next_idx)

        # DRFP 扩展到 [B, 2048]（同一个 reaction 的所有样本共享 DRFP）
        if self.drfp_data and rxn in self.drfp_data:
            drfp_np = self.drfp_data[rxn]
            drfp = torch.from_numpy(drfp_np).float()  # [2048]
            drfp = drfp.unsqueeze(0).expand(len(labels), -1)  # [B, 2048]
        else:
            drfp = torch.zeros(len(labels), 2048, dtype=torch.float32)

        if self.external_banks:
            batch_extra_features = []
            for sample_id in sample_ids:
                parts = []
                for bank in self.external_banks:
                    feat = bank.get(str(sample_id))
                    if feat is None:
                        if self.external_missing_policy == "error":
                            raise KeyError(f"sample_id={sample_id} missing in external bank: {bank.path}")
                        feat = np.zeros((bank.feature_dim,), dtype=np.float32)
                    parts.append(np.asarray(feat, dtype=np.float32))
                batch_extra_features.append(np.concatenate(parts, axis=-1).astype(np.float32, copy=False))
            extra_features = torch.from_numpy(np.stack(batch_extra_features, axis=0)).float()
        else:
            extra_features = torch.zeros(len(labels), 0, dtype=torch.float32)
        if self.context_store is not None:
            batch_context_features = torch.stack(context_features)
            batch_context_mask = torch.stack(context_masks)
            batch_context_scores = torch.stack(context_scores)
            batch_overall_corr = torch.stack(overall_corrs)
        else:
            batch_context_features = torch.zeros(len(labels), 0, 0, dtype=torch.float32)
            batch_context_mask = torch.zeros(len(labels), 0, dtype=torch.float32)
            batch_context_scores = torch.zeros(len(labels), 0, dtype=torch.float32)
            batch_overall_corr = torch.zeros(len(labels), 0, dtype=torch.float32)

        # Padding 到最大长度
        max_len = max(s.shape[0] for s in s_list)

        s_padded = []
        z_padded = []
        is_ligand_padded = []
        single_mask_padded = []

        for s, z, is_ligand, single_mask in zip(s_list, z_list, is_ligand_list, single_mask_list):
            pad_len = max_len - s.shape[0]
            if pad_len > 0:
                s_padded.append(torch.cat([s, torch.zeros(pad_len, s.shape[1])], dim=0))
                if z.dim() == 2:
                    # compressed: [N, C_z]
                    z_padded.append(torch.cat([z, torch.zeros(pad_len, z.shape[1])], dim=0))
                else:
                    # full: [N, N, C_z]
                    z_padded.append(torch.cat([
                        torch.cat([z, torch.zeros(pad_len, z.shape[1], z.shape[2])], dim=0),
                        torch.zeros(max_len, pad_len, z.shape[2])
                    ], dim=1))
                is_ligand_padded.append(torch.cat([is_ligand, torch.zeros(pad_len)], dim=0))
                single_mask_padded.append(torch.cat([single_mask, torch.zeros(pad_len)], dim=0))
            else:
                s_padded.append(s)
                z_padded.append(z)
                is_ligand_padded.append(is_ligand)
                single_mask_padded.append(single_mask)

        return {
            's': torch.stack(s_padded),                     # [B, N, 384]
            'z': torch.stack(z_padded),                     # [B, N, C_z] 或 [B, N, N, C_z]
            'is_ligand': torch.stack(is_ligand_padded),     # [B, N]
            'single_mask': torch.stack(single_mask_padded), # [B, N]
            'labels': torch.tensor(labels, dtype=torch.long),  # [B]
            'sample_ids': sample_ids,
            'drfp': drfp,                                   # [2048] 同一个 reaction 共享
            'extra_features': extra_features,               # [B, D_extra]
            'context_features': batch_context_features,     # [B, K, D_ctx]
            'context_mask': batch_context_mask,             # [B, K]
            'context_scores': batch_context_scores,         # [B, K]
            'overall_corr': batch_overall_corr,             # [B, 2]
            'reaction': rxn,
        }
