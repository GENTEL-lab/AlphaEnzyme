#!/usr/bin/env python
"""Multitask kinetics training with specificity replay.

This script trains the shared enzyme classifier with a kinetics regression
head. It can replay the reaction-level specificity pairwise loss during
kinetics fine-tuning to reduce catastrophic forgetting.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from alphaenzyme.data.external_feature_bank import load_external_feature_bank
from alphaenzyme.data.pairwise_dataset import PairwiseReactionDataset
from alphaenzyme.loss.enzyme_kinetics_loss import PairwiseRankingLoss
from alphaenzyme.models.enzyme_classification_compressed import EnzymeClassificationHeadCompressed

TASK_COLUMNS = ["log_kcat", "log_km", "log_kcat_over_km_derived"]
SUBSTRATE_COLUMN_CANDIDATES = ("substrate_smiles", "smiles", "standardized_substrate")
DEFAULT_SUBSTRATE_FP_DIM = 2048
DEFAULT_EXTRA_SPLIT_DIM = 1152
DEFAULT_S_DIM = 384
DEFAULT_Z_DIM = 128


class ResidualMLPBlock(nn.Module):
    """A stable residual block for high-dimensional row-level descriptors."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StandaloneKineticsResidualBlock(nn.Module):
    """Residual block matching train_mlp_alphaenzyme_allinone_kinetics.py."""

    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class StandaloneKineticsRegressor(nn.Module):
    """Full-feature MLP with the same trunk as the strongest standalone kinetics model."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[StandaloneKineticsResidualBlock(hidden_size, dropout) for _ in range(num_layers)]
        )
        self.output = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(self.input(x)))


class GatedResidualMLPBlock(nn.Module):
    """SwiGLU residual block for wide tabular AlphaEnzyme features."""

    def __init__(self, hidden_size: int, dropout: float, expansion: int = 2) -> None:
        super().__init__()
        inner_size = hidden_size * int(expansion)
        self.norm = nn.LayerNorm(hidden_size)
        self.value = nn.Linear(hidden_size, inner_size)
        self.gate = nn.Linear(hidden_size, inner_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(inner_size, hidden_size)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = F.silu(self.gate(y)) * self.value(y)
        y = self.out(self.dropout(y))
        return x + self.dropout(y)


class AlphaReprSpecificityHead(nn.Module):
    """Specificity logit head that consumes the shared AlphaEnzyme representation."""

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


class LoRALinear(nn.Module):
    """Low-rank adapter for an existing Linear layer.

    The base weight is frozen; only lora_a/lora_b are trainable.  Saved
    checkpoints merge the delta back into a normal Linear state dict so the
    existing evaluation code can load them unchanged.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        self.lora_a = nn.Parameter(
            torch.zeros((self.rank, base.in_features), dtype=base.weight.dtype, device=base.weight.device)
        )
        self.lora_b = nn.Parameter(
            torch.zeros((base.out_features, self.rank), dtype=base.weight.dtype, device=base.weight.device)
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_a), self.lora_b) * self.scaling
        return base_out + delta

    @property
    def weight(self) -> torch.Tensor:
        # MultiheadAttention accesses out_proj.weight directly instead of
        # calling out_proj.forward(), so expose a differentiable merged weight.
        delta = torch.matmul(self.lora_b, self.lora_a) * self.scaling
        return self.base.weight + delta.to(device=self.base.weight.device, dtype=self.base.weight.dtype)

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def merged_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        weight = self.weight.detach()
        bias = self.base.bias.detach() if self.base.bias is not None else None
        return weight, bias


def _get_child_module(parent: nn.Module, name: str) -> nn.Module:
    if isinstance(parent, nn.Sequential):
        return parent[int(name)]
    if isinstance(parent, nn.ModuleList):
        return parent[int(name)]
    return getattr(parent, name)


def _set_child_module(parent: nn.Module, name: str, module: nn.Module) -> None:
    if isinstance(parent, nn.Sequential):
        parent[int(name)] = module
    elif isinstance(parent, nn.ModuleList):
        parent[int(name)] = module
    else:
        setattr(parent, name, module)


def inject_lora_layers(
    module: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_keywords: tuple[str, ...],
) -> list[str]:
    if rank <= 0:
        return []
    replaced: list[str] = []
    for full_name, child in list(module.named_modules()):
        if not full_name or not isinstance(child, nn.Linear):
            continue
        if not any(keyword in full_name for keyword in target_keywords):
            continue
        parent_name, _, child_name = full_name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        _set_child_module(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(full_name)
    return replaced


def merged_lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    lora_prefixes: dict[str, LoRALinear] = {
        name: child for name, child in module.named_modules() if isinstance(child, LoRALinear)
    }
    if not lora_prefixes:
        return module.state_dict()
    out: dict[str, torch.Tensor] = {}
    for key, value in module.state_dict().items():
        if any(key.startswith(prefix + ".") for prefix in lora_prefixes):
            continue
        out[key] = value.detach().clone()
    for prefix, child in lora_prefixes.items():
        weight, bias = child.merged_weight_bias()
        out[f"{prefix}.weight"] = weight.detach().cpu()
        if bias is not None:
            out[f"{prefix}.bias"] = bias.detach().cpu()
    return out


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load numpy-2 pickles in numpy-1 environments."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infinite_loader(loader):
    """Yield batches forever without caching past batches in memory."""
    while True:
        for batch in loader:
            yield batch


def to_f32(arr: np.ndarray) -> torch.Tensor:
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr).view(torch.float16).float()
    return torch.from_numpy(arr.astype(np.float32, copy=False))


class KineticsDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        embedding_dir: Path | None = None,
        drfp_path: Path | None = None,
        protein_pocket_bank_path: Path | None = None,
        substrate_fp_path: Path | None = None,
        row_extra_features_path: Path | None = None,
        aux_row_extra_features_path: Path | None = None,
        teacher_targets_path: Path | None = None,
        source_specs: dict[str, dict] | None = None,
        source_column: str = "feature_source",
        default_source: str = "default",
        cache_path: Path | None = None,
        skip_structural_features: bool = False,
        target_columns: list[str] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.source_column = source_column
        self.default_source = default_source
        self.cache_path = Path(cache_path) if cache_path else None
        self.skip_structural_features = bool(skip_structural_features)
        self.target_columns = list(target_columns) if target_columns else list(TASK_COLUMNS)
        if len(self.target_columns) != len(TASK_COLUMNS):
            raise ValueError(f"target_columns must have {len(TASK_COLUMNS)} columns, got {self.target_columns}")
        self.substrate_fp_path = Path(substrate_fp_path) if substrate_fp_path else None
        self.substrate_fp = self._load_substrate_fp(self.substrate_fp_path)
        self.substrate_fp_dim = self._infer_substrate_fp_dim(self.substrate_fp)
        self.row_extra_features_path = Path(row_extra_features_path) if row_extra_features_path else None
        self.row_extra_features = self._load_row_extra_features(self.row_extra_features_path)
        self.row_extra_feature_dim = self._infer_row_extra_feature_dim(self.row_extra_features)
        self.aux_row_extra_features_path = Path(aux_row_extra_features_path) if aux_row_extra_features_path else None
        self.aux_row_extra_features = self._load_row_extra_features(self.aux_row_extra_features_path)
        self.aux_row_extra_feature_dim = self._infer_row_extra_feature_dim(self.aux_row_extra_features)
        self.teacher_targets_path = Path(teacher_targets_path) if teacher_targets_path else None
        self.teacher_targets, self.teacher_masks = self._load_teacher_targets(self.teacher_targets_path)
        self.sources = self._prepare_sources(
            embedding_dir=embedding_dir,
            drfp_path=drfp_path,
            protein_pocket_bank_path=protein_pocket_bank_path,
            source_specs=source_specs,
            default_source=default_source,
        )
        self.source_meta = {
            name: {
                "embedding_dir": str(spec["embedding_dir"]),
                "drfp_path": str(spec["drfp_path"]),
                "bank_path": str(spec["bank_path"]),
            }
            for name, spec in self.sources.items()
        }
        self.df = self._load_or_build_df()
        if self.source_column not in self.df.columns:
            self.df[self.source_column] = self.default_source
        self.df[self.source_column] = self.df[self.source_column].fillna(self.default_source).astype(str)
        self.sample_ids = self.df["id"].astype(str).tolist()
        self.feature_sources = self.df[self.source_column].astype(str).tolist()
        self.source_names = list(self.sources.keys())
        self.source_name_to_id = {name: idx for idx, name in enumerate(self.source_names)}
        self.feature_source_ids = [self.source_name_to_id[name] for name in self.feature_sources]
        self.reactions = self.df["CANO_RXN_SMILES"].astype(str).tolist()
        self.substrate_keys = self.df["_substrate_key"].astype(str).tolist()
        target_frame = self.df.reindex(columns=self.target_columns)
        target_values = target_frame.to_numpy(dtype=np.float32)
        self.target_masks = np.isfinite(target_values).astype(np.float32)
        self.target_values = np.nan_to_num(target_values, nan=0.0).astype(np.float32, copy=False)

    @staticmethod
    def _load_substrate_fp(path: Path | None) -> dict[str, np.ndarray]:
        if path is None:
            return {}
        with Path(path).open("rb") as handle:
            raw = NumpyCompatUnpickler(handle).load()
        out = {}
        for key, value in raw.items():
            if value is None:
                continue
            out[str(key)] = np.asarray(value, dtype=np.float16)
        return out

    @staticmethod
    def _infer_substrate_fp_dim(substrate_fp: dict[str, np.ndarray]) -> int:
        if not substrate_fp:
            return DEFAULT_SUBSTRATE_FP_DIM
        first = next(iter(substrate_fp.values()))
        if first.ndim != 1:
            raise ValueError(f"substrate fingerprint must be 1D, got shape={first.shape}")
        return int(first.shape[0])

    @staticmethod
    def _load_row_extra_features(path: Path | None) -> dict[str, np.ndarray]:
        if path is None:
            return {}
        payload = np.load(path, allow_pickle=True)
        if "features" not in payload or "ids" not in payload:
            raise KeyError(f"{path} must contain 'features' and 'ids'")
        features = np.asarray(payload["features"], dtype=np.float16)
        features = np.nan_to_num(features, nan=0.0, posinf=20.0, neginf=-20.0)
        features = np.clip(features, -20.0, 20.0).astype(np.float16, copy=False)
        ids = payload["ids"].astype(str)
        if features.ndim != 2:
            raise ValueError(f"row extra features must be 2D, got shape={features.shape}")
        if len(ids) != features.shape[0]:
            raise ValueError(f"row extra ids/features mismatch: ids={len(ids)} features={features.shape}")
        return {str(sample_id): features[i] for i, sample_id in enumerate(ids)}

    @staticmethod
    def _infer_row_extra_feature_dim(row_extra_features: dict[str, np.ndarray]) -> int:
        if not row_extra_features:
            return 0
        first = next(iter(row_extra_features.values()))
        if first.ndim != 1:
            raise ValueError(f"row extra feature must be 1D, got shape={first.shape}")
        return int(first.shape[0])

    @staticmethod
    def _load_teacher_targets(path: Path | None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if path is None:
            return {}, {}
        payload = np.load(path, allow_pickle=True)
        target_key = "teacher_targets" if "teacher_targets" in payload else "targets"
        mask_key = "teacher_masks" if "teacher_masks" in payload else "masks"
        if target_key not in payload or mask_key not in payload or "ids" not in payload:
            raise KeyError(f"{path} must contain ids plus teacher_targets/teacher_masks or targets/masks")
        targets = np.asarray(payload[target_key], dtype=np.float32)
        masks = np.asarray(payload[mask_key], dtype=np.float32)
        ids = payload["ids"].astype(str)
        if targets.shape != masks.shape or targets.ndim != 2 or targets.shape[1] != len(TASK_COLUMNS):
            raise ValueError(f"teacher target shape mismatch: targets={targets.shape} masks={masks.shape}")
        if len(ids) != targets.shape[0]:
            raise ValueError(f"teacher ids/targets mismatch: ids={len(ids)} targets={targets.shape}")
        target_map = {str(sample_id): targets[i] for i, sample_id in enumerate(ids)}
        mask_map = {str(sample_id): masks[i] for i, sample_id in enumerate(ids)}
        return target_map, mask_map

    @staticmethod
    def _resolve_substrate_keys(df: pd.DataFrame) -> pd.Series:
        substrate = pd.Series([""] * len(df), index=df.index, dtype=object)
        for column in SUBSTRATE_COLUMN_CANDIDATES:
            if column not in df.columns:
                continue
            values = df[column].fillna("").astype(str).str.strip()
            mask = substrate.eq("") & values.ne("") & values.ne("nan") & values.ne("None")
            substrate.loc[mask] = values.loc[mask]
        return substrate

    @staticmethod
    def _prepare_sources(
        *,
        embedding_dir: Path | None,
        drfp_path: Path | None,
        protein_pocket_bank_path: Path | None,
        source_specs: dict[str, dict] | None,
        default_source: str,
    ) -> dict[str, dict]:
        if source_specs is None:
            if embedding_dir is None or drfp_path is None or protein_pocket_bank_path is None:
                raise ValueError("Single-source kinetics dataset requires embedding_dir, drfp_path and protein_pocket_bank_path")
            source_specs = {
                default_source: {
                    "embedding_dir": embedding_dir,
                    "drfp_path": drfp_path,
                    "bank_path": protein_pocket_bank_path,
                }
            }

        prepared = {}
        for source_name, spec in source_specs.items():
            emb_dir = Path(spec["embedding_dir"])
            drfp_file = Path(spec["drfp_path"])
            bank_file = Path(spec["bank_path"])
            external_bank = load_external_feature_bank(str(bank_file), include_z=True)
            with drfp_file.open("rb") as handle:
                drfp = pickle.load(handle)
            prepared[str(source_name)] = {
                "embedding_dir": emb_dir,
                "drfp_path": drfp_file,
                "bank_path": bank_file,
                "external_bank": external_bank,
                "drfp": drfp,
            }
        return prepared

    def _cache_meta(self) -> dict:
        return {
            "csv_path": str(self.csv_path.resolve()),
            "csv_mtime": self.csv_path.stat().st_mtime if self.csv_path.exists() else None,
            "source_column": self.source_column,
            "default_source": self.default_source,
            "substrate_fp_path": str(self.substrate_fp_path) if self.substrate_fp_path else None,
            "substrate_fp_dim": int(self.substrate_fp_dim),
            "row_extra_features_path": str(self.row_extra_features_path) if self.row_extra_features_path else None,
            "row_extra_feature_dim": int(self.row_extra_feature_dim),
            "aux_row_extra_features_path": str(self.aux_row_extra_features_path) if self.aux_row_extra_features_path else None,
            "aux_row_extra_feature_dim": int(self.aux_row_extra_feature_dim),
            "teacher_targets_path": str(self.teacher_targets_path) if self.teacher_targets_path else None,
            "target_columns": self.target_columns,
            "sources": self.source_meta,
        }

    def _load_or_build_df(self) -> pd.DataFrame:
        if self.cache_path and self.cache_path.exists():
            try:
                with self.cache_path.open("rb") as handle:
                    payload = pickle.load(handle)
                if payload.get("meta") == self._cache_meta():
                    df = payload.get("df")
                    if isinstance(df, pd.DataFrame):
                        print(f"Loaded kinetics dataset cache from {self.cache_path}: {len(df)} rows")
                        return df
            except Exception as exc:
                print(f"Warning: failed to load kinetics dataset cache {self.cache_path}: {exc}")

        df = pd.read_csv(self.csv_path, low_memory=False, dtype={"id": str})
        df["id"] = df["id"].astype(str)
        df["_substrate_key"] = self._resolve_substrate_keys(df)
        keep = []
        for i, row in df.iterrows():
            sample_id = str(row["id"])
            source_name = str(row.get(self.source_column, self.default_source))
            source = self.sources.get(source_name)
            if source is None:
                continue
            if not (source["embedding_dir"] / f"{sample_id}.npz").exists():
                continue
            if not source["external_bank"].contains(sample_id):
                continue
            rxn = str(row["CANO_RXN_SMILES"])
            fp = source["drfp"].get(rxn)
            if fp is None or np.abs(np.asarray(fp)).sum() == 0:
                continue
            keep.append(i)
        if len(keep) != len(df):
            missing = df.loc[~df.index.isin(keep), "id"].tolist()
            raise ValueError(f"Missing source/structure/protein-pocket/DRFP features for {len(missing)} inputs, e.g. {missing[:5]}")
        if not df.id.is_unique or df.id.isna().any():
            raise ValueError("Kinetics input IDs must be nonempty and unique")
        for label, values in [("row-extra", self.row_extra_features), ("substrate", self.substrate_fp)]:
            if values:
                keys = df.id if label == "row-extra" else df._substrate_key
                missing = [str(k) for k in keys if str(k) not in values]
                if missing: raise ValueError(f"Missing {label} features: {missing[:5]}")
        df = df.iloc[keep].reset_index(drop=True)
        if self.cache_path:
            try:
                cache_dir = self.cache_path.parent
                cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp.{os.getpid()}")
                with tmp_path.open("wb") as handle:
                    pickle.dump({"meta": self._cache_meta(), "df": df}, handle, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp_path, self.cache_path)
                print(f"Saved kinetics dataset cache to {self.cache_path}")
            except Exception as exc:
                print(f"Warning: failed to save kinetics dataset cache {self.cache_path}: {exc}")
        return df

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        source_name = self.feature_sources[idx]
        source = self.sources[source_name]
        rxn = self.reactions[idx]
        targets = self.target_values[idx]
        masks = self.target_masks[idx]
        if self.skip_structural_features:
            s = torch.zeros((1, DEFAULT_S_DIM), dtype=torch.float32)
            z = torch.zeros((1, DEFAULT_Z_DIM), dtype=torch.float32)
            is_ligand = torch.zeros((1,), dtype=torch.float32)
            single_mask = torch.ones((1,), dtype=torch.float32)
            drfp = torch.zeros((2048,), dtype=torch.float32)
            extra_features = torch.zeros((2 * DEFAULT_EXTRA_SPLIT_DIM,), dtype=torch.float32)
        else:
            emb = np.load(source["embedding_dir"] / f"{sample_id}.npz", allow_pickle=False)
            s = to_f32(emb["s"])
            z = to_f32(emb["z"])
            is_ligand = torch.from_numpy(emb["is_ligand"].astype(np.float32, copy=False))
            single_mask = torch.from_numpy(emb["single_mask"].astype(np.float32, copy=False))
            drfp = torch.from_numpy(np.asarray(source["drfp"][rxn], dtype=np.float32))
            extra_features = torch.from_numpy(source["external_bank"].get(sample_id).astype(np.float32, copy=False))
        substrate_key = self.substrate_keys[idx]
        substrate_fp = self.substrate_fp.get(substrate_key)
        if substrate_fp is None:
            substrate_fp = np.zeros((self.substrate_fp_dim,), dtype=np.float16)
            substrate_mask = np.float32(0.0)
        else:
            substrate_mask = np.float32(1.0)
        row_extra = self.row_extra_features.get(sample_id)
        if row_extra is None:
            row_extra = np.zeros((self.row_extra_feature_dim,), dtype=np.float16)
            row_extra_mask = np.float32(0.0)
        else:
            row_extra_mask = np.float32(1.0)
        aux_row_extra = self.aux_row_extra_features.get(sample_id)
        if aux_row_extra is None:
            aux_row_extra = np.zeros((self.aux_row_extra_feature_dim,), dtype=np.float16)
            aux_row_extra_mask = np.float32(0.0)
        else:
            aux_row_extra_mask = np.float32(1.0)
        teacher_targets = self.teacher_targets.get(sample_id)
        teacher_masks = self.teacher_masks.get(sample_id)
        if teacher_targets is None or teacher_masks is None:
            teacher_targets = np.zeros((len(TASK_COLUMNS),), dtype=np.float32)
            teacher_masks = np.zeros((len(TASK_COLUMNS),), dtype=np.float32)

        return {
            "s": s,
            "z": z,
            "is_ligand": is_ligand,
            "single_mask": single_mask,
            "drfp": drfp,
            "extra_features": extra_features,
            "substrate_fp": torch.from_numpy(substrate_fp.astype(np.float32, copy=False)),
            "substrate_mask": torch.tensor(substrate_mask, dtype=torch.float32),
            "row_extra_features": torch.from_numpy(row_extra.astype(np.float32, copy=False)),
            "row_extra_mask": torch.tensor(row_extra_mask, dtype=torch.float32),
            "aux_row_extra_features": torch.from_numpy(aux_row_extra.astype(np.float32, copy=False)),
            "aux_row_extra_mask": torch.tensor(aux_row_extra_mask, dtype=torch.float32),
            "targets": torch.from_numpy(targets.copy()),
            "masks": torch.from_numpy(masks.copy()),
            "teacher_targets": torch.from_numpy(teacher_targets.copy()),
            "teacher_masks": torch.from_numpy(teacher_masks.copy()),
            "sample_id": sample_id,
            "feature_source": source_name,
            "source_id": self.feature_source_ids[idx],
        }


def kinetics_collate(batch: list[dict]) -> dict:
    max_len = max(x["s"].shape[0] for x in batch)
    bsz = len(batch)
    c_s = batch[0]["s"].shape[-1]
    c_z = batch[0]["z"].shape[-1]
    s = torch.zeros((bsz, max_len, c_s), dtype=torch.float32)
    z = torch.zeros((bsz, max_len, c_z), dtype=torch.float32)
    is_ligand = torch.zeros((bsz, max_len), dtype=torch.float32)
    single_mask = torch.zeros((bsz, max_len), dtype=torch.float32)
    for i, item in enumerate(batch):
        n = item["s"].shape[0]
        s[i, :n] = item["s"]
        z[i, :n] = item["z"]
        is_ligand[i, :n] = item["is_ligand"]
        single_mask[i, :n] = item["single_mask"]
    return {
        "s": s,
        "z": z,
        "is_ligand": is_ligand,
        "single_mask": single_mask,
        "drfp": torch.stack([x["drfp"] for x in batch]),
        "extra_features": torch.stack([x["extra_features"] for x in batch]),
        "substrate_fp": torch.stack([x["substrate_fp"] for x in batch]),
        "substrate_mask": torch.stack([x["substrate_mask"] for x in batch]),
        "row_extra_features": torch.stack([x["row_extra_features"] for x in batch]),
        "row_extra_mask": torch.stack([x["row_extra_mask"] for x in batch]),
        "aux_row_extra_features": torch.stack([x["aux_row_extra_features"] for x in batch]),
        "aux_row_extra_mask": torch.stack([x["aux_row_extra_mask"] for x in batch]),
        "targets": torch.stack([x["targets"] for x in batch]),
        "masks": torch.stack([x["masks"] for x in batch]),
        "teacher_targets": torch.stack([x["teacher_targets"] for x in batch]),
        "teacher_masks": torch.stack([x["teacher_masks"] for x in batch]),
        "sample_ids": [x["sample_id"] for x in batch],
        "feature_sources": [x["feature_source"] for x in batch],
        "source_ids": torch.tensor([x["source_id"] for x in batch], dtype=torch.long),
    }


class KineticsHeads(nn.Module):
    def __init__(
        self,
        in_dim: int = 1152,
        hidden_size: int = 384,
        num_layers: int = 2,
        dropout: float = 0.1,
        mode: str = "shared",
        source_names: list[str] | None = None,
        source_mode: str = "none",
        branch_dims: dict[str, int] | None = None,
        task_branch_weight_map: dict[str, dict[str, float]] | None = None,
        row_extra_residual_dim: int = 0,
        row_extra_residual_mode: str = "plain",
        aux_row_extra_residual_dim: int = 0,
        row_extra_network: str = "mlp",
        aux_row_extra_network: str = "mlp",
        aux_row_extra_residual_mode: str = "plain",
        aux_row_extra_blend_weights: list[float] | tuple[float, float, float] | None = None,
        second_aux_row_extra_residual_dim: int = 0,
        second_aux_row_extra_network: str = "mlp",
        second_aux_row_extra_residual_mode: str = "plain",
        second_aux_row_extra_weights: list[float] | tuple[float, float, float] | None = None,
        extra_dropout: float | None = None,
        aux_row_extra_hidden_size: int | None = None,
        aux_row_extra_layers: int | None = None,
        second_aux_row_extra_hidden_size: int | None = None,
        second_aux_row_extra_layers: int | None = None,
        moe_num_experts: int = 4,
        moe_residual_num_experts: int = 0,
        structured_residual_mode: str = "none",
        structured_residual_network: str = "mlp",
        structured_residual_hidden_size: int | None = None,
        structured_residual_layers: int | None = None,
        structured_residual_output_weights: list[float] | tuple[float, float, float] | None = None,
        full_feature_mean: torch.Tensor | None = None,
        full_feature_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.source_names = list(source_names or [])
        self.source_mode = source_mode
        self.branch_dims = {str(k): int(v) for k, v in (branch_dims or {}).items()}
        self.task_names = list(TASK_COLUMNS)
        self.row_extra_residual_dim = int(row_extra_residual_dim)
        self.row_extra_residual_mode = str(row_extra_residual_mode)
        self.aux_row_extra_residual_dim = int(aux_row_extra_residual_dim)
        self.second_aux_row_extra_residual_dim = int(second_aux_row_extra_residual_dim)
        self.row_extra_network = str(row_extra_network)
        self.aux_row_extra_network = str(aux_row_extra_network)
        self.aux_row_extra_residual_mode = str(aux_row_extra_residual_mode)
        self.second_aux_row_extra_network = str(second_aux_row_extra_network)
        self.second_aux_row_extra_residual_mode = str(second_aux_row_extra_residual_mode)
        blend_weights = aux_row_extra_blend_weights if aux_row_extra_blend_weights is not None else [0.0, 0.0, 0.0]
        if len(blend_weights) != len(self.task_names):
            raise ValueError(f"aux_row_extra_blend_weights must have {len(self.task_names)} values")
        self.register_buffer(
            "aux_row_extra_blend_weights",
            torch.tensor([float(x) for x in blend_weights], dtype=torch.float32),
            persistent=True,
        )
        second_weights = second_aux_row_extra_weights if second_aux_row_extra_weights is not None else [0.0, 0.0, 0.0]
        if len(second_weights) != len(self.task_names):
            raise ValueError(f"second_aux_row_extra_weights must have {len(self.task_names)} values")
        self.register_buffer(
            "second_aux_row_extra_weights",
            torch.tensor([float(x) for x in second_weights], dtype=torch.float32),
            persistent=True,
        )
        self.extra_dropout = float(dropout if extra_dropout is None else extra_dropout)
        self.aux_row_extra_hidden_size = int(aux_row_extra_hidden_size or hidden_size)
        self.aux_row_extra_layers = int(aux_row_extra_layers or num_layers)
        self.second_aux_row_extra_hidden_size = int(second_aux_row_extra_hidden_size or self.aux_row_extra_hidden_size)
        self.second_aux_row_extra_layers = int(second_aux_row_extra_layers or self.aux_row_extra_layers)
        self.moe_num_experts = max(1, int(moe_num_experts))
        self.moe_residual_num_experts = max(0, int(moe_residual_num_experts))
        self.structured_residual_mode = str(structured_residual_mode)
        self.structured_residual_network = str(structured_residual_network)
        self.structured_residual_hidden_size = int(structured_residual_hidden_size or hidden_size)
        self.structured_residual_layers = int(structured_residual_layers or num_layers)
        if full_feature_mean is not None or full_feature_std is not None:
            if full_feature_mean is None or full_feature_std is None:
                raise ValueError("full_feature_mean and full_feature_std must be provided together")
            full_feature_mean = full_feature_mean.detach().float().view(1, -1)
            full_feature_std = full_feature_std.detach().float().view(1, -1).clamp_min(1e-6)
            if tuple(full_feature_mean.shape) != tuple(full_feature_std.shape):
                raise ValueError("full_feature_mean and full_feature_std must have the same shape")
        else:
            full_feature_mean = torch.empty(0, dtype=torch.float32)
            full_feature_std = torch.empty(0, dtype=torch.float32)
        self.register_buffer("full_feature_mean", full_feature_mean, persistent=True)
        self.register_buffer("full_feature_std", full_feature_std, persistent=True)
        structured_weights = structured_residual_output_weights
        if structured_weights is None:
            structured_weights = [0.0, 1.0, 0.0]
        if len(structured_weights) != len(self.task_names):
            raise ValueError(f"structured_residual_output_weights must have {len(self.task_names)} values")
        self.register_buffer(
            "structured_residual_output_weights",
            torch.tensor([float(x) for x in structured_weights], dtype=torch.float32),
            persistent=True,
        )

        class ChunkTransformerRegressor(nn.Module):
            """Treat fixed row features as chunk tokens before regression."""

            def __init__(
                self,
                input_dim: int,
                output_dim: int,
                hidden_dim: int,
                n_layers: int,
                p_drop: float,
                *,
                chunk_size: int = 128,
            ) -> None:
                super().__init__()
                self.input_dim = int(input_dim)
                self.chunk_size = int(chunk_size)
                self.n_chunks = int(math.ceil(self.input_dim / self.chunk_size))
                padded_dim = self.n_chunks * self.chunk_size
                self.pad_dim = padded_dim - self.input_dim
                self.chunk_norm = nn.LayerNorm(self.chunk_size)
                self.chunk_proj = nn.Linear(self.chunk_size, hidden_dim)
                self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
                self.pos = nn.Parameter(torch.zeros(1, self.n_chunks + 1, hidden_dim))
                n_heads = 8 if hidden_dim % 8 == 0 else 4
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=n_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=p_drop,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(n_layers)))
                self.out = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(p_drop),
                    nn.Linear(hidden_dim, output_dim),
                )
                nn.init.normal_(self.pos, std=0.02)
                nn.init.normal_(self.cls, std=0.02)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if self.pad_dim:
                    x = F.pad(x, (0, self.pad_dim))
                x = x.view(x.shape[0], self.n_chunks, self.chunk_size)
                tokens = self.chunk_proj(self.chunk_norm(x))
                cls = self.cls.expand(tokens.shape[0], -1, -1)
                tokens = torch.cat([cls, tokens], dim=1) + self.pos.to(dtype=tokens.dtype)
                encoded = self.encoder(tokens)
                return self.out(encoded[:, 0])

        def make_tower(
            input_dim: int,
            output_dim: int,
            *,
            tower_hidden_size: int | None = None,
            tower_layers: int | None = None,
            tower_dropout: float | None = None,
            network: str = "mlp",
        ) -> nn.Sequential:
            tower_hidden_size = int(tower_hidden_size or hidden_size)
            tower_layers = int(tower_layers or num_layers)
            tower_dropout = float(dropout if tower_dropout is None else tower_dropout)
            if network == "mlp":
                layers = [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, tower_hidden_size),
                    nn.GELU(),
                    nn.Dropout(tower_dropout),
                ]
                for _ in range(max(0, tower_layers - 1)):
                    layers.extend([
                        nn.Linear(tower_hidden_size, tower_hidden_size),
                        nn.GELU(),
                        nn.Dropout(tower_dropout),
                    ])
                layers.append(nn.Linear(tower_hidden_size, output_dim))
            elif network == "linear":
                layers = [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, output_dim),
                ]
            elif network == "resnet":
                layers = [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, tower_hidden_size),
                    nn.GELU(),
                    nn.Dropout(tower_dropout),
                ]
                for _ in range(max(1, tower_layers)):
                    layers.append(ResidualMLPBlock(tower_hidden_size, tower_dropout))
                layers.extend([nn.LayerNorm(tower_hidden_size), nn.Linear(tower_hidden_size, output_dim)])
            elif network == "gated_resnet":
                layers = [
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, tower_hidden_size),
                    nn.SiLU(),
                    nn.Dropout(tower_dropout),
                ]
                for _ in range(max(1, tower_layers)):
                    layers.append(GatedResidualMLPBlock(tower_hidden_size, tower_dropout))
                layers.extend([nn.LayerNorm(tower_hidden_size), nn.Linear(tower_hidden_size, output_dim)])
            elif network == "standalone_resnet":
                return StandaloneKineticsRegressor(
                    input_dim,
                    tower_hidden_size,
                    max(0, tower_layers),
                    tower_dropout,
                    output_dim,
                )
            elif network == "chunk_transformer":
                return ChunkTransformerRegressor(
                    input_dim,
                    output_dim,
                    tower_hidden_size,
                    tower_layers,
                    tower_dropout,
                )
            else:
                raise ValueError(f"Unsupported extra feature network: {network}")
            return nn.Sequential(*layers)

        if mode == "shared":
            self.net = make_tower(in_dim, 3)
        elif mode == "separate":
            self.heads = nn.ModuleList([make_tower(in_dim, 1) for _ in range(3)])
        elif mode == "shared_separate":
            self.shared = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            tower_layers = max(1, int(num_layers) - 1)
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            layer
                            for _ in range(tower_layers)
                            for layer in (nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout))
                        ],
                        nn.Linear(hidden_size, 1),
                    )
                    for _ in range(3)
                ]
            )
        elif mode == "full_feature_mlp":
            full_feature_dim = int(self.branch_dims.get("full_feature", 0))
            if full_feature_dim <= 0:
                raise ValueError("mode=full_feature_mlp requires branch_dims['full_feature'] > 0")
            self.full_feature_mlp = make_tower(
                full_feature_dim,
                len(self.task_names),
                tower_dropout=self.extra_dropout,
                network=self.row_extra_network,
            )
        elif mode == "row_extra_only":
            row_extra_dim = int(self.branch_dims.get("row_extra", 0))
            if row_extra_dim <= 0:
                raise ValueError("mode=row_extra_only requires branch_dims['row_extra'] > 0")
            self.row_extra_only = make_tower(
                row_extra_dim,
                len(self.task_names),
                tower_dropout=self.extra_dropout,
                network=self.row_extra_network,
            )
        elif mode in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
            if not self.branch_dims:
                raise ValueError(f"mode={mode} requires non-empty branch_dims")
            self.branch_names = list(self.branch_dims.keys())
            self.branch_proj = nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.LayerNorm(dim),
                        nn.Linear(dim, hidden_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, hidden_size),
                        nn.GELU(),
                    )
                    for name, dim in self.branch_dims.items()
                }
            )
            self.interaction_pairs: list[tuple[str, str]] = []
            self.interaction_names: list[str] = []
            if mode in {"task_interaction_gate", "task_moe_gate"}:
                candidate_pairs = [
                    ("sample", "reaction"),
                    ("sample", "substrate"),
                    ("reaction", "substrate"),
                    ("protein", "substrate"),
                    ("pocket", "substrate"),
                    ("pocket", "reaction"),
                ]
                self.interaction_pairs = [
                    (left, right)
                    for left, right in candidate_pairs
                    if left in self.branch_names and right in self.branch_names
                ]
                self.interaction_names = [f"{left}_x_{right}" for left, right in self.interaction_pairs]
                self.interaction_proj = nn.ModuleDict(
                    {
                        name: nn.Sequential(
                            nn.LayerNorm(hidden_size),
                            nn.Linear(hidden_size, hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )
                        for name in self.interaction_names
                    }
                )
            self.fusion_names = self.branch_names + self.interaction_names
            fusion_in_dim = hidden_size * len(self.fusion_names)
            self.task_gate = nn.ModuleDict(
                {
                    task: nn.Sequential(
                        nn.LayerNorm(fusion_in_dim),
                        nn.Linear(fusion_in_dim, hidden_size),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_size, len(self.fusion_names)),
                    )
                    for task in self.task_names
                }
            )
            if mode == "task_moe_gate":
                self.task_expert_gate = nn.ModuleDict(
                    {
                        task: nn.Sequential(
                            nn.LayerNorm(fusion_in_dim),
                            nn.Linear(fusion_in_dim, hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_size, self.moe_num_experts),
                        )
                        for task in self.task_names
                    }
                )
                self.task_experts = nn.ModuleDict(
                    {
                        task: nn.ModuleList([make_tower(hidden_size, 1) for _ in range(self.moe_num_experts)])
                        for task in self.task_names
                    }
                )
            else:
                self.task_heads = nn.ModuleDict({task: make_tower(hidden_size, 1) for task in self.task_names})
            prior_map = {
                "log_kcat": {"sample": 1.0, "reaction": 1.0, "protein": 0.1, "pocket": 0.8, "substrate": 0.4},
                "log_km": {"sample": 0.6, "reaction": 0.2, "protein": 1.0, "pocket": 0.3, "substrate": 1.0},
                "log_kcat_over_km_derived": {"sample": 0.8, "reaction": 0.8, "protein": 0.5, "pocket": 0.6, "substrate": 0.8},
            }
            for task_prior in prior_map.values():
                task_prior.setdefault("row_extra", task_prior.get("substrate", 0.5))
            interaction_prior_map = {
                "log_kcat": {
                    "sample_x_reaction": 1.0,
                    "sample_x_substrate": 0.7,
                    "reaction_x_substrate": 0.8,
                    "protein_x_substrate": 0.5,
                    "pocket_x_substrate": 1.0,
                    "pocket_x_reaction": 0.8,
                },
                "log_km": {
                    "sample_x_reaction": 0.3,
                    "sample_x_substrate": 0.8,
                    "reaction_x_substrate": 0.6,
                    "protein_x_substrate": 1.1,
                    "pocket_x_substrate": 0.9,
                    "pocket_x_reaction": 0.4,
                },
                "log_kcat_over_km_derived": {
                    "sample_x_reaction": 0.8,
                    "sample_x_substrate": 0.8,
                    "reaction_x_substrate": 0.8,
                    "protein_x_substrate": 0.9,
                    "pocket_x_substrate": 1.0,
                    "pocket_x_reaction": 0.7,
                },
            }
            bias_rows = []
            for task in self.task_names:
                row = [
                    float(
                        prior_map.get(task, {}).get(
                            name,
                            interaction_prior_map.get(task, {}).get(name, 0.0),
                        )
                    )
                    for name in self.fusion_names
                ]
                bias_rows.append(row)
            self.task_branch_bias = nn.Parameter(torch.tensor(bias_rows, dtype=torch.float32))
            static_rows = []
            task_branch_weight_map = task_branch_weight_map or {}
            for task in self.task_names:
                branch_weights = task_branch_weight_map.get(task, {})
                row = [float(branch_weights.get(name, 1.0)) for name in self.fusion_names]
                if not any(weight > 0 for weight in row):
                    raise ValueError(f"task={task} has no positive branch weights")
                static_rows.append(row)
            self.register_buffer(
                "task_branch_static_weights",
                torch.tensor(static_rows, dtype=torch.float32),
                persistent=True,
            )
            if self.moe_residual_num_experts > 0:
                self.task_moe_residual_gate = nn.ModuleDict(
                    {
                        task: nn.Sequential(
                            nn.LayerNorm(fusion_in_dim),
                            nn.Linear(fusion_in_dim, hidden_size),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_size, self.moe_residual_num_experts),
                        )
                        for task in self.task_names
                    }
                )
                self.task_moe_residual_experts = nn.ModuleDict(
                    {
                        task: nn.ModuleList([make_tower(hidden_size, 1) for _ in range(self.moe_residual_num_experts)])
                        for task in self.task_names
                    }
                )
                for experts in self.task_moe_residual_experts.values():
                    for expert in experts:
                        for layer in reversed(expert):
                            if isinstance(layer, nn.Linear):
                                nn.init.zeros_(layer.weight)
                                nn.init.zeros_(layer.bias)
                                break
        else:
            raise ValueError(f"Unsupported kinetics head mode: {mode}")

        self.structured_residual = None
        self.structured_residual_names: list[str] = []
        self.structured_residual_pairs: list[tuple[str, str]] = []
        if self.structured_residual_mode != "none":
            if mode not in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
                raise ValueError("structured residual requires a branch kinetics head")
            if self.structured_residual_mode not in {"concat", "pairwise"}:
                raise ValueError(f"Unsupported structured_residual_mode: {self.structured_residual_mode}")
            self.structured_residual_names = [
                name
                for name in ("sample", "reaction", "protein", "pocket", "substrate")
                if name in self.branch_dims
            ]
            if len(self.structured_residual_names) < 2:
                raise ValueError("structured residual requires at least two branch features")
            if self.structured_residual_mode == "pairwise":
                candidate_pairs = [
                    ("protein", "substrate"),
                    ("pocket", "substrate"),
                    ("reaction", "substrate"),
                    ("sample", "substrate"),
                    ("pocket", "reaction"),
                    ("protein", "reaction"),
                ]
                self.structured_residual_pairs = [
                    (left, right)
                    for left, right in candidate_pairs
                    if left in self.structured_residual_names and right in self.structured_residual_names
                ]
            structured_input_dim = hidden_size * (
                len(self.structured_residual_names) + len(self.structured_residual_pairs)
            )
            self.structured_residual = make_tower(
                structured_input_dim,
                len(self.task_names),
                tower_hidden_size=self.structured_residual_hidden_size,
                tower_layers=self.structured_residual_layers,
                tower_dropout=self.extra_dropout,
                network=self.structured_residual_network,
            )
            for layer in reversed(list(self.structured_residual.modules())):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break

        self.row_extra_residual = None
        self.row_extra_residual_proj = None
        self.row_extra_residual_interaction = None
        self.row_extra_residual_interaction_names: list[str] = []
        if self.row_extra_residual_dim > 0:
            if self.row_extra_residual_mode == "plain":
                self.row_extra_residual = make_tower(
                    self.row_extra_residual_dim,
                    len(self.task_names),
                    tower_dropout=self.extra_dropout,
                    network=self.row_extra_network,
                )
                residual_module = self.row_extra_residual
            elif self.row_extra_residual_mode == "interaction":
                if mode not in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
                    raise ValueError("row_extra_residual_mode=interaction requires a branch kinetics head")
                self.row_extra_residual_interaction_names = [
                    name
                    for name in ("sample", "reaction", "protein", "pocket", "substrate")
                    if name in self.branch_dims
                ]
                self.row_extra_residual_proj = nn.Sequential(
                    nn.LayerNorm(self.row_extra_residual_dim),
                    nn.Linear(self.row_extra_residual_dim, hidden_size),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
                residual_input_dim = hidden_size * (1 + len(self.row_extra_residual_interaction_names))
                self.row_extra_residual_interaction = make_tower(
                    residual_input_dim,
                    len(self.task_names),
                    tower_dropout=self.extra_dropout,
                    network=self.row_extra_network,
                )
                residual_module = self.row_extra_residual_interaction
            else:
                    raise ValueError(f"Unsupported row_extra_residual_mode: {self.row_extra_residual_mode}")
            for layer in reversed(list(residual_module.modules())):
                if isinstance(layer, nn.Linear):
                    nn.init.zeros_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    break

        self.aux_row_extra_residual = None
        if self.aux_row_extra_residual_dim > 0:
            aux_residual_input_dim = self.aux_row_extra_residual_dim
            if self.aux_row_extra_residual_mode in {"prediction", "prediction_scaled"}:
                aux_residual_input_dim += len(self.task_names)
            elif self.aux_row_extra_residual_mode not in {"plain", "blend", "scaled"}:
                raise ValueError(f"Unsupported aux_row_extra_residual_mode: {self.aux_row_extra_residual_mode}")
            if self.aux_row_extra_network.startswith("per_task_"):
                base_network = self.aux_row_extra_network[len("per_task_") :]
                if not base_network:
                    raise ValueError("aux_row_extra_network=per_task_ must specify a base network")
                self.aux_row_extra_residual = nn.ModuleList(
                    [
                        make_tower(
                            aux_residual_input_dim,
                            1,
                            tower_hidden_size=self.aux_row_extra_hidden_size,
                            tower_layers=self.aux_row_extra_layers,
                            tower_dropout=self.extra_dropout,
                            network=base_network,
                        )
                        for _ in self.task_names
                    ]
                )
            else:
                self.aux_row_extra_residual = make_tower(
                    aux_residual_input_dim,
                    len(self.task_names),
                    tower_hidden_size=self.aux_row_extra_hidden_size,
                    tower_layers=self.aux_row_extra_layers,
                    tower_dropout=self.extra_dropout,
                    network=self.aux_row_extra_network,
                )
            for module in self.aux_row_extra_residual.modules():
                if isinstance(module, nn.Sequential):
                    for layer in reversed(list(module.modules())):
                        if isinstance(layer, nn.Linear):
                            nn.init.zeros_(layer.weight)
                            nn.init.zeros_(layer.bias)
                            break

        self.second_aux_row_extra_residual = None
        if self.second_aux_row_extra_residual_dim > 0:
            second_input_dim = self.second_aux_row_extra_residual_dim
            if self.second_aux_row_extra_residual_mode == "prediction":
                second_input_dim += len(self.task_names)
            elif self.second_aux_row_extra_residual_mode != "plain":
                raise ValueError(
                    f"Unsupported second_aux_row_extra_residual_mode: {self.second_aux_row_extra_residual_mode}"
                )
            if self.second_aux_row_extra_network.startswith("per_task_"):
                base_network = self.second_aux_row_extra_network[len("per_task_") :]
                if not base_network:
                    raise ValueError("second_aux_row_extra_network=per_task_ must specify a base network")
                self.second_aux_row_extra_residual = nn.ModuleList(
                    [
                        make_tower(
                            second_input_dim,
                            1,
                            tower_hidden_size=self.second_aux_row_extra_hidden_size,
                            tower_layers=self.second_aux_row_extra_layers,
                            tower_dropout=self.extra_dropout,
                            network=base_network,
                        )
                        for _ in self.task_names
                    ]
                )
            else:
                self.second_aux_row_extra_residual = make_tower(
                    second_input_dim,
                    len(self.task_names),
                    tower_hidden_size=self.second_aux_row_extra_hidden_size,
                    tower_layers=self.second_aux_row_extra_layers,
                    tower_dropout=self.extra_dropout,
                    network=self.second_aux_row_extra_network,
                )
            for module in self.second_aux_row_extra_residual.modules():
                if isinstance(module, nn.Sequential):
                    for layer in reversed(list(module.modules())):
                        if isinstance(layer, nn.Linear):
                            nn.init.zeros_(layer.weight)
                            nn.init.zeros_(layer.bias)
                            break

        if self.source_mode == "affine":
            if not self.source_names:
                raise ValueError("source_mode=affine requires non-empty source_names")
            self.source_scale = nn.Embedding(len(self.source_names), 3)
            self.source_bias = nn.Embedding(len(self.source_names), 3)
            nn.init.zeros_(self.source_scale.weight)
            nn.init.zeros_(self.source_bias.weight)
        elif self.source_mode != "none":
            raise ValueError(f"Unsupported source head mode: {self.source_mode}")

    def forward(
        self,
        x: torch.Tensor | None = None,
        source_ids: torch.Tensor | None = None,
        branch_features: dict[str, torch.Tensor] | None = None,
        branch_masks: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        projected_by_name = {}
        if self.mode == "shared":
            out = self.net(x)
        elif self.mode == "separate":
            out = torch.cat([head(x) for head in self.heads], dim=-1)
        elif self.mode == "shared_separate":
            shared = self.shared(x)
            out = torch.cat([head(shared) for head in self.heads], dim=-1)
        elif self.mode == "full_feature_mlp":
            if branch_features is None or "full_feature" not in branch_features:
                raise ValueError("full_feature_mlp kinetics head requires branch_features['full_feature']")
            full_feature = branch_features["full_feature"]
            if self.full_feature_mean.numel() > 0:
                mean = self.full_feature_mean.to(device=full_feature.device, dtype=full_feature.dtype)
                std = self.full_feature_std.to(device=full_feature.device, dtype=full_feature.dtype)
                if mean.shape[-1] != full_feature.shape[-1]:
                    raise ValueError(
                        f"full_feature stats dim {mean.shape[-1]} does not match input dim {full_feature.shape[-1]}"
                    )
                full_feature = (full_feature - mean) / std
            out = self.full_feature_mlp(full_feature)
            if branch_masks is not None and "full_feature" in branch_masks:
                out = out * branch_masks["full_feature"].to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        elif self.mode == "row_extra_only":
            if branch_features is None or "row_extra" not in branch_features:
                raise ValueError("row_extra_only kinetics head requires branch_features['row_extra']")
            out = self.row_extra_only(branch_features["row_extra"])
            if branch_masks is not None and "row_extra" in branch_masks:
                out = out * branch_masks["row_extra"].to(device=out.device, dtype=out.dtype).unsqueeze(-1)
        else:
            if branch_features is None:
                raise ValueError(f"branch_features are required when mode={self.mode}")
            projected = []
            masks_by_name = {}
            masks = []
            for name in self.branch_names:
                if name not in branch_features:
                    raise KeyError(f"Missing branch feature: {name}")
                feat = branch_features[name]
                projected_feat = self.branch_proj[name](feat)
                projected_by_name[name] = projected_feat
                projected.append(projected_feat)
                mask = None if branch_masks is None else branch_masks.get(name)
                if mask is None:
                    mask = torch.ones((feat.shape[0],), device=feat.device, dtype=feat.dtype)
                mask = mask.to(device=feat.device, dtype=feat.dtype)
                masks_by_name[name] = mask
                masks.append(mask)
            if self.mode in {"task_interaction_gate", "task_moe_gate"}:
                for (left, right), interaction_name in zip(self.interaction_pairs, self.interaction_names):
                    interaction = projected_by_name[left] * projected_by_name[right]
                    projected.append(self.interaction_proj[interaction_name](interaction))
                    masks.append(masks_by_name[left] * masks_by_name[right])
            stacked = torch.stack(projected, dim=1)
            mask_tensor = torch.stack(masks, dim=1)
            flat = torch.cat(projected, dim=-1)
            task_outputs = []
            for task_idx, task in enumerate(self.task_names):
                gate_logits = self.task_gate[task](flat) + self.task_branch_bias[task_idx].to(
                    device=flat.device,
                    dtype=flat.dtype,
                )
                gate_logits = gate_logits.masked_fill(mask_tensor <= 0, -1e4)
                weights = torch.softmax(gate_logits, dim=-1)
                static_weights = self.task_branch_static_weights[task_idx].to(device=flat.device, dtype=flat.dtype)
                weights = weights * static_weights.unsqueeze(0)
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)
                if self.mode == "task_moe_gate":
                    expert_logits = self.task_expert_gate[task](flat)
                    expert_weights = torch.softmax(expert_logits, dim=-1)
                    expert_outputs = torch.cat([expert(fused) for expert in self.task_experts[task]], dim=-1)
                    task_out = (expert_weights * expert_outputs).sum(dim=-1, keepdim=True)
                else:
                    task_out = self.task_heads[task](fused)
                if self.moe_residual_num_experts > 0:
                    residual_logits = self.task_moe_residual_gate[task](flat)
                    residual_weights = torch.softmax(residual_logits, dim=-1)
                    residual_outputs = torch.cat(
                        [expert(fused) for expert in self.task_moe_residual_experts[task]],
                        dim=-1,
                    )
                    task_out = task_out + (residual_weights * residual_outputs).sum(dim=-1, keepdim=True)
                task_outputs.append(task_out)
            out = torch.cat(task_outputs, dim=-1)
        structured_delta = None
        structured_weights = None
        if self.structured_residual is not None:
            structured_pieces = []
            for name in self.structured_residual_names:
                if name not in projected_by_name:
                    raise KeyError(f"Missing projected branch feature for structured residual: {name}")
                structured_pieces.append(projected_by_name[name])
            for left, right in self.structured_residual_pairs:
                structured_pieces.append(projected_by_name[left] * projected_by_name[right])
            structured_delta = self.structured_residual(torch.cat(structured_pieces, dim=-1))
            structured_weights = self.structured_residual_output_weights.to(
                device=out.device,
                dtype=out.dtype,
            ).view(1, -1)
        if self.row_extra_residual_dim > 0 and branch_features is not None and "row_extra" in branch_features:
            if self.row_extra_residual_mode == "plain":
                residual = self.row_extra_residual(branch_features["row_extra"])
            elif self.row_extra_residual_mode == "interaction":
                if self.row_extra_residual_proj is None or self.row_extra_residual_interaction is None:
                    raise RuntimeError("row_extra interaction residual modules are not initialized")
                row_extra_projected = self.row_extra_residual_proj(branch_features["row_extra"])
                residual_pieces = [row_extra_projected]
                for name in self.row_extra_residual_interaction_names:
                    if name not in projected_by_name:
                        raise KeyError(f"Missing projected branch feature for row_extra interaction: {name}")
                    residual_pieces.append(row_extra_projected * projected_by_name[name])
                residual = self.row_extra_residual_interaction(torch.cat(residual_pieces, dim=-1))
            else:
                raise ValueError(f"Unsupported row_extra_residual_mode: {self.row_extra_residual_mode}")
            residual_mask = None if branch_masks is None else branch_masks.get("row_extra")
            if residual_mask is not None:
                residual = residual * residual_mask.to(device=residual.device, dtype=residual.dtype).unsqueeze(-1)
            out = out + residual
        if self.aux_row_extra_residual_dim > 0 and branch_features is not None and "aux_row_extra" in branch_features:
            if self.aux_row_extra_residual is None:
                raise RuntimeError("aux row-extra residual module is not initialized")
            aux_residual_input = branch_features["aux_row_extra"]
            if self.aux_row_extra_residual_mode in {"prediction", "prediction_scaled"}:
                aux_residual_input = torch.cat([aux_residual_input, out.detach()], dim=-1)
            if isinstance(self.aux_row_extra_residual, nn.ModuleList):
                aux_residual = torch.cat([tower(aux_residual_input) for tower in self.aux_row_extra_residual], dim=-1)
            else:
                aux_residual = self.aux_row_extra_residual(aux_residual_input)
            aux_mask = None if branch_masks is None else branch_masks.get("aux_row_extra")
            if aux_mask is not None:
                aux_residual = aux_residual * aux_mask.to(device=aux_residual.device, dtype=aux_residual.dtype).unsqueeze(-1)
            if self.aux_row_extra_residual_mode == "blend":
                blend_weights = self.aux_row_extra_blend_weights.to(device=out.device, dtype=out.dtype).view(1, -1)
                out = out * (1.0 - blend_weights) + aux_residual * blend_weights
            elif self.aux_row_extra_residual_mode in {"scaled", "prediction_scaled"}:
                blend_weights = self.aux_row_extra_blend_weights.to(device=out.device, dtype=out.dtype).view(1, -1)
                out = out + aux_residual * blend_weights
            else:
                out = out + aux_residual
        if structured_delta is not None:
            # Apply this last so a blend-mode auxiliary head cannot overwrite the learnable correction.
            out = out + structured_delta * structured_weights
        if (
            self.second_aux_row_extra_residual_dim > 0
            and branch_features is not None
            and "aux_row_extra" in branch_features
        ):
            if self.second_aux_row_extra_residual is None:
                raise RuntimeError("second aux row-extra residual module is not initialized")
            second_input = branch_features["aux_row_extra"]
            if self.second_aux_row_extra_residual_mode == "prediction":
                second_input = torch.cat([second_input, out.detach()], dim=-1)
            if isinstance(self.second_aux_row_extra_residual, nn.ModuleList):
                second_residual = torch.cat(
                    [tower(second_input) for tower in self.second_aux_row_extra_residual],
                    dim=-1,
                )
            else:
                second_residual = self.second_aux_row_extra_residual(second_input)
            second_mask = None if branch_masks is None else branch_masks.get("aux_row_extra")
            if second_mask is not None:
                second_residual = second_residual * second_mask.to(
                    device=second_residual.device,
                    dtype=second_residual.dtype,
                ).unsqueeze(-1)
            second_weights = self.second_aux_row_extra_weights.to(device=out.device, dtype=out.dtype).view(1, -1)
            out = out + second_residual * second_weights
        if self.source_mode == "affine":
            if source_ids is None:
                raise ValueError("source_ids are required when source_mode=affine")
            scale = 1.0 + self.source_scale(source_ids)
            bias = self.source_bias(source_ids)
            out = out * scale + bias
        return out


def infer_extra_config_from_state(state: dict) -> tuple[list[int], str]:
    branch_keys = []
    for key in state:
        if key.startswith("extra_branch_proj.") and key.endswith(".1.weight"):
            try:
                branch_keys.append((int(key.split(".")[1]), key))
            except Exception:
                pass
    branch_keys.sort()
    branch_dims = [int(state[key].shape[1]) for _, key in branch_keys]
    if branch_dims:
        branch_count = 1 + len(branch_dims)
        if "extra_branch_fusion.4.weight" in state:
            mode = "branch_gate_sum" if int(state["extra_branch_fusion.4.weight"].shape[0]) == branch_count else "branch_concat_mlp"
        elif "extra_gate.weight" in state:
            mode = "gated_add"
        else:
            mode = "delta"
        return branch_dims, mode

    if "extra_fusion.4.weight" in state and int(state["extra_fusion.4.weight"].shape[0]) == 2:
        return [], "branch_gate_sum"
    if "extra_gate.weight" in state:
        return [], "gated_add"
    if "extra_fusion.4.weight" in state:
        return [], "branch_concat_mlp"
    return [], "delta"


def build_classifier(dropout: float, state: dict) -> EnzymeClassificationHeadCompressed:
    extra_branch_dims, extra_fusion_mode = infer_extra_config_from_state(state)
    return EnzymeClassificationHeadCompressed(
        {
            "c_s": 384,
            "c_z": 128,
            "hidden_size": 384,
            "dropout": dropout,
            "n_transformer_layers": 4,
            "n_heads_single": 8,
            "use_drfp": True,
            "drfp_dim": 2048,
            "pooling_mode": "ligand_plus_global",
            "global_pool_scale_init": 0.35,
            "main_input_mode": "sz",
            "use_extra_features": True,
            "extra_feature_dim": 2304,
            "extra_hidden_size": 384,
            "extra_fusion_mode": extra_fusion_mode,
            "extra_feature_branch_dims": extra_branch_dims,
            "extra_token_injection": True,
            "extra_branch_scale_init": 1.0,
        }
    )


def reset_module_parameters(module: nn.Module) -> None:
    for child in module.children():
        reset_module_parameters(child)
    if hasattr(module, "reset_parameters"):
        try:
            module.reset_parameters()
        except TypeError:
            pass


def load_shape_compatible_state(module: nn.Module, state: dict) -> tuple[list[str], list[str], dict[str, dict[str, list[int]]]]:
    """Load checkpoint tensors whose names and shapes still match the current module."""
    current_state = module.state_dict()
    compatible_state = {}
    skipped_shape = {}
    unexpected = []
    for name, value in state.items():
        if name not in current_state:
            unexpected.append(name)
            continue
        current_value = current_state[name]
        if tuple(current_value.shape) == tuple(value.shape):
            compatible_state[name] = value
        else:
            skipped_shape[name] = {
                "checkpoint": list(value.shape),
                "current": list(current_value.shape),
            }
    missing, unexpected_from_load = module.load_state_dict(compatible_state, strict=False)
    return list(missing), unexpected + list(unexpected_from_load), skipped_shape


def filter_state_prefixes(state: dict, prefixes: tuple[str, ...]) -> dict:
    """Return a state dict without parameters whose names start with any prefix."""
    if not prefixes:
        return state
    return {name: value for name, value in state.items() if not name.startswith(prefixes)}


def set_aux_row_extra_blend_weights(heads: nn.Module, weights: list[float] | tuple[float, ...]) -> None:
    """Keep the non-trainable aux blend buffer aligned with the active run config."""
    if not hasattr(heads, "aux_row_extra_blend_weights"):
        return
    blend = getattr(heads, "aux_row_extra_blend_weights")
    values = torch.tensor(list(weights), device=blend.device, dtype=blend.dtype)
    if tuple(values.shape) != tuple(blend.shape):
        raise ValueError(f"aux blend shape mismatch: expected {tuple(blend.shape)}, got {tuple(values.shape)}")
    blend.copy_(values)


@dataclass
class InitCheckpoint:
    checkpoint_kind: str
    classifier_state: dict
    kinetics_head_state: dict | None
    specificity_alpha_head_state: dict | None
    specificity_alpha_base_logit_residual_scale: object | None
    run_meta: dict
    dropout: float
    kin_head_hidden_size: int
    kin_head_layers: int
    kin_head_mode: str


def load_init_checkpoint(
    checkpoint_path: Path,
    *,
    default_dropout: float,
    default_head_hidden_size: int,
    default_head_layers: int,
    default_head_mode: str,
) -> InitCheckpoint:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "classifier" in payload and "kinetics_heads" in payload:
        run_meta = payload.get("run_meta") or {}
        return InitCheckpoint(
            checkpoint_kind="multitask",
            classifier_state=payload["classifier"],
            kinetics_head_state=payload.get("kinetics_heads"),
            specificity_alpha_head_state=payload.get("specificity_alpha_head"),
            specificity_alpha_base_logit_residual_scale=payload.get(
                "specificity_alpha_base_logit_residual_scale"
            ),
            run_meta=run_meta,
            dropout=float(run_meta.get("dropout", default_dropout)),
            kin_head_hidden_size=int(run_meta.get("kin_head_hidden_size", default_head_hidden_size)),
            kin_head_layers=int(run_meta.get("kin_head_layers", default_head_layers)),
            kin_head_mode=str(run_meta.get("kin_head_mode", default_head_mode)),
        )
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type from {checkpoint_path}: {type(payload)!r}")
    return InitCheckpoint(
        checkpoint_kind="classifier",
        classifier_state=payload,
        kinetics_head_state=None,
        specificity_alpha_head_state=None,
        specificity_alpha_base_logit_residual_scale=None,
        run_meta={},
        dropout=float(default_dropout),
        kin_head_hidden_size=int(default_head_hidden_size),
        kin_head_layers=int(default_head_layers),
        kin_head_mode=str(default_head_mode),
    )


def apply_train_mode(classifier: nn.Module, mode: str) -> list[str]:
    mode = mode.lower()
    trainable_names = []
    for name, param in classifier.named_parameters():
        trainable = False
        if mode == "all":
            trainable = True
        elif mode == "none":
            trainable = False
        elif mode == "top":
            trainable = name.startswith(("classifier", "extra_", "drfp_mlp", "drfp_scale", "global_pool_scale"))
        elif mode == "extra":
            trainable = name.startswith("extra_")
        elif mode == "fusion":
            trainable = name.startswith(("extra_fusion", "extra_branch_fusion", "extra_score", "extra_logit_scale", "classifier"))
        elif mode in {"classifier_only", "logit_only"}:
            trainable = name.startswith(("classifier", "extra_score", "extra_logit_scale"))
        elif mode == "lora":
            trainable = any(part in name for part in (".lora_a", ".lora_b"))
        else:
            raise ValueError(f"Unsupported classifier_train_mode: {mode}")
        param.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
    return trainable_names


def _compute_stats_from_df(df: pd.DataFrame) -> dict:
    values = df[TASK_COLUMNS].to_numpy(dtype=np.float32)
    stats = {}
    for i, name in enumerate(TASK_COLUMNS):
        col = values[:, i]
        col = col[np.isfinite(col)]
        stats[name] = {
            "mean": float(col.mean()) if col.size else 0.0,
            "std": float(col.std()) if col.size and col.std() > 1e-6 else 1.0,
        }
    return stats


def compute_label_stats(dataset: KineticsDataset, source_specific: bool = False) -> dict:
    global_stats = _compute_stats_from_df(dataset.df)
    if not source_specific:
        return global_stats
    by_source = {}
    for source_name, sub_df in dataset.df.groupby(dataset.source_column):
        by_source[str(source_name)] = _compute_stats_from_df(sub_df)
    return {"global": global_stats, "by_source": by_source}


def is_source_specific_stats(stats: dict) -> bool:
    return isinstance(stats, dict) and "global" in stats and "by_source" in stats


def resolve_row_stats(
    stats: dict,
    feature_sources: list[str] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_source_specific_stats(stats):
        means = torch.tensor([stats[k]["mean"] for k in TASK_COLUMNS], device=device, dtype=dtype)
        stds = torch.tensor([stats[k]["std"] for k in TASK_COLUMNS], device=device, dtype=dtype)
        return means.view(1, -1), stds.view(1, -1)

    global_stats = stats["global"]
    by_source = stats["by_source"]
    if not feature_sources:
        means = torch.tensor([global_stats[k]["mean"] for k in TASK_COLUMNS], device=device, dtype=dtype)
        stds = torch.tensor([global_stats[k]["std"] for k in TASK_COLUMNS], device=device, dtype=dtype)
        return means.view(1, -1), stds.view(1, -1)

    means = []
    stds = []
    for source_name in feature_sources:
        source_stats = by_source.get(str(source_name), global_stats)
        means.append([source_stats[k]["mean"] for k in TASK_COLUMNS])
        stds.append([source_stats[k]["std"] for k in TASK_COLUMNS])
    return (
        torch.tensor(means, device=device, dtype=dtype),
        torch.tensor(stds, device=device, dtype=dtype),
    )


def normalize_targets(
    targets: torch.Tensor,
    masks: torch.Tensor,
    stats: dict,
    device: torch.device,
    feature_sources: list[str] | None = None,
) -> torch.Tensor:
    means, stds = resolve_row_stats(stats, feature_sources, device=device, dtype=targets.dtype)
    return ((targets - means) / stds.clamp_min(1e-6)) * masks


def apply_ratio_prediction_mode(
    preds: torch.Tensor,
    stats: dict,
    mode: str,
    feature_sources: list[str] | None = None,
) -> torch.Tensor:
    if mode != "derived":
        return preds
    means, stds = resolve_row_stats(stats, feature_sources, preds.device, preds.dtype)
    raw_preds = preds * stds + means
    raw_ratio = raw_preds[:, 0] - raw_preds[:, 1]
    norm_ratio = (raw_ratio - means[:, 2]) / stds[:, 2].clamp_min(1e-6)
    adjusted = preds.clone()
    adjusted[:, 2] = norm_ratio
    return adjusted


def ratio_consistency_loss(
    preds: torch.Tensor,
    masks: torch.Tensor,
    stats: dict,
    feature_sources: list[str] | None = None,
) -> torch.Tensor:
    valid = (masks[:, 0] * masks[:, 1] * masks[:, 2]).to(dtype=preds.dtype)
    if valid.sum() <= 0:
        return preds.new_tensor(0.0)
    means, stds = resolve_row_stats(stats, feature_sources, preds.device, preds.dtype)
    raw_preds = preds * stds + means
    diff = (raw_preds[:, 2] - (raw_preds[:, 0] - raw_preds[:, 1])) / stds[:, 2].clamp_min(1e-6)
    return ((diff ** 2) * valid).sum() / valid.sum().clamp_min(1.0)


def denormalize(preds: np.ndarray, stats: dict, feature_sources: list[str] | None = None) -> np.ndarray:
    if not is_source_specific_stats(stats):
        means = np.asarray([stats[k]["mean"] for k in TASK_COLUMNS], dtype=np.float32)
        stds = np.asarray([stats[k]["std"] for k in TASK_COLUMNS], dtype=np.float32)
        return preds * stds[None, :] + means[None, :]

    global_stats = stats["global"]
    by_source = stats["by_source"]
    if not feature_sources:
        means = np.asarray([global_stats[k]["mean"] for k in TASK_COLUMNS], dtype=np.float32)
        stds = np.asarray([global_stats[k]["std"] for k in TASK_COLUMNS], dtype=np.float32)
        return preds * stds[None, :] + means[None, :]

    means = np.asarray(
        [[by_source.get(str(source), global_stats)[k]["mean"] for k in TASK_COLUMNS] for source in feature_sources],
        dtype=np.float32,
    )
    stds = np.asarray(
        [[by_source.get(str(source), global_stats)[k]["std"] for k in TASK_COLUMNS] for source in feature_sources],
        dtype=np.float32,
    )
    return preds * stds + means


def masked_mse(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = ((preds - targets) ** 2) * masks
    denom = masks.sum()
    if target_weights is not None:
        weights = target_weights.to(device=preds.device, dtype=preds.dtype).view(1, -1)
        loss = loss * weights
        denom = (masks * weights).sum()
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=preds.device, dtype=preds.dtype).view(-1, 1)
        loss = loss * sample_weights
        denom = (masks if target_weights is None else masks * weights) * sample_weights
        denom = denom.sum()
    return loss.sum() / denom.clamp_min(1.0)


def masked_huber(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    delta: float,
    target_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.huber_loss(preds, targets, reduction="none", delta=delta) * masks
    denom = masks.sum()
    if target_weights is not None:
        weights = target_weights.to(device=preds.device, dtype=preds.dtype).view(1, -1)
        loss = loss * weights
        denom = (masks * weights).sum()
    if sample_weights is not None:
        sample_weights = sample_weights.to(device=preds.device, dtype=preds.dtype).view(-1, 1)
        loss = loss * sample_weights
        denom = (masks if target_weights is None else masks * weights) * sample_weights
        denom = denom.sum()
    return loss.sum() / denom.clamp_min(1.0)


def masked_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    *,
    loss_mode: str,
    huber_delta: float,
    huber_mix: float,
    target_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_mode = loss_mode.lower()
    if loss_mode == "mse":
        return masked_mse(
            preds,
            targets,
            masks,
            target_weights=target_weights,
            sample_weights=sample_weights,
        )
    if loss_mode == "huber":
        return masked_huber(
            preds,
            targets,
            masks,
            delta=huber_delta,
            target_weights=target_weights,
            sample_weights=sample_weights,
        )
    if loss_mode == "hybrid":
        mse = masked_mse(
            preds,
            targets,
            masks,
            target_weights=target_weights,
            sample_weights=sample_weights,
        )
        huber = masked_huber(
            preds,
            targets,
            masks,
            delta=huber_delta,
            target_weights=target_weights,
            sample_weights=sample_weights,
        )
        mix = float(huber_mix)
        return (1.0 - mix) * mse + mix * huber
    raise ValueError(f"Unsupported kin_loss_mode: {loss_mode}")


def masked_pearson_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_weights: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-level correlation loss aligned with heldout R2-style selection."""
    losses = []
    weights = []
    for col in range(preds.shape[1]):
        valid = masks[:, col] > 0.5
        if sample_weights is not None:
            valid = valid & (sample_weights > 0)
        if valid.sum() < 3:
            continue
        p = preds[valid, col].float()
        y = targets[valid, col].float()
        if sample_weights is None:
            w = torch.ones_like(p)
        else:
            w = sample_weights[valid].float().clamp_min(0.0)
        if w.sum() <= 0:
            continue
        w = w / w.sum().clamp_min(1e-6)
        p_centered = p - (p * w).sum()
        y_centered = y - (y * w).sum()
        cov = (w * p_centered * y_centered).sum()
        p_var = (w * p_centered.pow(2)).sum()
        y_var = (w * y_centered.pow(2)).sum()
        corr = cov / (p_var.clamp_min(1e-6).sqrt() * y_var.clamp_min(1e-6).sqrt())
        task_weight = target_weights[col].to(device=preds.device, dtype=preds.dtype)
        losses.append((1.0 - corr.to(dtype=preds.dtype)) * task_weight)
        weights.append(task_weight)
    if not losses:
        return preds.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)


def pair_contrastive_loss(extra_feature: torch.Tensor, drfp_feature: torch.Tensor, temperature: float) -> torch.Tensor:
    if extra_feature.shape[0] < 2:
        return extra_feature.new_tensor(0.0)
    a = F.normalize(extra_feature, dim=-1)
    b = F.normalize(drfp_feature, dim=-1)
    logits = a @ b.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def kinetics_pairwise_rank_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    masks: torch.Tensor,
    target_weights: torch.Tensor,
    min_delta: float,
    temperature: float,
    source_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """In-batch ranking auxiliary loss for continuous kinetics labels."""
    if preds.shape[0] < 2:
        return preds.new_tensor(0.0)
    losses = []
    weights = []
    temperature = max(float(temperature), 1e-6)
    for col in range(preds.shape[1]):
        valid = masks[:, col] > 0.5
        if valid.sum() < 2:
            continue
        p = preds[valid, col]
        y = targets[valid, col]
        diff = y[:, None] - y[None, :]
        pair_mask = diff.abs() > float(min_delta)
        if source_ids is not None:
            src = source_ids[valid]
            pair_mask = pair_mask & (src[:, None] == src[None, :])
        pair_mask = torch.triu(pair_mask, diagonal=1)
        if pair_mask.sum() == 0:
            continue
        sign = diff.sign()
        pred_diff = p[:, None] - p[None, :]
        pair_loss = F.softplus(-(sign * pred_diff)[pair_mask] / temperature).mean()
        losses.append(pair_loss * target_weights[col].to(dtype=pair_loss.dtype, device=pair_loss.device))
        weights.append(target_weights[col].to(dtype=pair_loss.dtype, device=pair_loss.device))
    if not losses:
        return preds.new_tensor(0.0)
    return torch.stack(losses).sum() / torch.stack(weights).sum().clamp_min(1e-6)


def anchor_regularization(classifier: nn.Module, anchor_state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Keep the fine-tuned classifier close to the specificity base checkpoint."""
    losses = []
    for name, param in classifier.named_parameters():
        if not param.requires_grad:
            continue
        anchor = anchor_state.get(name)
        if anchor is None:
            continue
        losses.append((param.float() - anchor.to(device=param.device, dtype=torch.float32)).pow(2).mean())
    if not losses:
        first_param = next(classifier.parameters())
        return first_param.new_tensor(0.0)
    return torch.stack(losses).mean()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def squeeze_replay_batch(batch: dict) -> dict:
    squeezed = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == 1:
            squeezed[key] = value.squeeze(0)
        else:
            squeezed[key] = value
    return squeezed


def load_source_specs_json(path: Path | None) -> dict[str, dict] | None:
    """Load source specs and resolve relative feature paths beside the JSON file."""
    if path is None:
        return None
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    base = path.resolve().parent

    def normalize(item: dict) -> dict:
        if not isinstance(item, dict):
            raise TypeError(f"Source specification must be an object, got {type(item)!r}")
        out = dict(item)
        for key in ("embedding_dir", "drfp_path", "bank_path"):
            value = out.get(key)
            if value is None:
                continue
            candidate = Path(value)
            out[key] = candidate if candidate.is_absolute() else (base / candidate).resolve()
        return out

    if isinstance(raw, list):
        specs = {}
        for item in raw:
            name = item.get("name")
            if not name:
                raise ValueError(f"Source spec missing name: {item}")
            specs[str(name)] = normalize(item)
        return specs
    if isinstance(raw, dict):
        return {str(name): normalize(spec) for name, spec in raw.items()}
    raise TypeError(f"Unsupported source spec payload in {path}: {type(raw)!r}")


def load_source_weight_map(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported source weight payload in {path}: {type(raw)!r}")
    return {str(name): float(weight) for name, weight in raw.items()}


def load_task_branch_weight_map(path: Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"Unsupported task branch weight payload in {path}: {type(raw)!r}")
    out: dict[str, dict[str, float]] = {}
    for task_name, branch_map in raw.items():
        if not isinstance(branch_map, dict):
            raise TypeError(f"Task branch weight payload for {task_name} must be a dict, got {type(branch_map)!r}")
        out[str(task_name)] = {str(branch_name): float(weight) for branch_name, weight in branch_map.items()}
    return out


def classifier_forward(classifier: nn.Module, batch: dict) -> dict:
    return classifier(
        batch["s"],
        batch["z"],
        batch["is_ligand"],
        batch["single_mask"],
        drfp=batch["drfp"],
        extra_features=batch["extra_features"],
        context_features=batch.get("context_features"),
        context_mask=batch.get("context_mask"),
        context_scores=batch.get("context_scores"),
        overall_corr=batch.get("overall_corr"),
    )


def apply_drfp_input_policy(
    batch: dict,
    *,
    zero_drfp: bool = False,
    dropout_prob: float = 0.0,
    training: bool = False,
) -> dict:
    """Apply sample-level DRFP removal without changing the underlying dataset."""
    if "drfp" not in batch or batch["drfp"] is None:
        return batch
    if zero_drfp:
        batch["drfp"] = torch.zeros_like(batch["drfp"])
        return batch
    if training and dropout_prob > 0:
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")
        keep = (torch.rand((batch["drfp"].shape[0], 1), device=batch["drfp"].device) >= dropout_prob).to(
            dtype=batch["drfp"].dtype
        )
        batch["drfp"] = batch["drfp"] * keep
    return batch


def build_kinetics_branch_inputs(batch: dict, kin_out: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    rep = kin_out.get("alpha_repr")
    if rep is None:
        rep = torch.cat([kin_out["sample_feature"], kin_out["drfp_feature"], kin_out["extra_feature"]], dim=-1)
    extra_raw = batch["extra_features"]
    if extra_raw.shape[-1] >= 2 * DEFAULT_EXTRA_SPLIT_DIM:
        protein = extra_raw[:, :DEFAULT_EXTRA_SPLIT_DIM]
        pocket = extra_raw[:, DEFAULT_EXTRA_SPLIT_DIM : 2 * DEFAULT_EXTRA_SPLIT_DIM]
    else:
        half = extra_raw.shape[-1] // 2
        protein = extra_raw[:, :half]
        pocket = extra_raw[:, half:]
    ones = torch.ones((rep.shape[0],), device=rep.device, dtype=rep.dtype)
    branch_features = {
        "sample": kin_out["sample_feature"],
        "reaction": kin_out["drfp_feature"],
        "protein": protein,
        "pocket": pocket,
        "substrate": batch["substrate_fp"],
    }
    branch_masks = {
        "sample": ones,
        "reaction": ones,
        "protein": ones,
        "pocket": ones,
        "substrate": batch["substrate_mask"].to(device=rep.device, dtype=rep.dtype),
    }
    if batch.get("row_extra_features") is not None and batch["row_extra_features"].shape[-1] > 0:
        branch_features["row_extra"] = batch["row_extra_features"]
        branch_masks["row_extra"] = batch["row_extra_mask"].to(device=rep.device, dtype=rep.dtype)
    if batch.get("aux_row_extra_features") is not None and batch["aux_row_extra_features"].shape[-1] > 0:
        branch_features["aux_row_extra"] = batch["aux_row_extra_features"]
        branch_masks["aux_row_extra"] = batch["aux_row_extra_mask"].to(device=rep.device, dtype=rep.dtype)
    return rep, branch_features, branch_masks


def build_full_feature_input(batch: dict, kin_out: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Match the strongest standalone MLP feature contract inside the multitask checkpoint."""
    pieces = [
        kin_out["sample_feature"],
        kin_out["drfp_feature"],
        kin_out["extra_feature"],
        batch["extra_features"],
        batch["drfp"],
        batch["substrate_fp"],
    ]
    if batch.get("row_extra_features") is not None and batch["row_extra_features"].shape[-1] > 0:
        pieces.append(batch["row_extra_features"])
    full_feature = torch.cat(pieces, dim=-1)
    mask = torch.ones((full_feature.shape[0],), device=full_feature.device, dtype=full_feature.dtype)
    return full_feature, mask


def forward_kinetics_heads(heads: nn.Module, batch: dict, kin_out: dict | None) -> torch.Tensor:
    if getattr(heads, "mode", None) == "row_extra_only":
        branch_features = {"row_extra": batch["row_extra_features"]}
        branch_masks = {"row_extra": batch["row_extra_mask"].to(device=batch["row_extra_features"].device, dtype=batch["row_extra_features"].dtype)}
        return heads(None, batch["source_ids"], branch_features=branch_features, branch_masks=branch_masks)
    if kin_out is None:
        raise ValueError("kin_out is required unless heads.mode == row_extra_only")
    if getattr(heads, "mode", None) == "full_feature_mlp":
        full_feature, full_mask = build_full_feature_input(batch, kin_out)
        return heads(
            None,
            batch["source_ids"],
            branch_features={"full_feature": full_feature},
            branch_masks={"full_feature": full_mask},
        )
    rep, branch_features, branch_masks = build_kinetics_branch_inputs(batch, kin_out)
    if getattr(heads, "mode", None) in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
        return heads(None, batch["source_ids"], branch_features=branch_features, branch_masks=branch_masks)
    return heads(rep, batch["source_ids"])


def load_full_feature_stats(path: Path | None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if path is None:
        return None, None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
        raise ValueError(f"{path} must be a checkpoint/dict containing mean and std")
    mean = torch.as_tensor(payload["mean"], dtype=torch.float32).view(1, -1)
    std = torch.as_tensor(payload["std"], dtype=torch.float32).view(1, -1).clamp_min(1e-6)
    if tuple(mean.shape) != tuple(std.shape):
        raise ValueError(f"{path} mean/std shape mismatch: {tuple(mean.shape)} vs {tuple(std.shape)}")
    return mean, std


def init_full_feature_mlp_from_standalone(
    heads: nn.Module,
    path: Path | None,
    *,
    label_stats: dict | None = None,
    label_stats_source: str = "public_strict_iteration1_fulltrue_msa",
) -> dict:
    """Initialize full_feature_mlp from the standalone two-task kinetics MLP."""
    if path is None:
        return {}
    if not hasattr(heads, "full_feature_mlp"):
        raise ValueError("--kin_init_full_feature_mlp_checkpoint requires kin_head_mode=full_feature_mlp")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source_state = payload.get("model") if isinstance(payload, dict) else None
    if not isinstance(source_state, dict):
        raise ValueError(f"{path} does not contain a standalone MLP 'model' state dict")
    target = heads.full_feature_mlp
    target_state = target.state_dict()
    output_mean = None
    output_std = None
    if label_stats:
        source_stats = (label_stats.get("by_source") or {}).get(label_stats_source) or label_stats.get("global")
        if source_stats:
            output_mean = torch.tensor(
                [source_stats[name]["mean"] for name in TASK_COLUMNS[:2]],
                dtype=torch.float32,
            )
            output_std = torch.tensor(
                [source_stats[name]["std"] for name in TASK_COLUMNS[:2]],
                dtype=torch.float32,
            ).clamp_min(1e-6)
    copied: list[str] = []
    partial: list[str] = []
    skipped: dict[str, dict[str, list[int]]] = {}
    for name, source_value in source_state.items():
        if name not in target_state:
            continue
        target_value = target_state[name]
        if tuple(source_value.shape) == tuple(target_value.shape):
            target_state[name] = source_value.detach().clone().to(dtype=target_value.dtype)
            copied.append(name)
            continue
        if (
            name in {"output.1.weight", "output.1.bias"}
            and source_value.ndim == target_value.ndim
            and source_value.shape[0] <= target_value.shape[0]
            and tuple(source_value.shape[1:]) == tuple(target_value.shape[1:])
        ):
            source_copy = source_value.detach().clone().to(dtype=target_value.dtype)
            if output_mean is not None and output_std is not None:
                mean = output_mean.to(dtype=target_value.dtype, device=source_copy.device)
                std = output_std.to(dtype=target_value.dtype, device=source_copy.device)
                if name == "output.1.weight":
                    source_copy = source_copy / std.view(-1, 1)
                else:
                    source_copy = (source_copy - mean) / std
            updated = target_value.detach().clone()
            updated[: source_value.shape[0]] = source_copy
            target_state[name] = updated
            partial.append(name)
            continue
        skipped[name] = {"source": list(source_value.shape), "target": list(target_value.shape)}
    target.load_state_dict(target_state, strict=True)
    return {
        "path": str(path),
        "copied_count": len(copied),
        "partial_count": len(partial),
        "copied": copied[:50],
        "partial": partial,
        "skipped_shape": skipped,
        "skipped_shape_count": len(skipped),
        "label_stats_source": label_stats_source if output_mean is not None else None,
        "label_output_transform": output_mean is not None and output_std is not None,
    }


@torch.no_grad()
def evaluate_kinetics(
    classifier,
    heads,
    loader,
    stats,
    device,
    amp_dtype,
    ratio_prediction_mode: str,
    *,
    zero_drfp: bool = False,
) -> dict:
    classifier.eval()
    heads.eval()
    pred_rows = []
    target_rows = []
    mask_rows = []
    feature_sources = []
    for batch in loader:
        batch = move_batch(batch, device)
        batch = apply_drfp_input_policy(batch, zero_drfp=zero_drfp)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
            out = None if getattr(heads, "mode", None) == "row_extra_only" else classifier_forward(classifier, batch)
            preds = apply_ratio_prediction_mode(
                forward_kinetics_heads(heads, batch, out),
                stats,
                ratio_prediction_mode,
                feature_sources=batch["feature_sources"],
            )
        pred_rows.append(preds.float().cpu().numpy())
        target_rows.append(batch["targets"].float().cpu().numpy())
        mask_rows.append(batch["masks"].float().cpu().numpy())
        feature_sources.extend(batch["feature_sources"])

    pred = denormalize(np.concatenate(pred_rows, axis=0), stats, feature_sources=feature_sources)
    target = np.concatenate(target_rows, axis=0)
    masks = np.concatenate(mask_rows, axis=0)
    metrics = {}
    for i, name in enumerate(TASK_COLUMNS):
        mask = masks[:, i] > 0.5
        if mask.sum() < 3:
            continue
        y_true = target[mask, i]
        y_pred = pred[mask, i]
        metrics[f"{name}_n"] = int(mask.sum())
        metrics[f"{name}_rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        metrics[f"{name}_mae"] = float(np.mean(np.abs(y_pred - y_true)))
        try:
            metrics[f"{name}_pearson"] = float(pearsonr(y_true, y_pred).statistic)
        except Exception:
            metrics[f"{name}_pearson"] = float("nan")
        try:
            metrics[f"{name}_spearman"] = float(spearmanr(y_true, y_pred).statistic)
        except Exception:
            metrics[f"{name}_spearman"] = float("nan")
    if feature_sources:
        source_arr = np.asarray(feature_sources, dtype=object)
        for source_name in sorted(set(feature_sources)):
            source_mask_base = source_arr == source_name
            prefix = f"source_{str(source_name).replace('-', '_')}"
            for i, name in enumerate(TASK_COLUMNS):
                mask = source_mask_base & (masks[:, i] > 0.5)
                if mask.sum() < 3:
                    continue
                y_true = target[mask, i]
                y_pred = pred[mask, i]
                metrics[f"{prefix}_{name}_n"] = int(mask.sum())
                metrics[f"{prefix}_{name}_rmse"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
                metrics[f"{prefix}_{name}_mae"] = float(np.mean(np.abs(y_pred - y_true)))
                try:
                    metrics[f"{prefix}_{name}_pearson"] = float(pearsonr(y_true, y_pred).statistic)
                except Exception:
                    metrics[f"{prefix}_{name}_pearson"] = float("nan")
                try:
                    metrics[f"{prefix}_{name}_spearman"] = float(spearmanr(y_true, y_pred).statistic)
                except Exception:
                    metrics[f"{prefix}_{name}_spearman"] = float("nan")
    return metrics


def make_replay_loader(args) -> DataLoader | None:
    if args.specificity_weight <= 0 and args.distill_weight <= 0:
        return None
    dataset = PairwiseReactionDataset(
        json_dir=str(args.spec_json_dir),
        embedding_dir=str(args.spec_embedding_dir),
        csv_path=str(args.spec_csv_path),
        drfp_path=str(args.spec_drfp_path),
        max_seq_length=args.spec_max_seq_length,
        batch_size=args.spec_batch_size,
        batching_strategy=args.spec_batching_strategy,
        sampled_batches_per_reaction=args.spec_sampled_batches_per_reaction,
        random_seed=args.seed,
        external_feature_banks=[
            {"path": str(args.spec_train_bank), "include_z": True, "name": "protein_pocket"}
        ],
        metadata_cache_path=str(args.spec_replay_cache_path) if args.spec_replay_cache_path else None,
        batch_cache_path=str(args.spec_batch_cache_path) if args.spec_batch_cache_path else None,
    )
    replay_loader_kwargs = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": args.spec_num_workers,
        "pin_memory": args.pin_memory,
        "persistent_workers": bool(args.persistent_workers and args.spec_num_workers > 0),
    }
    if args.spec_num_workers > 0 and args.prefetch_factor > 0:
        replay_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **replay_loader_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train_csv", type=Path, required=True)
    parser.add_argument("--valid_csv", type=Path, required=True)
    parser.add_argument("--kin_embedding_dir", type=Path, required=True)
    parser.add_argument("--kin_drfp_path", type=Path, required=True)
    parser.add_argument("--kin_train_bank", type=Path, required=True)
    parser.add_argument("--kin_valid_bank", type=Path, required=True)
    parser.add_argument("--kin_substrate_fp_path", type=Path, default=None)
    parser.add_argument("--kin_train_row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--kin_valid_row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--kin_train_aux_row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--kin_valid_aux_row_extra_features_npz", type=Path, default=None)
    parser.add_argument("--kin_train_teacher_targets_npz", type=Path, default=None)
    parser.add_argument("--kin_valid_teacher_targets_npz", type=Path, default=None)
    parser.add_argument("--kin_train_sources_json", type=Path, default=None)
    parser.add_argument("--kin_valid_sources_json", type=Path, default=None)
    parser.add_argument("--source_loss_weights_json", type=Path, default=None)
    parser.add_argument("--task_branch_weight_json", type=Path, default=None)
    parser.add_argument("--kin_source_column", type=str, default="feature_source")
    parser.add_argument("--kin_default_source", type=str, default="default")
    parser.add_argument("--spec_json_dir", type=Path, required=True)
    parser.add_argument("--spec_embedding_dir", type=Path, required=True)
    parser.add_argument("--spec_csv_path", type=Path, required=True)
    parser.add_argument("--spec_drfp_path", type=Path, required=True)
    parser.add_argument("--spec_train_bank", type=Path, required=True)
    parser.add_argument("--spec_replay_cache_path", type=Path, default=None)
    parser.add_argument("--spec_batch_cache_path", type=Path, default=None)
    parser.add_argument("--kin_train_cache_path", type=Path, default=None)
    parser.add_argument("--kin_valid_cache_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_save_every",
        type=int,
        default=1,
        help="Save full multitask checkpoints every N epochs. Use 0 to only write metrics.",
    )
    parser.add_argument(
        "--save_classifier_epoch_state",
        action="store_true",
        help="Also save classifier-only model_epoch*.pt files. Disabled by default to avoid wasting disk.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--spec_batch_size", type=int, default=32)
    parser.add_argument("--spec_batching_strategy", type=str, default="reaction_sampled")
    parser.add_argument("--spec_sampled_batches_per_reaction", type=int, default=1)
    parser.add_argument("--spec_max_seq_length", type=int, default=700)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--head_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.03)
    parser.add_argument("--kinetics_weight", type=float, default=0.1)
    parser.add_argument("--specificity_weight", type=float, default=1.0)
    parser.add_argument(
        "--specificity_interval",
        type=int,
        default=1,
        help="Apply specificity replay once every N optimization steps. 1 means every step.",
    )
    parser.add_argument(
        "--specificity_head_mode",
        type=str,
        default="classifier_logits",
        choices=["classifier_logits", "alpha_repr"],
        help="Use the original classifier logits or a shared-alpha-representation specificity head for replay.",
    )
    parser.add_argument("--specificity_alpha_repr_dim", type=int, default=1152)
    parser.add_argument("--specificity_alpha_hidden_size", type=int, default=384)
    parser.add_argument("--specificity_alpha_layers", type=int, default=2)
    parser.add_argument("--specificity_alpha_dropout", type=float, default=0.0)
    parser.add_argument(
        "--specificity_alpha_head_lr",
        type=float,
        default=0.0,
        help="Learning rate for alpha_repr specificity head; 0 follows --head_lr.",
    )
    parser.add_argument(
        "--reset_specificity_alpha_head",
        action="store_true",
        help="Ignore specificity_alpha_head from a multitask checkpoint and reinitialize it.",
    )
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--kin_rank_weight", type=float, default=0.0)
    parser.add_argument("--kin_rank_min_delta", type=float, default=0.15)
    parser.add_argument("--kin_rank_temperature", type=float, default=0.5)
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--anchor_weight", type=float, default=0.0)
    parser.add_argument(
        "--classifier_train_mode",
        type=str,
        default="all",
        choices=["all", "none", "top", "extra", "fusion", "classifier_only", "logit_only", "lora"],
    )
    parser.add_argument("--classifier_lora_rank", type=int, default=0)
    parser.add_argument("--classifier_lora_alpha", type=float, default=16.0)
    parser.add_argument("--classifier_lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--classifier_lora_targets",
        type=str,
        nargs="+",
        default=["input_proj", "transformer", "drfp_mlp", "extra_proj", "extra_fusion", "extra_branch"],
        help="Linear module name substrings where LoRA adapters are inserted.",
    )
    parser.add_argument(
        "--classifier_warmup_epochs",
        type=int,
        default=0,
        help="Keep the classifier frozen for the first N epochs while training only the kinetics heads.",
    )
    parser.add_argument("--kin_head_hidden_size", type=int, default=384)
    parser.add_argument("--kin_head_layers", type=int, default=2)
    parser.add_argument(
        "--kin_head_mode",
        type=str,
        default="shared",
        choices=[
            "shared",
            "separate",
            "shared_separate",
            "task_branch_gate",
            "task_interaction_gate",
            "task_moe_gate",
            "row_extra_only",
            "full_feature_mlp",
        ],
    )
    parser.add_argument(
        "--kin_full_feature_stats_checkpoint",
        type=Path,
        default=None,
        help="Checkpoint containing mean/std used to standardize full_feature_mlp input.",
    )
    parser.add_argument(
        "--kin_init_full_feature_mlp_checkpoint",
        type=Path,
        default=None,
        help="Standalone AlphaEnzyme kinetics MLP checkpoint used to initialize full_feature_mlp weights.",
    )
    parser.add_argument(
        "--kin_moe_num_experts",
        type=int,
        default=4,
        help="Number of per-task experts used when kin_head_mode=task_moe_gate.",
    )
    parser.add_argument(
        "--kin_moe_residual_num_experts",
        type=int,
        default=0,
        help="Number of zero-initialized MoE residual experts added to branch-based kinetics heads.",
    )
    parser.add_argument(
        "--kin_structured_residual_mode",
        type=str,
        default="none",
        choices=["none", "concat", "pairwise"],
        help="Add a zero-initialized residual head over structural/reaction/substrate branch interactions.",
    )
    parser.add_argument(
        "--kin_structured_residual_network",
        type=str,
        default="mlp",
        choices=["mlp", "resnet", "gated_resnet", "standalone_resnet", "linear", "chunk_transformer"],
        help="Network used by the structured interaction residual head.",
    )
    parser.add_argument(
        "--kin_structured_residual_hidden_size",
        type=int,
        default=0,
        help="Hidden size for the structured residual head; 0 follows kin_head_hidden_size.",
    )
    parser.add_argument(
        "--kin_structured_residual_layers",
        type=int,
        default=0,
        help="Layer count for the structured residual head; 0 follows kin_head_layers.",
    )
    parser.add_argument(
        "--kin_structured_residual_output_weights",
        type=float,
        nargs=3,
        default=[0.0, 1.0, 0.0],
        help="Per-task output scales for the structured residual head.",
    )
    parser.add_argument("--kin_source_head_mode", type=str, default="none", choices=["none", "affine"])
    parser.add_argument(
        "--kin_row_extra_mode",
        type=str,
        default="branch",
        choices=["branch", "residual", "none"],
        help="How row-level expanded features enter the kinetics head.",
    )
    parser.add_argument(
        "--kin_row_extra_residual_mode",
        type=str,
        default="plain",
        choices=["plain", "interaction"],
        help="How residual row-level features are transformed before adding to kinetics outputs.",
    )
    parser.add_argument(
        "--kin_head_train_mode",
        type=str,
        default="all",
        choices=[
            "all",
            "none",
            "row_extra_residual",
            "row_extra_residual_source",
            "aux_row_extra_residual",
            "aux_row_extra_residual_source",
            "row_aux_residual",
            "row_aux_residual_source",
            "moe_residual",
            "moe_residual_source",
            "structured_residual",
            "structured_residual_source",
            "structured_aux_residual",
            "structured_aux_residual_source",
            "row_struct_aux_residual",
            "row_struct_aux_residual_source",
            "second_aux_residual",
            "second_aux_residual_source",
            "source",
            "gate_source",
            "task_heads_source",
        ],
        help="Restrict trainable kinetics-head parameters for conservative fine-tuning.",
    )
    parser.add_argument("--reset_heads", action="store_true", help="Ignore kinetics heads from a multitask checkpoint and reinitialize them.")
    parser.add_argument(
        "--reset_row_extra_residual",
        action="store_true",
        help="When resuming a kinetics head, reinitialize only the row-extra residual branch.",
    )
    parser.add_argument(
        "--reset_aux_row_extra_residual",
        action="store_true",
        help="When resuming a kinetics head, reinitialize only the auxiliary row-extra residual branch.",
    )
    parser.add_argument(
        "--reset_structured_residual",
        action="store_true",
        help="When resuming a kinetics head, reinitialize only the structured residual branch.",
    )
    parser.add_argument("--reset_classifier", action="store_true", help="Use the checkpoint only as an architecture template and randomly reinitialize classifier weights.")
    parser.add_argument("--ratio_prediction_mode", type=str, default="independent", choices=["independent", "derived"])
    parser.add_argument("--ratio_consistency_weight", type=float, default=0.0)
    parser.add_argument("--target_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--kin_drfp_dropout_prob",
        type=float,
        default=0.0,
        help="Sample-level probability of replacing kinetics DRFP with zeros during training.",
    )
    parser.add_argument(
        "--kin_zero_drfp_input",
        action="store_true",
        help="Always replace kinetics DRFP with zeros during training.",
    )
    parser.add_argument(
        "--spec_zero_drfp_input",
        action="store_true",
        help="Replace specificity replay DRFP with zeros during training.",
    )
    parser.add_argument(
        "--eval_zero_drfp_input",
        action="store_true",
        help="Replace kinetics validation DRFP with zeros during in-training evaluation.",
    )
    parser.add_argument(
        "--drop_loaded_posthoc_raw_calibration",
        action="store_true",
        help="Do not copy posthoc raw prediction calibration from the loaded multitask checkpoint.",
    )
    parser.add_argument("--kin_teacher_weight", type=float, default=0.0)
    parser.add_argument("--kin_teacher_target_weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument(
        "--kin_label_stats_mode",
        type=str,
        default="train",
        choices=["train", "checkpoint"],
        help="Use newly computed train-set label stats, or preserve stats from a multitask checkpoint.",
    )
    parser.add_argument(
        "--kin_target_columns",
        type=str,
        nargs=3,
        default=None,
        help=(
            "Optional CSV columns used as kinetics targets in TASK_COLUMNS order. "
            "Example: official_log10kcat_max official_log10km_mean log_kcat_over_km_derived."
        ),
    )
    parser.add_argument(
        "--kin_base_anchor_weight",
        type=float,
        default=0.0,
        help="Keep kinetics predictions close to the loaded checkpoint while learning residual corrections.",
    )
    parser.add_argument(
        "--kin_base_anchor_target_weights",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Per-task weights for the loaded-checkpoint kinetics prediction anchor.",
    )
    parser.add_argument("--kin_loss_mode", type=str, default="mse", choices=["mse", "huber", "hybrid"])
    parser.add_argument("--kin_huber_delta", type=float, default=0.5)
    parser.add_argument("--kin_huber_mix", type=float, default=0.5)
    parser.add_argument("--kin_pearson_weight", type=float, default=0.0)
    parser.add_argument("--kin_pearson_target_weights", type=float, nargs=3, default=[1.0, 1.0, 0.0])
    parser.add_argument(
        "--kin_row_extra_network",
        type=str,
        default="mlp",
        choices=["mlp", "resnet", "gated_resnet", "standalone_resnet", "linear", "chunk_transformer"],
        help="Network used for row-extra-only or row-extra residual branches.",
    )
    parser.add_argument(
        "--kin_aux_row_extra_network",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "resnet",
            "gated_resnet",
            "linear",
            "chunk_transformer",
            "per_task_mlp",
            "per_task_resnet",
            "per_task_gated_resnet",
            "per_task_linear",
        ],
        help="Network used for the auxiliary row-extra residual branch.",
    )
    parser.add_argument(
        "--kin_aux_row_extra_residual_mode",
        type=str,
        default="plain",
        choices=["plain", "prediction", "blend", "scaled", "prediction_scaled"],
        help="How auxiliary row-extra residuals are conditioned before adding to kinetics outputs.",
    )
    parser.add_argument(
        "--kin_aux_row_extra_blend_weights",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Per-task blend weights used when --kin_aux_row_extra_residual_mode=blend.",
    )
    parser.add_argument(
        "--kin_enable_second_aux_row_extra_residual",
        action="store_true",
        help="Add a zero-initialized second auxiliary row-extra residual branch without replacing the checkpoint branch.",
    )
    parser.add_argument(
        "--kin_second_aux_row_extra_network",
        type=str,
        default="mlp",
        choices=[
            "mlp",
            "resnet",
            "gated_resnet",
            "linear",
            "chunk_transformer",
            "per_task_mlp",
            "per_task_resnet",
            "per_task_gated_resnet",
            "per_task_linear",
        ],
        help="Network used for the second auxiliary row-extra residual branch.",
    )
    parser.add_argument(
        "--kin_second_aux_row_extra_residual_mode",
        type=str,
        default="plain",
        choices=["plain", "prediction"],
        help="Whether the second auxiliary residual also receives current kinetics predictions.",
    )
    parser.add_argument(
        "--kin_second_aux_row_extra_weights",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Per-task additive scales for the second auxiliary row-extra residual branch.",
    )
    parser.add_argument(
        "--kin_second_aux_row_extra_hidden_size",
        type=int,
        default=0,
        help="Hidden size for the second auxiliary residual branch; 0 follows auxiliary hidden size.",
    )
    parser.add_argument(
        "--kin_second_aux_row_extra_layers",
        type=int,
        default=0,
        help="Layer count for the second auxiliary residual branch; 0 follows auxiliary layer count.",
    )
    parser.add_argument(
        "--kin_extra_dropout",
        type=float,
        default=None,
        help="Dropout for row-extra residual networks; defaults to checkpoint dropout.",
    )
    parser.add_argument(
        "--kin_aux_row_extra_hidden_size",
        type=int,
        default=0,
        help="Hidden size for the auxiliary row-extra residual branch; 0 follows kin_head_hidden_size.",
    )
    parser.add_argument(
        "--kin_aux_row_extra_layers",
        type=int,
        default=0,
        help="Layer count for the auxiliary row-extra residual branch; 0 follows kin_head_layers.",
    )
    parser.add_argument(
        "--kin_row_extra_only_fast",
        action="store_true",
        help="Skip loading structural tensors for kinetics batches when the kinetics head uses row_extra_only.",
    )
    parser.add_argument("--source_specific_stats", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--spec_num_workers", type=int, default=2)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--persistent_workers", action="store_true")
    args = parser.parse_args()

    torch_num_threads = os.environ.get("TORCH_NUM_THREADS")
    if torch_num_threads:
        torch.set_num_threads(max(1, int(torch_num_threads)))

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_source_specs = load_source_specs_json(args.kin_train_sources_json)
    valid_source_specs = load_source_specs_json(args.kin_valid_sources_json)
    source_loss_weights = load_source_weight_map(args.source_loss_weights_json)
    task_branch_weight_map = load_task_branch_weight_map(args.task_branch_weight_json)
    train_set = KineticsDataset(
        args.train_csv,
        args.kin_embedding_dir,
        args.kin_drfp_path,
        args.kin_train_bank,
        substrate_fp_path=args.kin_substrate_fp_path,
        row_extra_features_path=args.kin_train_row_extra_features_npz,
        aux_row_extra_features_path=args.kin_train_aux_row_extra_features_npz,
        teacher_targets_path=args.kin_train_teacher_targets_npz,
        source_specs=train_source_specs,
        source_column=args.kin_source_column,
        default_source=args.kin_default_source,
        cache_path=args.kin_train_cache_path,
        skip_structural_features=args.kin_row_extra_only_fast,
        target_columns=args.kin_target_columns,
    )
    valid_set = KineticsDataset(
        args.valid_csv,
        args.kin_embedding_dir,
        args.kin_drfp_path,
        args.kin_valid_bank,
        substrate_fp_path=args.kin_substrate_fp_path,
        row_extra_features_path=args.kin_valid_row_extra_features_npz,
        aux_row_extra_features_path=args.kin_valid_aux_row_extra_features_npz,
        teacher_targets_path=args.kin_valid_teacher_targets_npz,
        source_specs=valid_source_specs,
        source_column=args.kin_source_column,
        default_source=args.kin_default_source,
        cache_path=args.kin_valid_cache_path,
        skip_structural_features=args.kin_row_extra_only_fast,
        target_columns=args.kin_target_columns,
    )
    train_stats = compute_label_stats(train_set, source_specific=args.source_specific_stats)

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "collate_fn": kinetics_collate,
        "persistent_workers": bool(args.persistent_workers and args.num_workers > 0),
    }
    if args.num_workers > 0 and args.prefetch_factor > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_set, shuffle=False, **loader_kwargs)
    replay_loader = make_replay_loader(args)

    init_ckpt = load_init_checkpoint(
        args.checkpoint,
        default_dropout=args.dropout,
        default_head_hidden_size=args.kin_head_hidden_size,
        default_head_layers=args.kin_head_layers,
        default_head_mode=args.kin_head_mode,
    )
    stats = train_stats
    if args.kin_label_stats_mode == "checkpoint":
        checkpoint_stats = init_ckpt.run_meta.get("label_stats") if init_ckpt.run_meta else None
        if not checkpoint_stats:
            raise ValueError("--kin_label_stats_mode checkpoint requires a multitask checkpoint with run_meta.label_stats")
        stats = checkpoint_stats
    (args.output_dir / "label_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    classifier = build_classifier(init_ckpt.dropout, init_ckpt.classifier_state).to(device)
    if args.reset_classifier:
        reset_module_parameters(classifier)
        missing, unexpected = [], []
    else:
        missing, unexpected = classifier.load_state_dict(init_ckpt.classifier_state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    lora_layer_names = inject_lora_layers(
        classifier,
        rank=args.classifier_lora_rank,
        alpha=args.classifier_lora_alpha,
        dropout=args.classifier_lora_dropout,
        target_keywords=tuple(args.classifier_lora_targets),
    )
    anchor_state = {name: param.detach().clone() for name, param in classifier.named_parameters()}
    head_hidden_size = args.kin_head_hidden_size if args.reset_heads else init_ckpt.kin_head_hidden_size
    head_layers = args.kin_head_layers if args.reset_heads else init_ckpt.kin_head_layers
    head_mode = args.kin_head_mode if args.reset_heads else init_ckpt.kin_head_mode
    head_branch_dims = None
    if head_mode == "row_extra_only":
        if train_set.row_extra_feature_dim <= 0:
            raise ValueError("kin_head_mode=row_extra_only requires --kin_train_row_extra_features_npz")
        head_branch_dims = {"row_extra": train_set.row_extra_feature_dim}
    elif head_mode == "full_feature_mlp":
        head_branch_dims = {
            "full_feature": (
                DEFAULT_S_DIM
                + DEFAULT_S_DIM
                + DEFAULT_S_DIM
                + 2 * DEFAULT_EXTRA_SPLIT_DIM
                + 2048
                + train_set.substrate_fp_dim
                + train_set.row_extra_feature_dim
            )
        }
    elif head_mode in {"task_branch_gate", "task_interaction_gate", "task_moe_gate"}:
        head_branch_dims = {
            "sample": 384,
            "reaction": 384,
            "protein": DEFAULT_EXTRA_SPLIT_DIM,
            "pocket": DEFAULT_EXTRA_SPLIT_DIM,
            "substrate": train_set.substrate_fp_dim,
        }
        if train_set.row_extra_feature_dim > 0 and args.kin_row_extra_mode == "branch":
            head_branch_dims["row_extra"] = train_set.row_extra_feature_dim
    row_extra_residual_dim = train_set.row_extra_feature_dim if args.kin_row_extra_mode == "residual" else 0
    full_feature_mean, full_feature_std = load_full_feature_stats(args.kin_full_feature_stats_checkpoint)
    heads = KineticsHeads(
        hidden_size=head_hidden_size,
        num_layers=head_layers,
        dropout=init_ckpt.dropout,
        mode=head_mode,
        source_names=train_set.source_names,
        source_mode=args.kin_source_head_mode,
        branch_dims=head_branch_dims,
        task_branch_weight_map=task_branch_weight_map,
        row_extra_residual_dim=row_extra_residual_dim,
        row_extra_residual_mode=args.kin_row_extra_residual_mode,
        aux_row_extra_residual_dim=train_set.aux_row_extra_feature_dim,
        row_extra_network=args.kin_row_extra_network,
        aux_row_extra_network=args.kin_aux_row_extra_network,
        aux_row_extra_residual_mode=args.kin_aux_row_extra_residual_mode,
        aux_row_extra_blend_weights=args.kin_aux_row_extra_blend_weights,
        second_aux_row_extra_residual_dim=(
            train_set.aux_row_extra_feature_dim if args.kin_enable_second_aux_row_extra_residual else 0
        ),
        second_aux_row_extra_network=args.kin_second_aux_row_extra_network,
        second_aux_row_extra_residual_mode=args.kin_second_aux_row_extra_residual_mode,
        second_aux_row_extra_weights=args.kin_second_aux_row_extra_weights,
        extra_dropout=args.kin_extra_dropout,
        aux_row_extra_hidden_size=args.kin_aux_row_extra_hidden_size or None,
        aux_row_extra_layers=args.kin_aux_row_extra_layers or None,
        second_aux_row_extra_hidden_size=args.kin_second_aux_row_extra_hidden_size or None,
        second_aux_row_extra_layers=args.kin_second_aux_row_extra_layers or None,
        moe_num_experts=args.kin_moe_num_experts,
        moe_residual_num_experts=args.kin_moe_residual_num_experts,
        structured_residual_mode=args.kin_structured_residual_mode,
        structured_residual_network=args.kin_structured_residual_network,
        structured_residual_hidden_size=args.kin_structured_residual_hidden_size or None,
        structured_residual_layers=args.kin_structured_residual_layers or None,
        structured_residual_output_weights=args.kin_structured_residual_output_weights,
        full_feature_mean=full_feature_mean,
        full_feature_std=full_feature_std,
    ).to(device)
    resumed_kinetics_heads = False
    kinetics_head_missing: list[str] = []
    kinetics_head_unexpected: list[str] = []
    kinetics_head_skipped_shape: dict[str, dict[str, list[int]]] = {}
    if init_ckpt.kinetics_head_state is not None and not args.reset_heads:
        kinetics_state = init_ckpt.kinetics_head_state
        reset_prefixes: list[str] = []
        if args.reset_row_extra_residual:
            reset_prefixes.extend([
                "row_extra_residual",
                "row_extra_residual_proj",
                "row_extra_residual_interaction",
            ])
        if args.reset_aux_row_extra_residual:
            reset_prefixes.append("aux_row_extra_residual")
        if args.reset_structured_residual:
            reset_prefixes.append("structured_residual")
        if reset_prefixes:
            kinetics_state = filter_state_prefixes(kinetics_state, tuple(reset_prefixes))
        head_missing, head_unexpected, head_skipped_shape = load_shape_compatible_state(
            heads,
            kinetics_state,
        )
        kinetics_head_missing = head_missing
        kinetics_head_unexpected = head_unexpected
        kinetics_head_skipped_shape = head_skipped_shape
        if head_unexpected:
            print(
                "Warning: skipped incompatible kinetics head checkpoint entries: "
                f"{head_unexpected}",
                flush=True,
            )
        resumed_kinetics_heads = True
    full_feature_mlp_init = init_full_feature_mlp_from_standalone(
        heads,
        args.kin_init_full_feature_mlp_checkpoint,
        label_stats=stats,
    )
    set_aux_row_extra_blend_weights(heads, args.kin_aux_row_extra_blend_weights)

    specificity_alpha_head = None
    specificity_alpha_head_missing: list[str] = []
    specificity_alpha_head_unexpected: list[str] = []
    specificity_alpha_head_skipped_shape: dict[str, dict[str, list[int]]] = {}
    if args.specificity_head_mode == "alpha_repr":
        specificity_alpha_head = AlphaReprSpecificityHead(
            input_dim=args.specificity_alpha_repr_dim,
            hidden_size=args.specificity_alpha_hidden_size,
            num_layers=args.specificity_alpha_layers,
            dropout=args.specificity_alpha_dropout,
        ).to(device)
        if init_ckpt.specificity_alpha_head_state is not None and not args.reset_specificity_alpha_head:
            (
                specificity_alpha_head_missing,
                specificity_alpha_head_unexpected,
                specificity_alpha_head_skipped_shape,
            ) = load_shape_compatible_state(
                specificity_alpha_head,
                init_ckpt.specificity_alpha_head_state,
            )

    if args.kin_head_train_mode != "all":
        for name, param in heads.named_parameters():
            trainable = False
            if args.kin_head_train_mode == "none":
                param.requires_grad_(False)
                continue
            if args.kin_head_train_mode in {
                "row_extra_residual",
                "row_extra_residual_source",
                "row_aux_residual",
                "row_aux_residual_source",
                "row_struct_aux_residual",
                "row_struct_aux_residual_source",
            }:
                trainable = name.startswith("row_extra_residual")
            if args.kin_head_train_mode in {
                "aux_row_extra_residual",
                "aux_row_extra_residual_source",
                "row_aux_residual",
                "row_aux_residual_source",
                "structured_aux_residual",
                "structured_aux_residual_source",
                "row_struct_aux_residual",
                "row_struct_aux_residual_source",
            }:
                trainable = name.startswith("aux_row_extra_residual")
            if args.kin_head_train_mode in {"second_aux_residual", "second_aux_residual_source"}:
                trainable = name.startswith("second_aux_row_extra_residual")
            if args.kin_head_train_mode in {"moe_residual", "moe_residual_source"}:
                trainable = trainable or name.startswith("task_moe_residual")
            if args.kin_head_train_mode in {
                "structured_residual",
                "structured_residual_source",
                "structured_aux_residual",
                "structured_aux_residual_source",
                "row_struct_aux_residual",
                "row_struct_aux_residual_source",
            }:
                trainable = trainable or name.startswith("structured_residual")
            if args.kin_head_train_mode in {
                "row_extra_residual_source",
                "aux_row_extra_residual_source",
                "row_aux_residual_source",
                "moe_residual_source",
                "structured_residual_source",
                "structured_aux_residual_source",
                "row_struct_aux_residual_source",
                "second_aux_residual_source",
            }:
                trainable = trainable or name.startswith(("source_scale", "source_bias"))
            if args.kin_head_train_mode == "source":
                trainable = name.startswith(("source_scale", "source_bias"))
            if args.kin_head_train_mode == "gate_source":
                trainable = name.startswith(
                    (
                        "task_gate",
                        "task_expert_gate",
                        "task_branch_bias",
                        "source_scale",
                        "source_bias",
                    )
                )
            if args.kin_head_train_mode == "task_heads_source":
                trainable = name.startswith(
                    (
                        "task_heads",
                        "task_experts",
                        "task_gate",
                        "task_expert_gate",
                        "task_branch_bias",
                        "source_scale",
                        "source_bias",
                    )
                )
            param.requires_grad_(trainable)

    base_anchor_heads = None
    if args.kin_base_anchor_weight > 0:
        base_anchor_heads = copy.deepcopy(heads).to(device)
        base_anchor_heads.eval()
        for param in base_anchor_heads.parameters():
            param.requires_grad_(False)

    teacher = None
    if args.distill_weight > 0:
        teacher = copy.deepcopy(classifier).to(device)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

    trainable_names = apply_train_mode(classifier, args.classifier_train_mode)
    classifier_params = [p for p in classifier.parameters() if p.requires_grad]
    head_params = [p for p in heads.parameters() if p.requires_grad]
    specificity_alpha_head_params = (
        [p for p in specificity_alpha_head.parameters() if p.requires_grad]
        if specificity_alpha_head is not None
        else []
    )
    trainable_head_names = [name for name, param in heads.named_parameters() if param.requires_grad]
    optimizer_groups = []
    if classifier_params:
        optimizer_groups.append({"params": classifier_params, "lr": args.lr})
    if head_params:
        optimizer_groups.append({"params": head_params, "lr": args.head_lr})
    if specificity_alpha_head_params:
        alpha_lr = args.specificity_alpha_head_lr if args.specificity_alpha_head_lr > 0 else args.head_lr
        optimizer_groups.append({"params": specificity_alpha_head_params, "lr": alpha_lr})
    if not optimizer_groups:
        raise ValueError("No trainable parameters selected")
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=args.weight_decay)
    pairwise_criterion = PairwiseRankingLoss()
    metrics_history = []
    use_kinetics_training = any(
        value > 0
        for value in [
            args.kinetics_weight,
            args.kin_teacher_weight,
            args.kin_base_anchor_weight,
            args.kin_rank_weight,
            args.kin_pearson_weight,
            args.contrastive_weight,
            args.ratio_consistency_weight,
        ]
    )
    preserve_loaded_kinetics_payload = (
        args.kin_head_train_mode == "none"
        and not use_kinetics_training
        and init_ckpt.kinetics_head_state is not None
        and bool(init_ckpt.run_meta)
    )

    run_meta = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_kind": init_ckpt.checkpoint_kind,
        "kin_train_sources_json": str(args.kin_train_sources_json) if args.kin_train_sources_json else None,
        "kin_valid_sources_json": str(args.kin_valid_sources_json) if args.kin_valid_sources_json else None,
        "source_loss_weights_json": str(args.source_loss_weights_json) if args.source_loss_weights_json else None,
        "task_branch_weight_json": str(args.task_branch_weight_json) if args.task_branch_weight_json else None,
        "kin_substrate_fp_path": str(args.kin_substrate_fp_path) if args.kin_substrate_fp_path else None,
        "kin_train_row_extra_features_npz": str(args.kin_train_row_extra_features_npz) if args.kin_train_row_extra_features_npz else None,
        "kin_valid_row_extra_features_npz": str(args.kin_valid_row_extra_features_npz) if args.kin_valid_row_extra_features_npz else None,
        "kin_train_aux_row_extra_features_npz": str(args.kin_train_aux_row_extra_features_npz) if args.kin_train_aux_row_extra_features_npz else None,
        "kin_valid_aux_row_extra_features_npz": str(args.kin_valid_aux_row_extra_features_npz) if args.kin_valid_aux_row_extra_features_npz else None,
        "source_loss_weights": source_loss_weights,
        "task_branch_weight_map": task_branch_weight_map,
        "kin_source_column": args.kin_source_column,
        "kin_default_source": args.kin_default_source,
        "source_specific_stats": bool(args.source_specific_stats),
        "source_names": train_set.source_names,
        "substrate_fp_dim": int(train_set.substrate_fp_dim),
        "row_extra_feature_dim": int(train_set.row_extra_feature_dim),
        "aux_row_extra_feature_dim": int(train_set.aux_row_extra_feature_dim),
        "reset_classifier": bool(args.reset_classifier),
        "reset_heads": bool(args.reset_heads),
        "reset_row_extra_residual": bool(args.reset_row_extra_residual),
        "reset_aux_row_extra_residual": bool(args.reset_aux_row_extra_residual),
        "reset_structured_residual": bool(args.reset_structured_residual),
        "resumed_kinetics_heads": resumed_kinetics_heads,
        "kinetics_head_missing": kinetics_head_missing[:50],
        "kinetics_head_missing_count": len(kinetics_head_missing),
        "kinetics_head_skipped_shape": kinetics_head_skipped_shape,
        "kinetics_head_skipped_shape_count": len(kinetics_head_skipped_shape),
        "train_rows": len(train_set),
        "valid_rows": len(valid_set),
        "replay_batches": len(replay_loader.dataset) if replay_loader is not None else 0,
        "batch_size": args.batch_size,
        "spec_batch_size": args.spec_batch_size,
        "lr": args.lr,
        "head_lr": args.head_lr,
        "kin_loss_mode": args.kin_loss_mode,
        "kin_huber_delta": args.kin_huber_delta,
        "kin_huber_mix": args.kin_huber_mix,
        "kinetics_weight": args.kinetics_weight,
        "specificity_weight": args.specificity_weight,
        "specificity_interval": args.specificity_interval,
        "specificity_head_mode": args.specificity_head_mode,
        "specificity_alpha_repr_dim": args.specificity_alpha_repr_dim,
        "specificity_alpha_hidden_size": args.specificity_alpha_hidden_size,
        "specificity_alpha_layers": args.specificity_alpha_layers,
        "specificity_alpha_dropout": args.specificity_alpha_dropout,
        "specificity_alpha_head_lr": (
            args.specificity_alpha_head_lr if args.specificity_alpha_head_lr > 0 else args.head_lr
        ),
        "specificity_alpha_head_missing": specificity_alpha_head_missing[:50],
        "specificity_alpha_head_missing_count": len(specificity_alpha_head_missing),
        "specificity_alpha_head_unexpected": specificity_alpha_head_unexpected[:50],
        "specificity_alpha_head_skipped_shape": specificity_alpha_head_skipped_shape,
        "specificity_alpha_head_skipped_shape_count": len(specificity_alpha_head_skipped_shape),
        "specificity_alpha_head_preserved_from_init": (
            specificity_alpha_head is None and init_ckpt.specificity_alpha_head_state is not None
        ),
        "contrastive_weight": args.contrastive_weight,
        "distill_weight": args.distill_weight,
        "kin_rank_weight": args.kin_rank_weight,
        "kin_rank_min_delta": args.kin_rank_min_delta,
        "kin_rank_temperature": args.kin_rank_temperature,
        "anchor_weight": args.anchor_weight,
        "classifier_train_mode": args.classifier_train_mode,
        "classifier_lora_rank": int(args.classifier_lora_rank),
        "classifier_lora_alpha": float(args.classifier_lora_alpha),
        "classifier_lora_dropout": float(args.classifier_lora_dropout),
        "classifier_lora_targets": list(args.classifier_lora_targets),
        "classifier_lora_layers": lora_layer_names[:200],
        "classifier_lora_layer_count": len(lora_layer_names),
        "classifier_warmup_epochs": args.classifier_warmup_epochs,
        "dropout": init_ckpt.dropout,
        "kin_head_hidden_size": head_hidden_size,
        "kin_head_layers": head_layers,
        "kin_head_mode": head_mode,
        "kin_full_feature_stats_checkpoint": (
            str(args.kin_full_feature_stats_checkpoint) if args.kin_full_feature_stats_checkpoint else None
        ),
        "kin_full_feature_stats_dim": int(full_feature_mean.shape[-1]) if full_feature_mean is not None else 0,
        "kin_init_full_feature_mlp_checkpoint": (
            str(args.kin_init_full_feature_mlp_checkpoint) if args.kin_init_full_feature_mlp_checkpoint else None
        ),
        "kin_init_full_feature_mlp_summary": full_feature_mlp_init,
        "kin_moe_num_experts": args.kin_moe_num_experts,
        "kin_moe_residual_num_experts": args.kin_moe_residual_num_experts,
        "kin_structured_residual_mode": args.kin_structured_residual_mode,
        "kin_structured_residual_network": args.kin_structured_residual_network,
        "kin_structured_residual_hidden_size": int(args.kin_structured_residual_hidden_size or head_hidden_size),
        "kin_structured_residual_layers": int(args.kin_structured_residual_layers or head_layers),
        "kin_structured_residual_output_weights": args.kin_structured_residual_output_weights,
        "kin_source_head_mode": args.kin_source_head_mode,
        "kin_row_extra_mode": args.kin_row_extra_mode,
        "kin_row_extra_residual_mode": args.kin_row_extra_residual_mode,
        "kin_row_extra_network": args.kin_row_extra_network,
        "kin_aux_row_extra_network": args.kin_aux_row_extra_network,
        "kin_aux_row_extra_residual_mode": args.kin_aux_row_extra_residual_mode,
        "kin_aux_row_extra_blend_weights": args.kin_aux_row_extra_blend_weights,
        "kin_enable_second_aux_row_extra_residual": bool(args.kin_enable_second_aux_row_extra_residual),
        "kin_second_aux_row_extra_network": args.kin_second_aux_row_extra_network,
        "kin_second_aux_row_extra_residual_mode": args.kin_second_aux_row_extra_residual_mode,
        "kin_second_aux_row_extra_weights": args.kin_second_aux_row_extra_weights,
        "kin_extra_dropout": args.kin_extra_dropout,
        "kin_aux_row_extra_hidden_size": int(args.kin_aux_row_extra_hidden_size or head_hidden_size),
        "kin_aux_row_extra_layers": int(args.kin_aux_row_extra_layers or head_layers),
        "kin_second_aux_row_extra_hidden_size": int(args.kin_second_aux_row_extra_hidden_size or args.kin_aux_row_extra_hidden_size or head_hidden_size),
        "kin_second_aux_row_extra_layers": int(args.kin_second_aux_row_extra_layers or args.kin_aux_row_extra_layers or head_layers),
        "kin_head_train_mode": args.kin_head_train_mode,
        "row_extra_residual_dim": int(row_extra_residual_dim),
        "aux_row_extra_residual_dim": int(train_set.aux_row_extra_feature_dim),
        "second_aux_row_extra_residual_dim": int(
            train_set.aux_row_extra_feature_dim if args.kin_enable_second_aux_row_extra_residual else 0
        ),
        "kin_train_teacher_targets_npz": str(args.kin_train_teacher_targets_npz) if args.kin_train_teacher_targets_npz else None,
        "kin_valid_teacher_targets_npz": str(args.kin_valid_teacher_targets_npz) if args.kin_valid_teacher_targets_npz else None,
        "kin_teacher_weight": args.kin_teacher_weight,
        "kin_teacher_target_weights": args.kin_teacher_target_weights,
        "kin_label_stats_mode": args.kin_label_stats_mode,
        "kin_target_columns": args.kin_target_columns or TASK_COLUMNS,
        "train_label_stats": train_stats,
        "kin_base_anchor_weight": args.kin_base_anchor_weight,
        "kin_base_anchor_target_weights": args.kin_base_anchor_target_weights,
        "kin_pearson_weight": args.kin_pearson_weight,
        "kin_pearson_target_weights": args.kin_pearson_target_weights,
        "ratio_prediction_mode": args.ratio_prediction_mode,
        "ratio_consistency_weight": args.ratio_consistency_weight,
        "target_weights": args.target_weights,
        "kin_drfp_dropout_prob": args.kin_drfp_dropout_prob,
        "kin_zero_drfp_input": bool(args.kin_zero_drfp_input),
        "spec_zero_drfp_input": bool(args.spec_zero_drfp_input),
        "eval_zero_drfp_input": bool(args.eval_zero_drfp_input),
        "drop_loaded_posthoc_raw_calibration": bool(args.drop_loaded_posthoc_raw_calibration),
        "trainable_classifier_param_count": sum(p.numel() for p in classifier_params),
        "trainable_classifier_param_names": trainable_names[:200],
        "trainable_head_param_count": sum(p.numel() for p in head_params),
        "trainable_head_param_names": trainable_head_names[:200],
        "trainable_specificity_alpha_head_param_count": sum(
            p.numel() for p in specificity_alpha_head_params
        ),
        "label_stats": stats,
    }
    if init_ckpt.run_meta and not args.drop_loaded_posthoc_raw_calibration:
        for key in [
            "kin_posthoc_raw_affine",
            "kin_posthoc_raw_affine_source",
            "kin_posthoc_raw_affine_created_by",
            "kin_posthoc_raw_poly",
            "kin_posthoc_raw_poly_source",
            "kin_posthoc_raw_poly_created_by",
            "kin_posthoc_raw_piecewise",
            "kin_posthoc_raw_piecewise_source",
            "kin_posthoc_raw_piecewise_created_by",
            "kin_posthoc_raw_basis",
            "kin_posthoc_raw_basis_source",
            "kin_posthoc_raw_basis_created_by",
        ]:
            if key in init_ckpt.run_meta:
                run_meta[key] = copy.deepcopy(init_ckpt.run_meta[key])
    checkpoint_run_meta = run_meta
    if preserve_loaded_kinetics_payload:
        checkpoint_run_meta = copy.deepcopy(init_ckpt.run_meta)
        checkpoint_run_meta["classifier_repair_applied"] = True
        checkpoint_run_meta["classifier_repair_latest"] = copy.deepcopy(run_meta)
    if specificity_alpha_head is None and init_ckpt.specificity_alpha_head_state is not None:
        for key in (
            "specificity_alpha_repr_dim",
            "specificity_alpha_hidden_size",
            "specificity_alpha_layers",
            "specificity_alpha_dropout",
            "specificity_alpha_base_logit_residual_init",
            "specificity_alpha_base_logit_residual_trainable",
        ):
            if key in init_ckpt.run_meta:
                checkpoint_run_meta[key] = copy.deepcopy(init_ckpt.run_meta[key])
        checkpoint_run_meta["specificity_alpha_head_preserved_from_init"] = True
    (args.output_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "checkpoint_run_meta.json").write_text(
        json.dumps(checkpoint_run_meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(run_meta, indent=2, ensure_ascii=False))

    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else len(train_loader)
    for epoch in range(1, args.epochs + 1):
        warmup_classifier = epoch <= args.classifier_warmup_epochs
        if warmup_classifier:
            apply_train_mode(classifier, "none")
        else:
            apply_train_mode(classifier, args.classifier_train_mode)

        classifier.train((not warmup_classifier) and args.classifier_train_mode != "none")
        heads.train()
        if replay_loader is not None and hasattr(replay_loader.dataset, "shuffle_batches"):
            replay_loader.dataset.shuffle_batches()
        replay_iter = infinite_loader(replay_loader) if replay_loader is not None else None
        kin_iter = infinite_loader(train_loader) if use_kinetics_training else None
        totals = {
            "loss": 0.0,
            "kin": 0.0,
            "spec": 0.0,
            "contrast": 0.0,
            "distill": 0.0,
            "ratio_consistency": 0.0,
            "kin_rank": 0.0,
            "kin_pearson": 0.0,
            "kin_teacher": 0.0,
            "anchor": 0.0,
            "kin_base_anchor": 0.0,
        }

        pbar = tqdm(range(steps_per_epoch), desc=f"epoch {epoch}/{args.epochs}")
        for step_idx in pbar:
            optimizer.zero_grad(set_to_none=True)
            kin_batch = move_batch(next(kin_iter), device) if kin_iter is not None else None
            if kin_batch is not None:
                kin_batch = apply_drfp_input_policy(
                    kin_batch,
                    zero_drfp=args.kin_zero_drfp_input,
                    dropout_prob=args.kin_drfp_dropout_prob,
                    training=True,
                )
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                if use_kinetics_training:
                    kin_out = None if getattr(heads, "mode", None) == "row_extra_only" else classifier_forward(classifier, kin_batch)
                    raw_kin_preds = forward_kinetics_heads(heads, kin_batch, kin_out)
                    kin_preds = apply_ratio_prediction_mode(
                        raw_kin_preds,
                        stats,
                        args.ratio_prediction_mode,
                        feature_sources=kin_batch["feature_sources"],
                    )
                    kin_targets = normalize_targets(
                        kin_batch["targets"],
                        kin_batch["masks"],
                        stats,
                        device,
                        feature_sources=kin_batch["feature_sources"],
                    )
                    target_weights = torch.tensor(args.target_weights, device=device, dtype=kin_preds.dtype)
                    sample_source_weights = torch.tensor(
                        [source_loss_weights.get(src, 1.0) for src in kin_batch["feature_sources"]],
                        device=device,
                        dtype=kin_preds.dtype,
                    )
                    kin_loss = masked_regression_loss(
                        kin_preds,
                        kin_targets,
                        kin_batch["masks"],
                        loss_mode=args.kin_loss_mode,
                        huber_delta=args.kin_huber_delta,
                        huber_mix=args.kin_huber_mix,
                        target_weights=target_weights,
                        sample_weights=sample_source_weights,
                    )
                    kin_pearson_loss = kin_loss.new_tensor(0.0)
                    if args.kin_pearson_weight > 0:
                        pearson_target_weights = torch.tensor(
                            args.kin_pearson_target_weights,
                            device=device,
                            dtype=kin_preds.dtype,
                        )
                        kin_pearson_loss = masked_pearson_loss(
                            kin_preds,
                            kin_targets,
                            kin_batch["masks"],
                            target_weights=pearson_target_weights,
                            sample_weights=sample_source_weights,
                        )
                    teacher_loss = kin_loss.new_tensor(0.0)
                    if args.kin_teacher_weight > 0:
                        teacher_masks = kin_batch["teacher_masks"]
                        if teacher_masks.detach().sum().item() > 0:
                            teacher_targets = normalize_targets(
                                kin_batch["teacher_targets"],
                                teacher_masks,
                                stats,
                                device,
                                feature_sources=kin_batch["feature_sources"],
                            )
                            teacher_target_weights = torch.tensor(
                                args.kin_teacher_target_weights,
                                device=device,
                                dtype=kin_preds.dtype,
                            )
                            teacher_loss = masked_regression_loss(
                                kin_preds,
                                teacher_targets,
                                teacher_masks,
                                loss_mode="mse",
                                huber_delta=args.kin_huber_delta,
                                huber_mix=args.kin_huber_mix,
                                target_weights=teacher_target_weights,
                                sample_weights=sample_source_weights,
                            )
                    base_anchor_loss = kin_loss.new_tensor(0.0)
                    if base_anchor_heads is not None:
                        with torch.no_grad():
                            base_raw_preds = forward_kinetics_heads(base_anchor_heads, kin_batch, kin_out)
                            base_preds = apply_ratio_prediction_mode(
                                base_raw_preds,
                                stats,
                                args.ratio_prediction_mode,
                                feature_sources=kin_batch["feature_sources"],
                            )
                        base_anchor_masks = kin_batch["masks"]
                        if "teacher_masks" in kin_batch:
                            base_anchor_masks = torch.maximum(base_anchor_masks, kin_batch["teacher_masks"])
                        base_anchor_target_weights = torch.tensor(
                            args.kin_base_anchor_target_weights,
                            device=device,
                            dtype=kin_preds.dtype,
                        )
                        base_anchor_loss = masked_regression_loss(
                            kin_preds,
                            base_preds,
                            base_anchor_masks,
                            loss_mode="mse",
                            huber_delta=args.kin_huber_delta,
                            huber_mix=args.kin_huber_mix,
                            target_weights=base_anchor_target_weights,
                            sample_weights=sample_source_weights,
                        )
                    kin_rank_loss = kinetics_pairwise_rank_loss(
                        kin_preds,
                        kin_targets,
                        kin_batch["masks"],
                        target_weights=target_weights,
                        min_delta=args.kin_rank_min_delta,
                        temperature=args.kin_rank_temperature,
                        source_ids=kin_batch["source_ids"],
                    )
                    ratio_aux_loss = ratio_consistency_loss(
                        raw_kin_preds,
                        kin_batch["masks"],
                        stats,
                        feature_sources=kin_batch["feature_sources"],
                    )
                    contrast_loss = kin_loss.new_tensor(0.0)
                    if args.contrastive_weight > 0 and kin_out is not None:
                        contrast_loss = pair_contrastive_loss(
                            kin_out["extra_feature"], kin_out["drfp_feature"], args.contrastive_temperature
                        )
                else:
                    kin_loss = torch.zeros((), device=device)
                    teacher_loss = torch.zeros((), device=device)
                    base_anchor_loss = torch.zeros((), device=device)
                    kin_rank_loss = torch.zeros((), device=device)
                    ratio_aux_loss = torch.zeros((), device=device)
                    contrast_loss = torch.zeros((), device=device)
                    kin_pearson_loss = torch.zeros((), device=device)
                spec_loss = kin_loss.new_tensor(0.0)
                distill_loss = kin_loss.new_tensor(0.0)
                anchor_loss = anchor_regularization(classifier, anchor_state)
                use_replay_this_step = (
                    replay_iter is not None
                    and args.specificity_weight > 0
                    and args.specificity_interval > 0
                    and step_idx % args.specificity_interval == 0
                )
                if use_replay_this_step:
                    spec_batch = squeeze_replay_batch(next(replay_iter))
                    spec_batch = move_batch(spec_batch, device)
                    spec_batch = apply_drfp_input_policy(spec_batch, zero_drfp=args.spec_zero_drfp_input)
                    spec_out = classifier_forward(classifier, spec_batch)
                    if specificity_alpha_head is not None:
                        spec_logits = specificity_alpha_head(spec_out["alpha_repr"])
                    else:
                        spec_logits = spec_out["logits"]
                    labels = spec_batch["labels"].float()
                    spec_loss = pairwise_criterion(spec_logits, labels)
                    if teacher is not None and args.distill_weight > 0:
                        with torch.no_grad():
                            teacher_logits = classifier_forward(teacher, spec_batch)["logits"]
                        distill_loss = F.mse_loss(spec_logits, teacher_logits)
                loss = (
                    args.kinetics_weight * kin_loss
                    + args.kin_teacher_weight * teacher_loss
                    + args.specificity_weight * spec_loss
                    + args.contrastive_weight * contrast_loss
                    + args.distill_weight * distill_loss
                    + args.ratio_consistency_weight * ratio_aux_loss
                    + args.kin_rank_weight * kin_rank_loss
                    + args.kin_pearson_weight * kin_pearson_loss
                    + args.kin_base_anchor_weight * base_anchor_loss
                    + args.anchor_weight * anchor_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier_params + head_params + specificity_alpha_head_params, 1.0)
            optimizer.step()

            totals["loss"] += float(loss.detach().item())
            totals["kin"] += float(kin_loss.detach().item())
            totals["kin_teacher"] += float(teacher_loss.detach().item())
            totals["kin_base_anchor"] += float(base_anchor_loss.detach().item())
            totals["spec"] += float(spec_loss.detach().item())
            totals["contrast"] += float(contrast_loss.detach().item())
            totals["distill"] += float(distill_loss.detach().item())
            totals["ratio_consistency"] += float(ratio_aux_loss.detach().item())
            totals["kin_rank"] += float(kin_rank_loss.detach().item())
            totals["kin_pearson"] += float(kin_pearson_loss.detach().item())
            totals["anchor"] += float(anchor_loss.detach().item())
            denom = max(1, pbar.n + 1)
            pbar.set_postfix(loss=totals["loss"] / denom, kin=totals["kin"] / denom, spec=totals["spec"] / denom)

        metrics = evaluate_kinetics(
            classifier,
            heads,
            valid_loader,
            stats,
            device,
            amp_dtype,
            args.ratio_prediction_mode,
            zero_drfp=args.eval_zero_drfp_input,
        )
        metrics.update({f"train_{k}_loss": v / max(steps_per_epoch, 1) for k, v in totals.items()})
        metrics["epoch"] = epoch
        metrics_history.append(metrics)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        should_save_checkpoint = args.checkpoint_save_every > 0 and (
            epoch == args.epochs or epoch % args.checkpoint_save_every == 0
        )
        if args.save_classifier_epoch_state and should_save_checkpoint:
            torch.save(merged_lora_state_dict(classifier), args.output_dir / f"model_epoch{epoch}.pt")
        if should_save_checkpoint:
            kinetics_heads_state = (
                init_ckpt.kinetics_head_state if preserve_loaded_kinetics_payload else heads.state_dict()
            )
            checkpoint_payload = {
                "classifier": merged_lora_state_dict(classifier),
                "kinetics_heads": kinetics_heads_state,
                "optimizer": optimizer.state_dict(),
                "metrics": metrics,
                "run_meta": checkpoint_run_meta,
            }
            if specificity_alpha_head is not None:
                checkpoint_payload["specificity_alpha_head"] = specificity_alpha_head.state_dict()
            elif init_ckpt.specificity_alpha_head_state is not None:
                # Kinetics-only continuation can keep the specificity alpha head
                # unchanged while training only kinetics heads.  Preserve it so
                # a saved multitask checkpoint remains directly evaluable on Enzyme-405.
                checkpoint_payload["specificity_alpha_head"] = init_ckpt.specificity_alpha_head_state
            if init_ckpt.specificity_alpha_base_logit_residual_scale is not None:
                checkpoint_payload["specificity_alpha_base_logit_residual_scale"] = (
                    init_ckpt.specificity_alpha_base_logit_residual_scale
                )
            if preserve_loaded_kinetics_payload:
                checkpoint_payload["kinetics_classifier"] = init_ckpt.classifier_state
            torch.save(checkpoint_payload, args.output_dir / f"multitask_epoch{epoch}.pt")
        (args.output_dir / "metrics_history.json").write_text(
            json.dumps(metrics_history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
