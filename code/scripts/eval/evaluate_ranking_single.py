#!/usr/bin/env python
"""
排名评估脚本 - 集成版本（包含结果收集）

用法:
  python scripts/eval/evaluate_ranking_single.py --gpu 0 --checkpoint path/to/model.pt --use_drfp
"""

import os
import sys
import json
import argparse
import pickle
import time
import random
from functools import lru_cache
from contextlib import nullcontext
import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BUNDLE_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_ROOT = os.path.join(BUNDLE_ROOT, "data")
OUTPUT_ROOT = os.path.join(BUNDLE_ROOT, "runtime_outputs")
sys.path.insert(0, PROJECT_ROOT)
os.environ['PYTHONPATH'] = PROJECT_ROOT + ':' + os.environ.get('PYTHONPATH', '')

from alphaenzyme.data.direct_feature_support import DirectContextAssetStore
from alphaenzyme.data.external_feature_bank import load_aligned_external_feature_matrix, normalize_bank_specs
from alphaenzyme.utils.sharding import partition_indices


def set_eval_reproducibility(seed: int, deterministic: bool = True):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
                torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def build_eval_autocast(eval_precision: str, device_type: str = "cuda"):
    precision = str(eval_precision).lower()
    if precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type=device_type, dtype=torch.float16)
    raise ValueError(f"Unsupported eval_precision: {eval_precision}")


def normalize_reaction_value(value):
    """Normalize optional reaction strings read from CSV/JSON tables."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none"}:
        return None
    return value


def load_reactions_from_csv(csv_paths):
    """Load unique canonical reaction SMILES from one or more CSV files."""
    if not csv_paths:
        return []
    import pandas as pd

    reactions = []
    seen = set()
    for csv_path in csv_paths:
        if not csv_path:
            continue
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"DRFP gate CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        rxn_col = "CANO_RXN_SMILES" if "CANO_RXN_SMILES" in df.columns else "reaction"
        if rxn_col not in df.columns:
            raise ValueError(f"CSV for DRFP gate has no reaction column: {csv_path}")
        for value in df[rxn_col].values:
            reaction = normalize_reaction_value(value)
            if reaction is not None and reaction not in seen:
                seen.add(reaction)
                reactions.append(reaction)
    return reactions


def tanimoto_to_binary_bank(fp, bank_matrix):
    """Compute max Tanimoto similarity between one binary DRFP and a bank."""
    if bank_matrix.size == 0:
        return 0.0
    fp_bool = np.asarray(fp, dtype=np.uint8) > 0
    fp_u8 = fp_bool.astype(np.uint8, copy=False)
    fp_sum = float(fp_u8.sum())
    if fp_sum <= 0:
        return 0.0
    intersections = bank_matrix @ fp_u8
    unions = bank_matrix.sum(axis=1) + fp_sum - intersections
    valid = unions > 0
    if not np.any(valid):
        return 0.0
    sims = intersections[valid] / unions[valid]
    return float(np.max(sims)) if sims.size else 0.0


def build_similarity_gated_drfp_dict(
    drfp_dict,
    gate_train_csvs,
    eval_csv_path,
    low,
    high,
    power,
):
    """Scale eval DRFP vectors by similarity to training reactions.

    The DRFP branch remains enabled, but each eval reaction receives
    gate_weight * raw_drfp.  Reactions outside eval_csv_path are left unchanged
    so this function is safe for targeted offline evaluation.
    """
    train_reactions = load_reactions_from_csv(gate_train_csvs)
    train_fps = []
    for reaction in train_reactions:
        fp = drfp_dict.get(reaction)
        if fp is not None:
            train_fps.append((np.asarray(fp, dtype=np.uint8) > 0).astype(np.uint8, copy=False))
    if not train_fps:
        raise ValueError("No train DRFP fingerprints found for similarity-gated DRFP")
    bank_matrix = np.stack(train_fps, axis=0)

    eval_reactions = load_reactions_from_csv([eval_csv_path]) if eval_csv_path else list(drfp_dict.keys())
    low = float(low)
    high = float(high)
    if high <= low:
        raise ValueError(f"DRFP gate high threshold must be > low threshold, got low={low}, high={high}")
    power = max(float(power), 1e-8)

    gated = dict(drfp_dict)
    weights = {}
    max_sims = {}
    for reaction in eval_reactions:
        fp = drfp_dict.get(reaction)
        if fp is None:
            continue
        max_sim = tanimoto_to_binary_bank(fp, bank_matrix)
        weight = (max_sim - low) / (high - low)
        weight = float(np.clip(weight, 0.0, 1.0) ** power)
        gated[reaction] = np.asarray(fp, dtype=np.float32) * weight
        weights[reaction] = weight
        max_sims[reaction] = max_sim

    if weights:
        weight_values = np.array(list(weights.values()), dtype=np.float32)
        sim_values = np.array(list(max_sims.values()), dtype=np.float32)
        print(
            "DRFP similarity gate: "
            f"eval_reactions={len(weights)}, train_reactions={len(train_fps)}, "
            f"weight_mean={weight_values.mean():.4f}, weight_min={weight_values.min():.4f}, "
            f"weight_max={weight_values.max():.4f}, maxsim_mean={sim_values.mean():.4f}"
        )
    else:
        print("Warning: DRFP similarity gate did not match any eval reactions")
    return gated, {
        "enabled": True,
        "train_csvs": list(gate_train_csvs or []),
        "low": low,
        "high": high,
        "power": power,
        "n_train_reactions_with_drfp": int(len(train_fps)),
        "n_eval_reactions_gated": int(len(weights)),
        "weight_mean": float(np.mean(list(weights.values()))) if weights else 0.0,
        "weight_min": float(np.min(list(weights.values()))) if weights else 0.0,
        "weight_max": float(np.max(list(weights.values()))) if weights else 0.0,
        "max_similarity_mean": float(np.mean(list(max_sims.values()))) if max_sims else 0.0,
    }


class AlphaReprSpecificityHead(torch.nn.Module):
    """Specificity score head over the shared AlphaEnzyme representation."""

    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers = [
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(max(0, int(num_layers) - 1)):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.GELU(),
                torch.nn.Dropout(dropout),
            ])
        layers.append(torch.nn.Linear(hidden_size, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, alpha_repr):
        return self.net(alpha_repr).squeeze(-1)


def load_checkpoint_state_dict(checkpoint_path, map_location="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if isinstance(checkpoint, dict) and 'model' in checkpoint and isinstance(checkpoint['model'], dict):
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'classifier' in checkpoint and isinstance(checkpoint['classifier'], dict):
        # Multitask all-in-one checkpoints store the specificity model under this key.
        state_dict = checkpoint['classifier']
    else:
        state_dict = checkpoint
    return state_dict


def load_specificity_alpha_head(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "specificity_alpha_head" not in checkpoint:
        raise ValueError(
            "--use_alpha_repr_specificity_head requires a multitask checkpoint "
            "containing specificity_alpha_head"
        )
    state = checkpoint["specificity_alpha_head"]
    run_meta = checkpoint.get("run_meta") or {}
    first_linear = state.get("net.1.weight")
    if first_linear is None:
        raise ValueError("Invalid specificity_alpha_head state: missing net.1.weight")
    input_dim = int(run_meta.get("specificity_alpha_repr_dim", first_linear.shape[1]))
    hidden_size = int(run_meta.get("specificity_alpha_hidden_size", first_linear.shape[0]))
    num_layers = int(run_meta.get("specificity_alpha_layers", 2))
    dropout = float(run_meta.get("specificity_alpha_dropout", 0.0))
    head = AlphaReprSpecificityHead(input_dim, hidden_size, num_layers, dropout)
    missing, unexpected = head.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"specificity_alpha_head load mismatch: missing={missing}, unexpected={unexpected}")
    head = head.to(device)
    head.eval()
    residual_scale = checkpoint.get("specificity_alpha_base_logit_residual_scale")
    if residual_scale is None:
        residual_scale = run_meta.get("specificity_alpha_base_logit_residual_init", 0.0)
    residual_scale = torch.as_tensor(residual_scale, dtype=torch.float32, device=device).view(())
    return head, residual_scale


def infer_checkpoint_config(state_dict, embedding_dir, use_drfp=None, compressed=None,
                            hidden_size=None, n_transformer_layers=None):
    inferred = {}

    if compressed is None:
        inferred['compressed'] = 'input_proj.weight' in state_dict
        if not inferred['compressed']:
            npz_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.npz')])
            if npz_files:
                sample = np.load(os.path.join(embedding_dir, npz_files[0]))
                inferred['compressed'] = sample['z'].ndim == 2
            else:
                inferred['compressed'] = False
    else:
        inferred['compressed'] = compressed

    if use_drfp is None:
        inferred['use_drfp'] = any('drfp' in k for k in state_dict)
    else:
        inferred['use_drfp'] = use_drfp

    if hidden_size is None:
        if inferred['compressed'] and 'input_proj.weight' in state_dict:
            inferred['hidden_size'] = int(state_dict['input_proj.weight'].shape[0])
        elif not inferred['compressed'] and 's_mlp.0.weight' in state_dict:
            inferred['hidden_size'] = int(state_dict['s_mlp.0.weight'].shape[0])
        else:
            inferred['hidden_size'] = 256 if inferred['compressed'] else 512
    else:
        inferred['hidden_size'] = hidden_size

    if n_transformer_layers is None:
        layer_ids = set()
        for key in state_dict:
            if key.startswith('transformer.layers.'):
                try:
                    layer_ids.add(int(key.split('.')[2]))
                except Exception:
                    pass
        inferred['n_transformer_layers'] = (max(layer_ids) + 1) if layer_ids else 4
    else:
        inferred['n_transformer_layers'] = n_transformer_layers

    inferred['pooling_mode'] = 'ligand_plus_global' if 'global_pool_scale' in state_dict else 'ligand_only'
    branch_keys = []
    for key in state_dict:
        if key.startswith('extra_branch_proj.') and key.endswith('.1.weight'):
            try:
                branch_idx = int(key.split('.')[1])
                branch_keys.append((branch_idx, key))
            except Exception:
                pass
    branch_keys.sort()
    inferred['extra_feature_branch_dims'] = [int(state_dict[key].shape[1]) for _, key in branch_keys]
    inferred['use_extra_features'] = any(
        k.startswith('extra_proj.') or k.startswith('extra_branch_proj.') for k in state_dict
    )
    if inferred['extra_feature_branch_dims']:
        inferred['extra_feature_dim'] = int(sum(inferred['extra_feature_branch_dims']))
        inferred['extra_hidden_size'] = int(state_dict[branch_keys[0][1]].shape[0])
    else:
        inferred['extra_feature_dim'] = int(state_dict['extra_proj.0.weight'].shape[1]) if 'extra_proj.0.weight' in state_dict else 0
        inferred['extra_hidden_size'] = int(state_dict['extra_proj.0.weight'].shape[0]) if 'extra_proj.0.weight' in state_dict else inferred['hidden_size']
    branch_count = 1 + max(1, len(inferred['extra_feature_branch_dims'])) if inferred['use_extra_features'] else 0
    if 'extra_branch_fusion.4.weight' in state_dict:
        inferred['extra_fusion_mode'] = (
            'branch_gate_sum'
            if int(state_dict['extra_branch_fusion.4.weight'].shape[0]) == branch_count
            else 'branch_concat_mlp'
        )
    elif 'extra_fusion.4.weight' in state_dict:
        # Single-extra-feature checkpoints can also use branch_gate_sum.  In
        # that case the final layer predicts two weights: one for the pooled
        # pairformer sample feature and one for the extra feature branch.
        out_dim = int(state_dict['extra_fusion.4.weight'].shape[0])
        if out_dim == 2:
            inferred['extra_fusion_mode'] = 'branch_gate_sum'
        elif 'extra_fusion.0.weight' in state_dict and int(state_dict['extra_fusion.0.weight'].shape[0]) == inferred['hidden_size'] * 2:
            inferred['extra_fusion_mode'] = 'branch_concat_mlp'
        else:
            inferred['extra_fusion_mode'] = 'delta'
    elif 'extra_gate.weight' in state_dict:
        inferred['extra_fusion_mode'] = 'gated_add'
    else:
        inferred['extra_fusion_mode'] = 'delta'
    inferred['use_context_features'] = 'context_proj.0.weight' in state_dict
    inferred['context_dim'] = int(state_dict['context_proj.0.weight'].shape[0]) if 'context_proj.0.weight' in state_dict else 0
    inferred['overall_corr_dim'] = int(state_dict['overall_corr_proj.0.weight'].shape[0]) if 'overall_corr_proj.0.weight' in state_dict else 0

    return inferred


class EmbeddingDataset(Dataset):
    """Embedding 数据集"""

    def __init__(
        self,
        embedding_dir,
        json_dir,
        drfp_dict=None,
        csv_path=None,
        external_feature_banks=None,
        external_missing_policy="error",
        context_cache_path=None,
        context_bank_path=None,
        context_missing_policy="error",
    ):
        self.embedding_dir = embedding_dir
        self.json_dir = json_dir
        self.drfp_dict = drfp_dict
        self.npz_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.npz')])
        self.external_features = None
        self.external_feature_dim = 0
        self.context_store = None

        # 加载 CSV 映射 (sample_id -> CANO_RXN_SMILES)
        self.cano_rxn_mapping = {}
        if csv_path and drfp_dict:
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    sample_id = str(row['id']) if 'id' in row.index else str(idx)
                    if pd.notna(row.get('CANO_RXN_SMILES')):
                        self.cano_rxn_mapping[sample_id] = str(row['CANO_RXN_SMILES'])
                print(f"Loaded CANO_RXN_SMILES mapping for {len(self.cano_rxn_mapping)} samples")
            except Exception as e:
                print(f"Warning: Failed to load CSV mapping: {e}")
        sample_ids = [f.replace('.npz', '') for f in self.npz_files]
        external_features, external_meta = load_aligned_external_feature_matrix(
            sample_ids=sample_ids,
            bank_specs=external_feature_banks,
            missing_policy=external_missing_policy,
        )
        if external_features is not None:
            self.external_features = external_features
            self.external_feature_dim = int(external_features.shape[1])
            print(
                f"Loaded external feature matrix: dim={self.external_feature_dim}, "
                f"banks={[m.get('name') for m in external_meta]}"
            )
        if context_cache_path and context_bank_path:
            self.context_store = DirectContextAssetStore(
                cache_path=context_cache_path,
                bank_path=context_bank_path,
                missing_policy=context_missing_policy,
            )
            print(
                f"Loaded direct context assets: len={self.context_store.context_len}, "
                f"dim={self.context_store.context_dim}"
            )

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_file = self.npz_files[idx]
        sample_id = npz_file.replace('.npz', '')

        data = np.load(os.path.join(self.embedding_dir, npz_file))

        def _to_f32(arr):
            if arr.dtype == np.uint16:
                return torch.from_numpy(arr).view(torch.float16).float()
            return torch.tensor(arr, dtype=torch.float32)

        s = _to_f32(data['s'])
        z = _to_f32(data['z'])
        is_ligand = torch.tensor(data['is_ligand'], dtype=torch.float32)
        single_mask = torch.tensor(data['single_mask'], dtype=torch.float32)

        json_file = os.path.join(self.json_dir, f"{sample_id}.json")
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Missing JSON file for sample_id={sample_id}: {json_file}")
        try:
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            label = json_data.get('label', 0)
        except Exception as e:
            raise RuntimeError(f"Failed to parse JSON label for sample_id={sample_id}: {e}") from e

        result = {
            's': s, 'z': z, 'is_ligand': is_ligand,
            'single_mask': single_mask, 'label': label, 'sample_id': sample_id
        }

        # 添加 DRFP 特征
        drfp_loaded = False
        if self.drfp_dict is not None:
            drfp = np.zeros(2048, dtype=np.float32)
            # 优先从 JSON 读取 cano_rxn_smiles 作为 DRFP key
            drfp_key = None
            if hasattr(self, 'cano_rxn_mapping') and sample_id in self.cano_rxn_mapping:
                drfp_key = self.cano_rxn_mapping[sample_id]
            else:
                json_file = os.path.join(self.json_dir, f"{sample_id}.json")
                if os.path.exists(json_file):
                    try:
                        with open(json_file, 'r') as f:
                            _jd = json.load(f)
                        drfp_key = _jd.get('cano_rxn_smiles', '')
                    except Exception:
                        pass
            if drfp_key and drfp_key in self.drfp_dict:
                drfp = self.drfp_dict[drfp_key]
                drfp_loaded = True
            result['drfp'] = torch.tensor(drfp, dtype=torch.float32)

            # 统计 DRFP 加载情况
            if not hasattr(self, 'drfp_stats'):
                self.drfp_stats = {'total': 0, 'loaded': 0}
            self.drfp_stats['total'] += 1
            if drfp_loaded:
                self.drfp_stats['loaded'] += 1

        result['extra_features'] = (
            torch.from_numpy(self.external_features[idx].astype(np.float32, copy=False)).float()
            if self.external_features is not None else torch.zeros(0, dtype=torch.float32)
        )
        if self.context_store is not None:
            context_features, context_mask, context_scores, overall_corr = self.context_store.get(sample_id)
        else:
            context_features = torch.zeros((0, 0), dtype=torch.float32)
            context_mask = torch.zeros((0,), dtype=torch.float32)
            context_scores = torch.zeros((0,), dtype=torch.float32)
            overall_corr = torch.zeros((0,), dtype=torch.float32)
        result['context_features'] = context_features
        result['context_mask'] = context_mask
        result['context_scores'] = context_scores
        result['overall_corr'] = overall_corr

        return result


def load_model(
    checkpoint_path,
    device,
    use_drfp=None,
    drfp_dim=2048,
    drfp_injection='z_pair_pre_pairformer_global',
    drfp_pair_scale_init=1.0,
    compressed=None,
    hidden_size=None,
    n_transformer_layers=None,
    embedding_dir=None,
    context_dim=None,
    overall_corr_dim=None,
    resolved_config=None,
):
    """加载模型"""
    state_dict = load_checkpoint_state_dict(checkpoint_path, map_location=device)
    inferred = infer_checkpoint_config(
        state_dict,
        embedding_dir=embedding_dir,
        use_drfp=use_drfp,
        compressed=compressed,
        hidden_size=hidden_size,
        n_transformer_layers=n_transformer_layers,
    )
    use_drfp = inferred['use_drfp']
    compressed = inferred['compressed']
    hidden_size = inferred['hidden_size']
    n_transformer_layers = inferred['n_transformer_layers']
    pooling_mode = inferred['pooling_mode']
    use_extra_features = inferred['use_extra_features']
    extra_feature_dim = inferred['extra_feature_dim']
    extra_hidden_size = inferred['extra_hidden_size']
    extra_fusion_mode = inferred['extra_fusion_mode']
    extra_feature_branch_dims = inferred.get('extra_feature_branch_dims', [])
    extra_token_injection = False
    extra_branch_scale_init = 1.0
    main_input_mode = "sz"
    use_context_features = inferred['use_context_features']
    if context_dim is None:
        context_dim = inferred['context_dim']
    if overall_corr_dim is None:
        overall_corr_dim = inferred['overall_corr_dim']

    resolved_model_config = (resolved_config or {}).get('model', {}) if resolved_config else {}
    if resolved_model_config:
        pooling_mode = resolved_model_config.get('pooling_mode', pooling_mode)
        use_extra_features = bool(resolved_model_config.get('use_extra_features', use_extra_features))
        extra_feature_dim = int(resolved_model_config.get('extra_feature_dim', extra_feature_dim) or 0)
        extra_hidden_size = int(resolved_model_config.get('extra_hidden_size', extra_hidden_size) or hidden_size)
        extra_fusion_mode = str(resolved_model_config.get('extra_fusion_mode', extra_fusion_mode))
        extra_feature_branch_dims = [
            int(dim) for dim in resolved_model_config.get('extra_feature_branch_dims', extra_feature_branch_dims or []) if int(dim) > 0
        ]
        extra_token_injection = bool(resolved_model_config.get('extra_token_injection', False))
        extra_branch_scale_init = float(resolved_model_config.get('extra_branch_scale_init', 1.0))
        main_input_mode = str(resolved_model_config.get('main_input_mode', main_input_mode)).lower()
        context_dim = int(resolved_model_config.get('context_dim', context_dim or 0) or 0)
        use_context_features = bool(
            resolved_model_config.get('use_context_features', use_context_features or context_dim > 0)
        )
        if not use_context_features:
            context_dim = 0
        use_overall_corr = bool(
            resolved_model_config.get('use_overall_corr', (overall_corr_dim or 0) > 0)
        )
        if use_overall_corr:
            overall_corr_dim = int(
                resolved_model_config.get('overall_corr_dim', overall_corr_dim or inferred['overall_corr_dim']) or 0
            )
        else:
            overall_corr_dim = 0

    if not compressed or use_context_features:
        raise ValueError("This release supports the final compressed AlphaEnzyme architecture.")
    configs = {
        'c_s': 384, 'c_z': 128, 'hidden_size': hidden_size, 'dropout': 0.0,
        'n_transformer_layers': n_transformer_layers,
        'n_heads_single': 8, 'n_heads_pair': 4, 'blocks_per_ckpt': 1,
        'use_drfp': use_drfp,
        'drfp_dim': drfp_dim,
        'drfp_injection': drfp_injection,
        'drfp_pair_scale_init': drfp_pair_scale_init,
        'pooling_mode': pooling_mode,
        'global_pool_scale_init': 0.5,
        'main_input_mode': main_input_mode,
        'use_extra_features': use_extra_features,
        'extra_feature_dim': extra_feature_dim,
        'extra_hidden_size': extra_hidden_size,
        'extra_fusion_mode': extra_fusion_mode,
        'extra_feature_branch_dims': extra_feature_branch_dims,
        'extra_token_injection': extra_token_injection,
        'extra_branch_scale_init': extra_branch_scale_init,
        'context_dim': int(context_dim or 0),
        'overall_corr_dim': int(overall_corr_dim or 0),
    }
    from alphaenzyme.models.enzyme_classification_compressed import EnzymeClassificationHeadCompressed
    model = EnzymeClassificationHeadCompressed(configs)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        raise RuntimeError(f"Missing keys in checkpoint (would be randomly initialized): {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected keys in checkpoint (architecture mismatch): {unexpected_keys}")
    model = model.to(device)
    model.eval()
    return model


def compute_dcg(rels, k=None):
    if k is None:
        k = len(rels)
    else:
        k = min(k, len(rels))
    dcg = 0
    for i in range(k):
        dcg += (2**rels[i] - 1) / np.log2(i + 2)
    return dcg


def compute_ndcg(rels, k=None):
    rels = np.asarray(rels, dtype=np.float32)
    dcg = compute_dcg(rels, k=k)
    ideal_rels = np.sort(rels)[::-1]
    idcg = compute_dcg(ideal_rels, k=k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def calculate_enrichment_factor(labels_sorted, top_percent):
    n = len(labels_sorted)
    topk = max(int(top_percent * n), 5)
    num_active_top = labels_sorted[:topk].sum()
    total_active = labels_sorted.sum()
    if total_active == 0:
        return 0.0
    return num_active_top / (total_active * top_percent)


def recover_sample_ids_from_indices(indices, embedding_dir):
    """从 indices 回溯 sample_id"""
    npz_files = sorted([f for f in os.listdir(embedding_dir) if f.endswith('.npz')])
    sample_ids = []
    for idx in indices:
        sample_ids.append(npz_files[int(idx)].replace('.npz', ''))
    return np.array(sample_ids)


def normalize_optional_value(value):
    """将空值/NaN 统一成 None，其余转成字符串。"""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {'nan', 'none'}:
        return None
    return value


def build_reaction_mapping(csv_path):
    """从 CSV 建立 (uniprot_id, rhea_id) → CANO_RXN_SMILES 映射"""
    if csv_path is None or not os.path.exists(csv_path):
        print("Warning: CSV file not provided or not found, skipping reaction-level evaluation")
        return None

    import pandas as pd
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        uniprot_id = normalize_optional_value(row.get('UniprotID', row.get('enzyme')))
        rhea_id = normalize_optional_value(row.get('RHEA_ID'))
        cano_rxn_smiles = normalize_optional_value(row.get('CANO_RXN_SMILES', row.get('reaction')))
        if uniprot_id is None or rhea_id is None or cano_rxn_smiles is None:
            continue
        key = (uniprot_id, rhea_id)
        mapping[key] = cano_rxn_smiles
    return mapping


def build_sample_metadata_index(csv_path):
    """从 CSV 建立 sample_id -> 元信息 索引，用于还原官方去重顺序。"""
    if csv_path is None or not os.path.exists(csv_path):
        return {}

    import pandas as pd
    df = pd.read_csv(csv_path)
    metadata_index = {}
    for idx, row in df.iterrows():
        sample_id = str(row['id']) if 'id' in row.index else str(idx)
        metadata_index[sample_id] = {
            'row_order': idx,
            'reaction': normalize_optional_value(row.get('CANO_RXN_SMILES', row.get('reaction'))),
            'enzyme': normalize_optional_value(row.get('UniprotID', row.get('enzyme'))),
            'rhea_id': normalize_optional_value(row.get('RHEA_ID')),
        }
    return metadata_index


@lru_cache(maxsize=None)
def load_json_sample_metadata(json_dir, sample_id):
    """从 JSON 读取 reaction / enzyme / rhea 元信息。"""
    meta = {'reaction': None, 'enzyme': None, 'rhea_id': None}
    json_file = os.path.join(json_dir, f"{sample_id}.json")
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            meta['reaction'] = normalize_optional_value(data.get('cano_rxn_smiles'))
            meta['enzyme'] = normalize_optional_value(data.get('uniprot_id', data.get('enzyme')))
            meta['rhea_id'] = normalize_optional_value(data.get('rhea_id'))
        except Exception:
            return meta
    return meta


def resolve_sample_metadata(sample_id, sample_metadata_index, reaction_mapping, json_dir):
    """融合 CSV / JSON 元信息，得到 reaction 和 enzyme。"""
    base_meta = sample_metadata_index.get(sample_id, {})
    json_meta = load_json_sample_metadata(json_dir, sample_id)

    enzyme = base_meta.get('enzyme') or json_meta.get('enzyme')
    rhea_id = base_meta.get('rhea_id') or json_meta.get('rhea_id')
    reaction = base_meta.get('reaction') or json_meta.get('reaction')
    if reaction is None and reaction_mapping is not None and enzyme is not None and rhea_id is not None:
        reaction = reaction_mapping.get((enzyme, rhea_id))

    return {
        'sample_id': sample_id,
        'row_order': base_meta.get('row_order'),
        'reaction': reaction,
        'enzyme': enzyme,
    }


def deduplicate_by_reaction_enzyme(scores, labels, sample_ids, sample_metadata, enabled=True):
    """按 (reaction, enzyme) 去重，保留 CSV 原始顺序中的第一条。"""
    raw_count = len(scores)
    if not enabled:
        return scores, labels, sample_ids, sample_metadata, {
            'Duplicate Enzyme Groups Removed': 0,
            'Duplicate Rows Removed': 0,
        }

    ordered_records = []
    for idx, meta in enumerate(sample_metadata):
        row_order = meta.get('row_order')
        ordered_records.append((
            row_order is None,
            row_order if row_order is not None else idx,
            idx,
        ))
    ordered_records.sort()

    seen_keys = set()
    duplicate_groups = set()
    kept_indices = []
    for _, _, idx in ordered_records:
        meta = sample_metadata[idx]
        reaction = meta.get('reaction')
        enzyme = meta.get('enzyme')
        if reaction is None or enzyme is None:
            kept_indices.append(idx)
            continue

        dedup_key = (reaction, enzyme)
        if dedup_key in seen_keys:
            duplicate_groups.add(dedup_key)
            continue

        seen_keys.add(dedup_key)
        kept_indices.append(idx)

    kept_indices.sort(key=lambda idx: (
        sample_metadata[idx].get('row_order') is None,
        sample_metadata[idx].get('row_order') if sample_metadata[idx].get('row_order') is not None else idx,
        idx,
    ))

    dedup_scores = scores[kept_indices]
    dedup_labels = labels[kept_indices]
    dedup_sample_ids = sample_ids[kept_indices]
    dedup_metadata = [sample_metadata[idx] for idx in kept_indices]

    stats = {
        'Duplicate Enzyme Groups Removed': int(len(duplicate_groups)),
        'Duplicate Rows Removed': int(raw_count - len(dedup_scores)),
    }
    return dedup_scores, dedup_labels, dedup_sample_ids, dedup_metadata, stats


def evaluate_ranking_by_reaction(scores, labels, sample_ids, sample_metadata, k_values=[1, 3, 5, 10, 20, 50]):
    """按反应粒度评估排名性能"""
    reaction_groups = defaultdict(list)
    n_unmapped = 0
    for i, sid in enumerate(sample_ids):
        rxn_smiles = sample_metadata[i].get('reaction')
        if rxn_smiles is not None:
            reaction_groups[rxn_smiles].append((scores[i], labels[i], sid))
        else:
            n_unmapped += 1
    if n_unmapped > 0:
        print(f"Warning: {n_unmapped} samples could not be mapped to a reaction")

    n_reactions = len(reaction_groups)
    best_positive_ranks = []
    per_reaction_recall_at_k = {k: [] for k in k_values}
    reactions_with_positive = 0
    rankable_reactions_for_auc = 0
    per_reaction_auc = []
    per_reaction_ap = []
    results = {}

    if n_reactions == 0:
        for k in k_values:
            results[f'Top-{k} Success Rate (%)'] = 0.0
            results[f'Top-{k} Recall (%)'] = 0.0
        results['MRR'] = 0.0
        results['Mean Rank'] = 0.0
        results['Median Rank'] = 0.0
        results['DCG@10'] = 0.0
        results['NDCG@10'] = 0.0
        results['EF@1%'] = 0.0
        results['EF@2%'] = 0.0
        results['_n_total_reactions'] = 0
        results['_n_reactions_with_positive'] = 0
        results['_n_rankable_reactions_for_auc'] = 0
        print("\nWarning: No reaction groups available for reaction-level evaluation")
        return results

    for rxn_smiles, items in reaction_groups.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        item_scores = np.array([score for score, _, _ in items_sorted], dtype=np.float32)
        item_labels = np.array([label for _, label, _ in items_sorted], dtype=np.int32)
        pos_ranks = []
        for rank_idx, (score, label, sid) in enumerate(items_sorted):
            if label == 1:
                pos_ranks.append(rank_idx + 1)
        if pos_ranks:
            n_pos = len(pos_ranks)
            best_rank = min(pos_ranks)
            best_positive_ranks.append(best_rank)
            reactions_with_positive += 1
            for k in k_values:
                hits = sum(1 for r in pos_ranks if r <= k)
                per_reaction_recall_at_k[k].append(hits / n_pos)

        if 0 < item_labels.sum() < len(item_labels):
            rankable_reactions_for_auc += 1
            per_reaction_auc.append(roc_auc_score(item_labels, item_scores))
            per_reaction_ap.append(average_precision_score(item_labels, item_scores))

    best_positive_ranks = np.array(best_positive_ranks)

    print("\n" + "=" * 60)
    print("Reaction-Level Ranking Evaluation")
    print("=" * 60)
    print(f"Total reactions: {n_reactions}")
    print(f"Reactions with positive samples: {reactions_with_positive}")
    print(f"Rankable reactions for AUC/AP: {rankable_reactions_for_auc}")

    group_sizes = [len(v) for v in reaction_groups.values()]
    print(f"Candidates per reaction: mean={np.mean(group_sizes):.1f}, "
          f"median={np.median(group_sizes):.1f}, "
          f"min={np.min(group_sizes)}, max={np.max(group_sizes)}")

    print(f"\n--- Top-k Success Rate (reaction-level) ---")
    for k in k_values:
        success_rate = (np.mean(best_positive_ranks <= k) * 100) if len(best_positive_ranks) > 0 else 0.0
        results[f'Top-{k} Success Rate (%)'] = success_rate
        hits = int(np.sum(best_positive_ranks <= k)) if len(best_positive_ranks) > 0 else 0
        print(f"Top-{k} Success Rate: {success_rate:.2f}% ({hits}/{len(best_positive_ranks)})")

    print(f"\n--- Top-k Recall (reaction-level) ---")
    for k in k_values:
        recall = (np.mean(per_reaction_recall_at_k[k]) * 100) if per_reaction_recall_at_k[k] else 0.0
        results[f'Top-{k} Recall (%)'] = recall
        print(f"Top-{k} Recall: {recall:.2f}%")

    mrr = float(np.mean(1.0 / best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    results['MRR'] = mrr
    print(f"\nMRR: {mrr:.4f}")

    mean_rank = float(np.mean(best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    median_rank = float(np.median(best_positive_ranks)) if len(best_positive_ranks) > 0 else 0.0
    results['Mean Rank'] = mean_rank
    results['Median Rank'] = median_rank
    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Median Rank: {median_rank:.2f}")

    if per_reaction_auc:
        mean_auc = float(np.mean(per_reaction_auc))
        mean_ap = float(np.mean(per_reaction_ap))
        results['Mean AUC (per-reaction)'] = mean_auc
        results['Mean AP (per-reaction)'] = mean_ap
        print(f"Mean AUC (per-reaction): {mean_auc:.4f}")
        print(f"Mean AP (per-reaction): {mean_ap:.4f}")

    # DCG/NDCG@10 and EF@1%/2%
    dcg_list = []
    ndcg_list = []
    ef1_list = []
    ef2_list = []
    for rxn_smiles, items in reaction_groups.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        rels = np.array([lbl for _, lbl, _ in items_sorted], dtype=np.float32)
        dcg_list.append(compute_dcg(rels, k=10))
        ndcg_list.append(compute_ndcg(rels, k=10))
        ef1_list.append(calculate_enrichment_factor(rels, top_percent=0.01))
        ef2_list.append(calculate_enrichment_factor(rels, top_percent=0.02))

    dcg10 = float(np.mean(dcg_list)) if dcg_list else 0.0
    ndcg10 = float(np.mean(ndcg_list)) if ndcg_list else 0.0
    ef1 = float(np.mean(ef1_list)) if ef1_list else 0.0
    ef2 = float(np.mean(ef2_list)) if ef2_list else 0.0
    results['DCG@10'] = dcg10
    results['NDCG@10'] = ndcg10
    results['EF@1%'] = ef1
    results['EF@2%'] = ef2
    print(f"\nDCG@10: {dcg10:.4f}")
    print(f"NDCG@10: {ndcg10:.4f}")
    print(f"EF@1%: {ef1:.4f}")
    print(f"EF@2%: {ef2:.4f}")

    results['_n_total_reactions'] = n_reactions
    results['_n_reactions_with_positive'] = reactions_with_positive
    results['_n_rankable_reactions_for_auc'] = rankable_reactions_for_auc
    return results


def evaluate_global(scores, labels):
    """全局指标"""
    results = {}

    print("\n" + "=" * 60)
    print("Global Evaluation Results")
    print("=" * 60)

    try:
        auc = roc_auc_score(labels, scores)
    except Exception:
        auc = 0.0
    try:
        ap = average_precision_score(labels, scores)
    except Exception:
        ap = 0.0
    predictions = (scores > 0.5).astype(int)
    accuracy = np.mean(predictions == labels)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)

    results['AUC'] = auc
    results['Average Precision'] = ap
    results['Accuracy'] = accuracy
    results['Balanced Accuracy'] = balanced_acc
    results['F1'] = f1
    results['Precision'] = precision
    results['Recall'] = recall

    print(f"AUC: {auc:.4f}")
    print(f"Average Precision (PR-AUC): {ap:.4f}")
    print(f"Accuracy (threshold=0.5): {accuracy:.4f}")
    print(f"Balanced Accuracy (threshold=0.5): {balanced_acc:.4f}")
    print(f"F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    return results


def collect_and_evaluate(args):
    """收集所有GPU结果并进行评估"""
    print("\n" + "=" * 60)
    print("Collecting Results from All GPUs")
    print("=" * 60)

    # 等待所有GPU完成
    max_wait = 3600  # 最多等待1小时
    wait_interval = 10
    elapsed = 0

    while elapsed < max_wait:
        all_exist = True
        for rank in range(args.world_size):
            result_file = os.path.join(args.score_dir, f"eval_gpu_{rank}.npz")
            if not os.path.exists(result_file):
                all_exist = False
                break

        if all_exist:
            print("All GPU results are ready!")
            break

        time.sleep(wait_interval)
        elapsed += wait_interval
        if elapsed % 60 == 0:
            print(f"Waiting for other GPUs... ({elapsed}s elapsed)")

    if elapsed >= max_wait:
        print("Warning: Timeout waiting for other GPUs")
        return

    # 收集结果
    all_scores = []
    all_labels = []
    all_indices = []
    all_sample_ids = []
    has_sample_ids = True

    for rank in range(args.world_size):
        result_file = os.path.join(args.score_dir, f"eval_gpu_{rank}.npz")
        if os.path.exists(result_file):
            data = np.load(result_file, allow_pickle=True)
            all_scores.extend(data['scores'])
            all_labels.extend(data['labels'])
            all_indices.extend(data['indices'])
            if 'sample_ids' in data:
                all_sample_ids.extend(data['sample_ids'])
            else:
                has_sample_ids = False
            print(f"Loaded GPU {rank}: {len(data['scores'])} samples")
        else:
            print(f"Warning: Result file not found for GPU {rank}")

    if len(all_scores) == 0:
        raise RuntimeError("No results found")

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_indices = np.array(all_indices)

    # 按原始索引排序
    sort_idx = np.argsort(all_indices)
    final_scores = all_scores[sort_idx]
    final_labels = all_labels[sort_idx]

    # 恢复 sample_ids
    if has_sample_ids and len(all_sample_ids) == len(all_scores):
        final_sample_ids = np.array(all_sample_ids)[sort_idx]
    else:
        print("sample_ids not found in npz, recovering from indices...")
        sorted_indices = all_indices[sort_idx]
        final_sample_ids = recover_sample_ids_from_indices(sorted_indices, args.embedding_dir)

    raw_sample_count = len(final_scores)
    print(f"\nRaw samples: {raw_sample_count}")
    print(f"Raw positive: {np.sum(final_labels == 1)}, Raw negative: {np.sum(final_labels == 0)}")

    # 建立元信息并按 (reaction, enzyme) 去重，贴近 EnzymeCAGE 官方 evaluate.py
    reaction_mapping = build_reaction_mapping(args.csv_path)
    sample_metadata_index = build_sample_metadata_index(args.csv_path)
    resolved_sample_metadata = [
        resolve_sample_metadata(sample_id, sample_metadata_index, reaction_mapping, args.json_dir)
        for sample_id in final_sample_ids
    ]
    final_scores, final_labels, final_sample_ids, resolved_sample_metadata, dedup_stats = deduplicate_by_reaction_enzyme(
        final_scores,
        final_labels,
        final_sample_ids,
        resolved_sample_metadata,
        enabled=args.deduplicate_reaction_enzymes,
    )

    if args.deduplicate_reaction_enzymes:
        print(
            "Deduplicated reaction-level candidates by (reaction, enzyme): "
            f"removed {dedup_stats['Duplicate Rows Removed']} rows across "
            f"{dedup_stats['Duplicate Enzyme Groups Removed']} duplicate groups"
        )

    print(f"Evaluation samples after dedup: {len(final_scores)}")
    print(f"Positive: {np.sum(final_labels == 1)}, Negative: {np.sum(final_labels == 0)}")

    print("\nRunning reaction-level evaluation...")
    reaction_results = evaluate_ranking_by_reaction(
        final_scores, final_labels, final_sample_ids, resolved_sample_metadata
    )

    # 全局指标
    global_results = evaluate_global(final_scores, final_labels)

    # 合并并保存
    sample_stats = {
        'Raw Samples': int(raw_sample_count),
        'Total Samples': int(len(final_scores)),
        'Positive Samples': int(np.sum(final_labels == 1)),
        'Negative Samples': int(np.sum(final_labels == 0)),
        'Evaluation Precision': str(args.eval_precision).lower(),
        'Evaluation Seed': int(args.seed),
        'Specificity Score Head': (
            'alpha_repr_specificity_head' if getattr(args, 'use_alpha_repr_specificity_head', False) else 'classifier_logits'
        ),
        'DRFP Input Zeroed': bool(getattr(args, 'zero_drfp_input', False)),
        'DRFP Similarity Gate': getattr(args, '_drfp_gate_stats', {"enabled": False}),
        'Reaction-Level Enzyme Deduplication': bool(args.deduplicate_reaction_enzymes),
        **dedup_stats,
    }

    if reaction_results:
        sample_stats['Total Reactions'] = int(reaction_results.pop('_n_total_reactions', 0))
        sample_stats['Reactions With Positive Samples'] = int(reaction_results.pop('_n_reactions_with_positive', 0))
        sample_stats['Rankable Reactions For AUC'] = int(reaction_results.pop('_n_rankable_reactions_for_auc', 0))

    all_results = {**sample_stats, **reaction_results, **global_results}
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ranking_evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--embedding_dir', type=str, required=True, help='Path to embedding directory')
    parser.add_argument('--json_dir', type=str, required=True, help='Path to JSON directory')
    parser.add_argument('--use_drfp', dest='use_drfp', action='store_true', help='Force-enable DRFP features')
    parser.add_argument('--no_drfp', dest='use_drfp', action='store_false', help='Force-disable DRFP features')
    parser.add_argument('--drfp_dim', type=int, default=2048, help='DRFP feature dimension')
    parser.add_argument('--drfp_injection', type=str, default='z_pair_pre_pairformer_global',
                        help='DRFP injection mode used by the checkpoint')
    parser.add_argument('--drfp_pair_scale_init', type=float, default=1.0,
                        help='Initial DRFP pair scale used when constructing the model')
    parser.add_argument('--zero_drfp_input', action='store_true',
                        help='Keep the DRFP model branch enabled but feed zero DRFP vectors at inference')
    parser.add_argument('--drfp_similarity_gate_train_csv', action='append', default=None,
                        help='CSV containing training reactions used as the DRFP similarity bank; can be repeated')
    parser.add_argument('--drfp_similarity_gate_low', type=float, default=0.30,
                        help='Max-Tanimoto value mapped to zero DRFP weight')
    parser.add_argument('--drfp_similarity_gate_high', type=float, default=0.70,
                        help='Max-Tanimoto value mapped to full DRFP weight')
    parser.add_argument('--drfp_similarity_gate_power', type=float, default=1.0,
                        help='Power applied to the linear DRFP similarity gate weight')
    parser.add_argument('--drfp_path', type=str, default=os.path.join(DATA_ROOT, 'inputs', 'specificity', 'features', 'drfp', 'reaction_fingerprints.pkl'),
                        help='Path to DRFP pickle file')
    parser.add_argument('--score_dir', type=str, default=os.path.join(OUTPUT_ROOT, 'outputs', 'score'),
                        help='Directory to save scores')
    parser.add_argument('--output_dir', type=str, default=os.path.join(OUTPUT_ROOT, 'outputs', 'eval'),
                        help='Directory to save evaluation results')
    parser.add_argument('--csv_path', type=str, default=None, help='Path to CSV file for reaction mapping (optional)')
    parser.add_argument('--world_size', type=int, default=1, help='Total number of GPUs')
    parser.add_argument('--rank', type=int, default=None,
                        help='Data shard rank; defaults to 0 for single-process runs and GPU id for multi-GPU runs')
    parser.add_argument('--compressed', dest='compressed', action='store_true',
                        help='Force use compressed model (EnzymeClassificationHeadCompressed)')
    parser.add_argument('--hidden_size', type=int, default=None, help='Model hidden size (auto-detected when omitted)')
    parser.add_argument('--n_transformer_layers', type=int, default=None,
                        help='Number of transformer layers (auto-detected when omitted)')
    parser.add_argument('--deduplicate_reaction_enzymes', dest='deduplicate_reaction_enzymes', action='store_true',
                        help='Deduplicate candidates within each reaction by UniprotID before evaluation')
    parser.add_argument('--no_deduplicate_reaction_enzymes', dest='deduplicate_reaction_enzymes', action='store_false',
                        help='Disable reaction-level enzyme deduplication (debug only)')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Optional resolved_config.yaml path; defaults to checkpoint directory when present')
    parser.add_argument('--external_feature_bank', action='append', default=None,
                        help='Optional external feature bank path; can be repeated')
    parser.add_argument('--external_include_z', action='store_true', help='Include single and pair features from the external bank')
    parser.add_argument('--context_cache_path', type=str, default=None,
                        help='Optional direct context cache path')
    parser.add_argument('--context_bank_path', type=str, default=None,
                        help='Optional direct context bank path')
    parser.add_argument('--eval_precision', type=str, default='fp32', choices=['fp32', 'bf16', 'fp16'],
                        help='Forward precision for offline evaluation; fp32 is the official comparable setting')
    parser.add_argument(
        '--use_alpha_repr_specificity_head',
        action='store_true',
        help='Score Enzyme-405 with checkpoint["specificity_alpha_head"](output["alpha_repr"]) instead of output["logits"]',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed used for deterministic offline evaluation')
    parser.set_defaults(use_drfp=None, compressed=None, deduplicate_reaction_enzymes=True)
    args = parser.parse_args()
    args._drfp_gate_stats = {"enabled": False}
    set_eval_reproducibility(seed=args.seed, deterministic=True)

    checkpoint_state_dict = load_checkpoint_state_dict(args.checkpoint, map_location='cpu')
    inferred = infer_checkpoint_config(
        checkpoint_state_dict,
        embedding_dir=args.embedding_dir,
        use_drfp=args.use_drfp,
        compressed=args.compressed,
        hidden_size=args.hidden_size,
        n_transformer_layers=args.n_transformer_layers,
    )
    args.use_drfp = inferred['use_drfp']
    args.compressed = inferred['compressed']
    args.hidden_size = inferred['hidden_size']
    args.n_transformer_layers = inferred['n_transformer_layers']

    resolved_config = {}
    config_path = args.config_path
    if not config_path:
        auto_config_path = os.path.join(os.path.dirname(args.checkpoint), 'resolved_config.yaml')
        if os.path.exists(auto_config_path):
            config_path = auto_config_path
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            resolved_config = yaml.safe_load(f) or {}
        print(f"Loaded resolved config from {config_path}")
    external_feature_banks = normalize_bank_specs(args.external_feature_bank)
    if args.external_include_z:
        for spec in external_feature_banks: spec['include_z'] = True
    if not external_feature_banks and resolved_config:
        external_feature_banks = normalize_bank_specs(
            resolved_config.get('data', {}).get('test_external_feature_banks')
            or resolved_config.get('data', {}).get('external_feature_banks')
        )
    external_missing_policy = str(
        resolved_config.get('data', {}).get('external_missing_policy', 'error') if resolved_config else 'error'
    ).lower()
    context_cache_path = args.context_cache_path or (
        resolved_config.get('data', {}).get('test_context_cache_path') if resolved_config else None
    )
    context_bank_path = args.context_bank_path or (
        resolved_config.get('data', {}).get('test_context_bank_path')
        or resolved_config.get('data', {}).get('context_bank_path')
        if resolved_config else None
    )
    context_missing_policy = str(
        resolved_config.get('data', {}).get('context_missing_policy', 'error') if resolved_config else 'error'
    ).lower()

    device = f'cuda:{args.gpu}'
    print(f"Using device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Embedding dir: {args.embedding_dir}")
    print(f"JSON dir: {args.json_dir}")
    print(f"Use DRFP: {args.use_drfp}")
    print(f"Use compressed model: {args.compressed}")
    print(f"Deduplicate reaction enzymes: {args.deduplicate_reaction_enzymes}")
    print(f"Eval precision: {args.eval_precision}")
    print(f"Eval seed: {args.seed}")
    print(f"Inferred hidden_size: {args.hidden_size}, n_transformer_layers: {args.n_transformer_layers}")
    print(f"Context cache: {context_cache_path}")
    print(f"Context bank: {context_bank_path}")

    os.makedirs(args.score_dir, exist_ok=True)

    # 加载 DRFP 数据
    drfp_dict = None
    if args.use_drfp:
        if args.zero_drfp_input:
            print("Using zero DRFP input for every sample")
            # Keep the checkpoint architecture with its DRFP branch, but force
            # all DRFP vectors to zero.  This is different from --no_drfp,
            # which constructs a model without DRFP parameters.
            drfp_dict = {}
        elif os.path.exists(args.drfp_path):
            print(f"Loading DRFP from {args.drfp_path}")
            with open(args.drfp_path, 'rb') as f:
                drfp_dict = pickle.load(f)
            print(f"Loaded DRFP for {len(drfp_dict)} samples")
            if args.drfp_similarity_gate_train_csv:
                drfp_dict, args._drfp_gate_stats = build_similarity_gated_drfp_dict(
                    drfp_dict=drfp_dict,
                    gate_train_csvs=args.drfp_similarity_gate_train_csv,
                    eval_csv_path=args.csv_path,
                    low=args.drfp_similarity_gate_low,
                    high=args.drfp_similarity_gate_high,
                    power=args.drfp_similarity_gate_power,
                )
        else:
            raise FileNotFoundError(args.drfp_path)

    # 加载数据和模型
    dataset = EmbeddingDataset(
        args.embedding_dir,
        args.json_dir,
        drfp_dict,
        args.csv_path,
        external_feature_banks=external_feature_banks,
        external_missing_policy=external_missing_policy,
        context_cache_path=context_cache_path,
        context_bank_path=context_bank_path,
        context_missing_policy=context_missing_policy,
    )
    model = load_model(
        args.checkpoint,
        device,
        args.use_drfp,
        drfp_dim=args.drfp_dim,
        drfp_injection=args.drfp_injection,
        drfp_pair_scale_init=args.drfp_pair_scale_init,
        compressed=args.compressed,
        hidden_size=args.hidden_size,
        n_transformer_layers=args.n_transformer_layers,
        embedding_dir=args.embedding_dir,
        context_dim=getattr(dataset.context_store, "context_dim", 0) if getattr(dataset, "context_store", None) is not None else 0,
        overall_corr_dim=None,
        resolved_config=resolved_config,
    )
    specificity_alpha_bundle = (
        load_specificity_alpha_head(args.checkpoint, device)
        if args.use_alpha_repr_specificity_head
        else None
    )
    specificity_alpha_head = specificity_alpha_bundle[0] if specificity_alpha_bundle is not None else None
    specificity_alpha_residual_scale = (
        specificity_alpha_bundle[1] if specificity_alpha_bundle is not None else torch.tensor(0.0, device=device)
    )

    # 分割数据给多个GPU
    rank, start_idx, end_idx = partition_indices(
        len(dataset), args.gpu, args.world_size, args.rank
    )
    indices = list(range(start_idx, end_idx))

    print(f"Rank {rank}: processing indices {start_idx} to {end_idx} ({len(indices)} samples)")

    all_scores = []
    all_labels = []
    all_original_indices = []
    all_sample_ids = []

    # 推理 - 用 DataLoader 多线程预取
    from torch.utils.data import Subset, DataLoader

    def collate_fn(batch):
        return batch

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        num_workers=4, prefetch_factor=4, pin_memory=True,
                        collate_fn=collate_fn)
    for position, batch_items in enumerate(tqdm(loader, desc=f"GPU {rank}")):
        try:
            item = batch_items[0]
            orig_idx = indices[position]
            s = item['s'].unsqueeze(0).to(device)
            z = item['z'].unsqueeze(0).to(device)
            is_ligand = item['is_ligand'].unsqueeze(0).to(device)
            single_mask = item['single_mask'].unsqueeze(0).to(device)

            with torch.no_grad():
                extra_features = item['extra_features'].unsqueeze(0).to(device)
                context_features = item['context_features'].unsqueeze(0).to(device)
                context_mask = item['context_mask'].unsqueeze(0).to(device)
                context_scores = item['context_scores'].unsqueeze(0).to(device)
                overall_corr = item['overall_corr'].unsqueeze(0).to(device)
                with build_eval_autocast(args.eval_precision, device_type='cuda'):
                    if args.use_drfp:
                        drfp = item['drfp'].unsqueeze(0).to(device)
                        output = model(
                            s,
                            z,
                            is_ligand,
                            single_mask,
                            drfp=drfp,
                            extra_features=extra_features if extra_features.shape[-1] > 0 else None,
                            context_features=context_features if context_features.shape[-1] > 0 else None,
                            context_mask=context_mask if context_mask.shape[-1] > 0 else None,
                            context_scores=context_scores if context_scores.shape[-1] > 0 else None,
                            overall_corr=overall_corr if overall_corr.shape[-1] > 0 else None,
                        )
                    else:
                        output = model(
                            s,
                            z,
                            is_ligand,
                            single_mask,
                            extra_features=extra_features if extra_features.shape[-1] > 0 else None,
                            context_features=context_features if context_features.shape[-1] > 0 else None,
                            context_mask=context_mask if context_mask.shape[-1] > 0 else None,
                            context_scores=context_scores if context_scores.shape[-1] > 0 else None,
                            overall_corr=overall_corr if overall_corr.shape[-1] > 0 else None,
                        )
                if specificity_alpha_head is not None:
                    logit = specificity_alpha_head(output["alpha_repr"]) + specificity_alpha_residual_scale * output["logits"]
                else:
                    logit = output["logits"]
                score = torch.sigmoid(logit).cpu().item()

            all_scores.append(score)
            all_labels.append(item['label'])
            all_original_indices.append(orig_idx)
            all_sample_ids.append(item['sample_id'])
        except Exception as e:
            raise RuntimeError(f"Inference failed at input position {position}: {e}") from e

    # 保存结果
    result_file = os.path.join(args.score_dir, f"eval_gpu_{rank}.npz")
    np.savez(result_file,
             scores=np.array(all_scores),
             labels=np.array(all_labels),
             indices=np.array(all_original_indices),
             sample_ids=np.array(all_sample_ids))
    print(f"GPU {rank}: Saved {len(all_scores)} results to {result_file}")

    # 打印 DRFP 加载统计
    if args.use_drfp and hasattr(dataset, 'drfp_stats'):
        stats = dataset.drfp_stats
        loaded_pct = (stats['loaded'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"GPU {rank}: DRFP loaded: {stats['loaded']}/{stats['total']} ({loaded_pct:.1f}%)")

    # GPU 0 负责收集和评估所有结果
    if rank == 0:
        collect_and_evaluate(args)


if __name__ == "__main__":
    main()
