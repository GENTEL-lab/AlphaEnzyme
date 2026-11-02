#!/usr/bin/env python
"""
Train the AlphaEnzyme specificity classifier on the 50w enzyme-reaction table.

The classifier consumes compressed Pairformer token features and an optional
protein-pocket feature bank. Training supports pairwise ranking, BCE and hybrid
objectives, gradient accumulation, EMA and optional multi-GPU execution through
`accelerate`.
"""

import os
import sys

# 设置 PYTHONPATH
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BUNDLE_ROOT = os.path.dirname(PROJECT_ROOT)
DATA_ROOT = os.path.join(BUNDLE_ROOT, "data")
OUTPUT_ROOT = os.path.join(BUNDLE_ROOT, "runtime_outputs")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_party', 'protenix_runtime'))

import argparse
import yaml
import logging
import random
from datetime import datetime
import pandas as pd
import json
import glob
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from accelerate import Accelerator, DistributedDataParallelKwargs
from easydict import EasyDict
try:
    import swanlab
except ImportError:  # optional experiment logger
    swanlab = None
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import get_cosine_schedule_with_warmup
from alphaenzyme.utils.ema_utils import EMAWrapperV1
from alphaenzyme.data.direct_feature_support import DirectContextAssetStore
from alphaenzyme.data.external_feature_bank import load_aligned_external_feature_matrix, normalize_bank_specs

# 显存优化配置 - 减少显存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'


# ==================== Logger 设置 ====================
def setup_logger(name: str, log_dir: str, log_filename: str = None):
    """
    设置 logger，同时输出到控制台和文件

    Args:
        name: logger 名称
        log_dir: 日志目录
        log_filename: 日志文件名（默认带时间戳）

    Returns:
        logger 对象
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"train_{timestamp}.log"

    log_path = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件 handler
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 控制台 handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def set_global_reproducibility(seed: int, deterministic: bool, logger=None):
    """固定全链路随机源，尽量保证同硬件环境下可复现。"""
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

    if logger is not None:
        logger.info(f"Reproducibility setup: seed={seed}, deterministic={deterministic}")

from alphaenzyme.models.enzyme_kinetics import EnzymeKineticsHead
from alphaenzyme.models.enzyme_classification import EnzymeClassificationHead
from alphaenzyme.models.enzyme_classification_compressed import EnzymeClassificationHeadCompressed
from alphaenzyme.models.enzyme_classification_compressed_direct_context import EnzymeClassificationHeadCompressedDirectContext
from alphaenzyme.loss.enzyme_kinetics_loss import MaskedEnzymeKineticsLoss, BinaryClassificationLoss
from alphaenzyme.utils.dropout import Dropout


def set_dropout_rate(model, rate_scale):
    """Scale the dropout rate of every Dropout module by `rate_scale`."""
    for module in model.modules():
        if isinstance(module, Dropout):
            module.dropout.p = module.r * rate_scale


class SimpleEmbeddingDataset(Dataset):
    """简单的 embedding 数据集"""
    def __init__(self, csv_path, embedding_dir):
        self.data = pd.read_csv(csv_path)
        self.embedding_dir = embedding_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample_id = row['sample_id']

        # 加载 embedding (npz format - bf16)
        emb_path = os.path.join(self.embedding_dir, f"{sample_id}.npz")
        emb_data = np.load(emb_path)
        # Convert stored float16 bits (uint16) back to float32 for model
        def _to_f32(arr):
            if arr.dtype == np.uint16:
                return torch.from_numpy(arr).view(torch.float16).float()
            return torch.from_numpy(arr.astype(np.float32))
        emb = {
            's': _to_f32(emb_data['s']),
            'z': _to_f32(emb_data['z']),
            'is_ligand': torch.from_numpy(emb_data['is_ligand'].astype(np.float32)),
            'single_mask': torch.from_numpy(emb_data['single_mask'].astype(np.float32)),
        }

        # 准备标签 - 需要 log 形式
        labels = {}
        labels['has_km'] = torch.tensor(1 if row.get('has_km', False) else 0, dtype=torch.long)
        labels['has_kcat'] = torch.tensor(1 if row.get('has_kcat', False) else 0, dtype=torch.long)
        labels['has_kcat_km'] = torch.tensor(1 if row.get('has_kcat_km', False) else 0, dtype=torch.long)

        if 'has_km' in row and row['has_km']:
            labels['km'] = torch.tensor(row['km'], dtype=torch.float32)
            labels['log_km'] = torch.tensor(row['km'], dtype=torch.float32).log()
        else:
            labels['km'] = torch.tensor(0.0, dtype=torch.float32)
            labels['log_km'] = torch.tensor(0.0, dtype=torch.float32)

        if 'has_kcat' in row and row['has_kcat']:
            labels['kcat'] = torch.tensor(row['kcat'], dtype=torch.float32)
            labels['log_kcat'] = torch.tensor(row['kcat'], dtype=torch.float32).log()
        else:
            labels['kcat'] = torch.tensor(0.0, dtype=torch.float32)
            labels['log_kcat'] = torch.tensor(0.0, dtype=torch.float32)

        if 'has_kcat_km' in row and row['has_kcat_km']:
            labels['kcat_km'] = torch.tensor(row['kcat_km'], dtype=torch.float32)
            labels['log_kcat_km'] = torch.tensor(row['kcat_km'], dtype=torch.float32).log()
        else:
            labels['kcat_km'] = torch.tensor(0.0, dtype=torch.float32)
            labels['log_kcat_km'] = torch.tensor(0.0, dtype=torch.float32)

        # 准备条件
        ph = row.get('ph', 7.0) if pd.notna(row.get('ph', 7.0)) else 7.0
        temp = row.get('temp', 25.0) if pd.notna(row.get('temp', 25.0)) else 25.0
        condition = torch.tensor([ph, temp], dtype=torch.float32)

        return {
            's': emb['s'],
            'z': emb['z'],
            'is_ligand': emb['is_ligand'],
            'single_mask': emb['single_mask'],
            'labels': labels,
            'condition': condition,
        }


class BinaryClassificationDataset(Dataset):
    """二分类 embedding 数据集 - 从JSON目录读取"""
    def __init__(
        self,
        json_dir,
        embedding_dir,
        csv_path=None,
        drfp_path=None,
        max_seq_length=None,
        external_feature_banks=None,
        external_missing_policy="error",
        context_cache_path=None,
        context_bank_path=None,
        context_missing_policy="error",
    ):
        """
        Args:
            json_dir: Directory of per-pair JSON files
            embedding_dir: embedding文件目录 (*.pt)
            csv_path: CSV文件路径 (用于映射 uniprot_id + ec_number -> reaction_smiles)
            drfp_path: DRFP pickle文件路径
            max_seq_length: 最大序列长度限制，超过此长度的样本将被过滤 (默认: None, 不过滤)
        """
        self.json_dir = json_dir
        self.embedding_dir = embedding_dir
        self.max_seq_length = max_seq_length
        self.external_features = None
        self.external_feature_dim = 0
        self.external_feature_meta = []
        self.context_store = None
        self.context_len = 0
        self.context_dim = 0

        # 加载 DRFP 数据
        self.drfp_data = {}
        self.rxn_mapping = {}
        if drfp_path:
            import pickle
            with open(drfp_path, 'rb') as f:
                self.drfp_data = pickle.load(f)
            print(f"Loaded {len(self.drfp_data)} DRFP features from {drfp_path}")

            # 加载 CSV 建立 sample_id -> CANO_RXN_SMILES 映射
            if csv_path and os.path.exists(csv_path):
                import pandas as pd
                df = pd.read_csv(csv_path)
                for idx, row in df.iterrows():
                    sample_id = str(row['id']) if 'id' in row.index else str(idx)
                    if pd.notna(row.get('CANO_RXN_SMILES')):
                        self.rxn_mapping[sample_id] = str(row['CANO_RXN_SMILES'])
                print(f"Loaded {len(self.rxn_mapping)} reaction mappings from {csv_path}")
            else:
                print("No CSV mapping provided, will read CANO_RXN_SMILES directly from JSON")

        # 扫描所有json文件
        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        print(f"Found {len(json_files)} JSON files in {json_dir}")

        # 预先扫描可用的 embedding 文件
        emb_files = glob.glob(os.path.join(embedding_dir, "*.npz"))
        available_emb_ids = set(os.path.splitext(os.path.basename(f))[0] for f in emb_files)
        print(f"Found {len(available_emb_ids)} embedding files")

        # 建立 sample_id -> json_path 的映射，只保留有 embedding 的样本
        self.samples = []
        filtered_by_length = 0
        for json_path in json_files:
            sample_id = os.path.splitext(os.path.basename(json_path))[0]
            if sample_id in available_emb_ids:
                # 如果设置了长度限制，检查序列长度
                if self.max_seq_length is not None:
                    try:
                        with open(json_path, 'r') as f:
                            data = json.load(f)
                        if 'sequences' in data and len(data['sequences']) > 0:
                            protein_seq = data['sequences'][0].get('proteinChain', {}).get('sequence', '')
                            seq_len = len(protein_seq)
                            if seq_len > self.max_seq_length:
                                filtered_by_length += 1
                                continue
                    except Exception as e:
                        print(f"Warning: Failed to check length for {sample_id}: {e}")
                        continue

                self.samples.append({
                    'sample_id': sample_id,
                    'json_path': json_path,
                })

        print(f"Loaded {len(self.samples)} samples with matching embeddings")
        if self.max_seq_length is not None:
            print(f"Filtered out {filtered_by_length} samples with length > {self.max_seq_length}")
        ordered_sample_ids = [str(sample["sample_id"]) for sample in self.samples]
        external_features, external_meta = load_aligned_external_feature_matrix(
            sample_ids=ordered_sample_ids,
            bank_specs=external_feature_banks,
            missing_policy=external_missing_policy,
        )
        if external_features is not None:
            self.external_features = external_features
            self.external_feature_dim = int(external_features.shape[1])
            self.external_feature_meta = external_meta
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
            self.context_len = int(self.context_store.context_len)
            self.context_dim = int(self.context_store.context_dim)
            print(f"Loaded direct context assets: len={self.context_len}, dim={self.context_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_id = sample['sample_id']

        # 加载 embedding (npz format - bf16)
        emb_path = os.path.join(self.embedding_dir, f"{sample_id}.npz")
        try:
            emb_data = np.load(emb_path)
        except Exception as e:
            # 损坏的 npz 文件（如 tar 不完整），返回零向量
            print(f"Warning: Failed to load {emb_path}: {e}, returning zeros")
            # 用随机选一个好样本替代（递归找下一个）
            return self.__getitem__((idx + 1) % len(self.samples))
        # Convert to float32; s/z may be stored as uint16 (raw float16 bytes)
        def _to_f32(arr):
            if arr.dtype == np.uint16:
                return torch.from_numpy(arr).view(torch.float16).float()
            return torch.from_numpy(arr.astype(np.float32))
        emb = {
            's': _to_f32(emb_data['s']),
            'z': _to_f32(emb_data['z']),
            'is_ligand': torch.from_numpy(emb_data['is_ligand'].astype(np.float32)),
            'single_mask': torch.from_numpy(emb_data['single_mask'].astype(np.float32)),
        }

        # 加载 label
        with open(sample['json_path'], 'r') as f:
            data = json.load(f)

        label = data.get('label', 0)  # 默认为0

        # 获取 DRFP
        drfp = torch.zeros(2048, dtype=torch.float32)  # 默认零向量
        if self.drfp_data:
            # 优先用 CSV 映射，否则直接从 JSON 读 CANO_RXN_SMILES
            if self.rxn_mapping:
                rxn_smiles = self.rxn_mapping.get(sample_id)
            else:
                rxn_smiles = data.get('cano_rxn_smiles')
            if rxn_smiles and rxn_smiles in self.drfp_data:
                drfp = torch.from_numpy(self.drfp_data[rxn_smiles]).float()

        if self.context_store is not None:
            context_features, context_mask, context_scores, overall_corr = self.context_store.get(sample_id)
        else:
            context_features = torch.zeros((0, 0), dtype=torch.float32)
            context_mask = torch.zeros((0,), dtype=torch.float32)
            context_scores = torch.zeros((0,), dtype=torch.float32)
            overall_corr = torch.zeros((0,), dtype=torch.float32)

        return {
            's': emb['s'],
            'z': emb['z'],
            'is_ligand': emb['is_ligand'],
            'single_mask': emb['single_mask'],
            'label': torch.tensor(label, dtype=torch.long),  # 0 或 1
            'sample_id': sample_id,
            'drfp': drfp,
            'extra_features': torch.from_numpy(self.external_features[idx].astype(np.float32, copy=False)).float()
            if self.external_features is not None else torch.zeros(0, dtype=torch.float32),
            'context_features': context_features,
            'context_mask': context_mask,
            'context_scores': context_scores,
            'overall_corr': overall_corr,
        }


def binary_collate_fn(batch):
    """二分类的 collate 函数"""
    s_list = []
    z_list = []
    is_ligand_list = []
    single_mask_list = []
    labels = []
    sample_ids = []
    drfp_list = []
    extra_feature_list = []
    context_feature_list = []
    context_mask_list = []
    context_score_list = []
    overall_corr_list = []

    for item in batch:
        s_list.append(item['s'])
        z_list.append(item['z'])
        is_ligand_list.append(item['is_ligand'])
        single_mask_list.append(item['single_mask'])
        labels.append(item['label'])
        sample_ids.append(item['sample_id'])
        drfp_list.append(item['drfp'])
        extra_feature_list.append(item['extra_features'])
        context_feature_list.append(item['context_features'])
        context_mask_list.append(item['context_mask'])
        context_score_list.append(item['context_scores'])
        overall_corr_list.append(item['overall_corr'])

    # Padding 到最大长度
    max_len = max(s.shape[0] for s in s_list)

    s_padded = []
    z_padded = []
    is_ligand_padded = []
    single_mask_padded = []

    for s, z, is_ligand, single_mask in zip(s_list, z_list, is_ligand_list, single_mask_list):
        pad_len = max_len - s.shape[0]
        if pad_len > 0:
            # s padding
            s_padded.append(torch.cat([s, torch.zeros(pad_len, s.shape[1])], dim=0))
            # z padding: detect 2D (compressed) vs 3D (full)
            if z.dim() == 2:
                # compressed: [N, C_z] -> [N+pad, C_z]
                z_padded.append(torch.cat([z, torch.zeros(pad_len, z.shape[1])], dim=0))
            else:
                # full: [N, N, C_z] -> [N+pad, N+pad, C_z]
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
        's': torch.stack(s_padded),
        'z': torch.stack(z_padded),
        'is_ligand': torch.stack(is_ligand_padded),
        'single_mask': torch.stack(single_mask_padded),
        'labels': torch.stack(labels),
        'sample_ids': sample_ids,
        'drfp': torch.stack(drfp_list),  # [B, 2048]
        'extra_features': torch.stack(extra_feature_list),
        'context_features': torch.stack(context_feature_list),
        'context_mask': torch.stack(context_mask_list),
        'context_scores': torch.stack(context_score_list),
        'overall_corr': torch.stack(overall_corr_list),
    }


def kinetics_collate_fn(batch):
    """简单的 collate 函数"""
    s_list = []
    z_list = []
    is_ligand_list = []
    single_mask_list = []
    labels = {'km': [], 'kcat': [], 'kcat_km': [], 'has_km': [], 'has_kcat': [], 'has_kcat_km': []}
    conditions = []

    for item in batch:
        s_list.append(item['s'])
        z_list.append(item['z'])
        is_ligand_list.append(item['is_ligand'])
        single_mask_list.append(item['single_mask'])
        labels['km'].append(item['labels']['km'])
        labels['kcat'].append(item['labels']['kcat'])
        labels['kcat_km'].append(item['labels']['kcat_km'])
        labels['has_km'].append(item['labels']['has_km'])
        labels['has_kcat'].append(item['labels']['has_kcat'])
        labels['has_kcat_km'].append(item['labels']['has_kcat_km'])
        conditions.append(item['condition'])

    # Padding 到最大长度
    max_len = max(s.shape[0] for s in s_list)

    s_padded = []
    z_padded = []
    is_ligand_padded = []
    single_mask_padded = []

    for s, z, is_ligand, single_mask in zip(s_list, z_list, is_ligand_list, single_mask_list):
        pad_len = max_len - s.shape[0]
        if pad_len > 0:
            # s padding
            s_padded.append(torch.cat([s, torch.zeros(pad_len, s.shape[1])], dim=0))
            # z padding: detect 2D (compressed) vs 3D (full)
            if z.dim() == 2:
                # compressed: [N, C_z] -> [N+pad, C_z]
                z_padded.append(torch.cat([z, torch.zeros(pad_len, z.shape[1])], dim=0))
            else:
                # full: [N, N, C_z] -> [N+pad, N+pad, C_z]
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
        's': torch.stack(s_padded),
        'z': torch.stack(z_padded),
        'is_ligand': torch.stack(is_ligand_padded),
        'single_mask': torch.stack(single_mask_padded),
        'labels': {k: torch.stack(v) for k, v in labels.items()},
        'conditions': torch.stack(conditions),
    }


def gather_sample_ids(accelerator, sample_ids, is_distributed):
    """收集所有进程的 sample_ids（分布式时），用于 reaction-level 评估。"""
    if not is_distributed:
        return list(sample_ids)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return list(sample_ids)
    gathered = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(gathered, list(sample_ids))
    merged = []
    for ids in gathered:
        if ids is not None:
            merged.extend(ids)
    return merged


def gather_tensor_for_metrics(accelerator, tensor, is_distributed):
    """收集可变 batch 大小的 metric tensor；pairwise 分组 batch 在各 rank 长度可不同。"""
    tensor = tensor.detach().reshape(-1).float().cpu()
    if not is_distributed:
        return tensor
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return tensor
    gathered = [None for _ in range(accelerator.num_processes)]
    torch.distributed.all_gather_object(gathered, tensor)
    chunks = [item for item in gathered if item is not None and item.numel() > 0]
    if not chunks:
        return torch.empty(0, dtype=tensor.dtype)
    return torch.cat(chunks, dim=0)


def normalize_optional_value(value):
    """统一处理空值/NaN，并返回去空格后的字符串。"""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {'nan', 'none'}:
        return None
    return value


def compute_reaction_topk_success(scores, labels, sample_ids, json_dir, k_values=(1, 3, 5, 10, 20, 50)):
    """复用官方去重口径的 Top-k Success Rate。"""
    sr_default = {k: 0.0 for k in k_values}
    reaction_metrics = compute_reaction_ranking_metrics(
        scores,
        labels,
        sample_ids,
        json_dir,
        k_values=k_values,
    )
    return reaction_metrics.get("sr", sr_default)


def get_sample_csv_metadata_mapping(csv_path, csv_mapping_cache=None):
    if not csv_path or not os.path.exists(csv_path):
        return {}
    if csv_mapping_cache is None:
        csv_mapping_cache = {}
    if csv_path not in csv_mapping_cache:
        by_id = {}
        tuple_mapping = {}
        try:
            df = pd.read_csv(csv_path)
            for idx, row in df.iterrows():
                sample_id = str(row.get('id', idx))
                rxn = normalize_optional_value(row.get('CANO_RXN_SMILES'))
                enzyme = normalize_optional_value(row.get('UniprotID'))
                ec_number = normalize_optional_value(row.get('EC number'))
                if sample_id:
                    by_id[sample_id] = {
                        "row_order": idx,
                        "reaction": rxn,
                        "enzyme": enzyme,
                        "ec_number": ec_number,
                    }
                if enzyme and ec_number and rxn:
                    tuple_mapping[(enzyme, ec_number)] = {
                        "reaction": rxn,
                        "enzyme": enzyme,
                    }
        except Exception:
            by_id = {}
            tuple_mapping = {}
        csv_mapping_cache[csv_path] = {
            "by_id": by_id,
            "by_tuple": tuple_mapping,
        }
    return csv_mapping_cache[csv_path]


def get_sample_metadata_mapping(sample_ids, json_dir, reaction_cache=None, csv_path=None, csv_mapping_cache=None):
    """获取 sample_id -> {reaction, enzyme, row_order} 的映射，并在可选 cache 中复用结果。"""
    sample_ids = list(sample_ids)
    cache_key = (json_dir, csv_path or "")
    if reaction_cache is None:
        dir_cache = {}
    else:
        dir_cache = reaction_cache.setdefault(cache_key, {})
    csv_mapping_pack = get_sample_csv_metadata_mapping(csv_path, csv_mapping_cache=csv_mapping_cache)
    csv_mapping = csv_mapping_pack.get("by_id", {})
    csv_tuple_mapping = csv_mapping_pack.get("by_tuple", {})

    mapping = {}
    for sid in sample_ids:
        if sid not in dir_cache:
            json_path = os.path.join(json_dir, f'{sid}.json')
            csv_meta = csv_mapping.get(str(sid), {})
            metadata = {
                "sample_id": sid,
                "row_order": csv_meta.get("row_order"),
                "reaction": csv_meta.get("reaction"),
                "enzyme": csv_meta.get("enzyme"),
            }
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                if metadata["reaction"] is None:
                    metadata["reaction"] = normalize_optional_value(data.get('cano_rxn_smiles'))
                if metadata["enzyme"] is None:
                    metadata["enzyme"] = normalize_optional_value(data.get('uniprot_id', data.get('enzyme')))
                ec_number = normalize_optional_value(data.get('ec_number'))
                if metadata["reaction"] is None and csv_tuple_mapping and metadata["enzyme"] and ec_number:
                    tuple_meta = csv_tuple_mapping.get((metadata["enzyme"], ec_number), {})
                    metadata["reaction"] = tuple_meta.get("reaction")
            except Exception:
                pass

            if (metadata["reaction"] is None or metadata["enzyme"] is None) and csv_tuple_mapping:
                parts = str(sid).split('_')
                if len(parts) >= 2:
                    enzyme = normalize_optional_value(parts[0])
                    ec_number = normalize_optional_value('.'.join(parts[1:]))
                    if enzyme and ec_number:
                        tuple_meta = csv_tuple_mapping.get((enzyme, ec_number), {})
                        if metadata["reaction"] is None:
                            metadata["reaction"] = tuple_meta.get("reaction")
                        if metadata["enzyme"] is None:
                            metadata["enzyme"] = tuple_meta.get("enzyme")
            dir_cache[sid] = metadata
        mapping[sid] = dir_cache[sid]
    return mapping


def compute_dcg(rels, k=10):
    rels = np.asarray(rels, dtype=np.float32)
    if rels.size == 0:
        return 0.0
    k = min(k, rels.size)
    discounts = np.log2(np.arange(k, dtype=np.float32) + 2.0)
    return float(np.sum((2 ** rels[:k] - 1) / discounts))


def compute_ndcg(rels, k=10):
    rels = np.asarray(rels, dtype=np.float32)
    if rels.size == 0:
        return 0.0
    dcg = compute_dcg(rels, k=k)
    ideal = np.sort(rels)[::-1]
    idcg = compute_dcg(ideal, k=k)
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def compute_reaction_ranking_metrics(
    scores,
    labels,
    sample_ids,
    json_dir,
    reaction_cache=None,
    csv_path=None,
    csv_mapping_cache=None,
    k_values=(1, 3, 5, 10, 20, 50),
):
    """计算 reaction-level 排名指标，用于 train/valid/test 统一监控。"""
    results = {
        "sr": {k: 0.0 for k in k_values},
        "reaction_auc": 0.0,
        "reaction_ap": 0.0,
        "dcg10": 0.0,
        "ndcg10": 0.0,
        "mrr": 0.0,
        "mean_rank": 0.0,
        "reaction_count": 0,
        "positive_reaction_count": 0,
    }
    if not sample_ids:
        return results

    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    n = min(len(sample_ids), len(scores), len(labels))
    if n == 0:
        return results

    sample_ids = list(sample_ids)[:n]
    scores = scores[:n]
    labels = labels[:n]
    sample_metadata = get_sample_metadata_mapping(
        sample_ids,
        json_dir,
        reaction_cache=reaction_cache,
        csv_path=csv_path,
        csv_mapping_cache=csv_mapping_cache,
    )

    ordered_records = []
    for idx, sid in enumerate(sample_ids):
        meta = sample_metadata.get(sid, {})
        row_order = meta.get("row_order")
        ordered_records.append((
            row_order is None,
            row_order if row_order is not None else idx,
            idx,
        ))
    ordered_records.sort()

    dedup_records = []
    seen_keys = set()
    for _, _, idx in ordered_records:
        sid = sample_ids[idx]
        meta = sample_metadata.get(sid, {})
        reaction = meta.get("reaction")
        enzyme = meta.get("enzyme")
        if reaction is None or enzyme is None:
            continue
        dedup_key = (reaction, enzyme)
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        dedup_records.append((reaction, float(scores[idx]), int(labels[idx])))

    rxn_groups = defaultdict(list)
    for reaction, score, label in dedup_records:
        rxn_groups[reaction].append((score, label))

    if not rxn_groups:
        return results

    best_ranks = []
    per_reaction_auc = []
    per_reaction_ap = []
    dcg_values = []
    ndcg_values = []

    for pairs in rxn_groups.values():
        pairs.sort(key=lambda x: x[0], reverse=True)
        rxn_scores = np.array([p[0] for p in pairs], dtype=np.float32)
        rxn_labels = np.array([p[1] for p in pairs], dtype=np.int32)
        pos_ranks = np.where(rxn_labels == 1)[0] + 1

        if pos_ranks.size > 0:
            best_ranks.append(int(pos_ranks.min()))

        if 0 < rxn_labels.sum() < len(rxn_labels):
            try:
                per_reaction_auc.append(float(roc_auc_score(rxn_labels, rxn_scores)))
            except Exception:
                pass
            try:
                per_reaction_ap.append(float(average_precision_score(rxn_labels, rxn_scores)))
            except Exception:
                pass

        dcg_values.append(compute_dcg(rxn_labels, k=10))
        ndcg_values.append(compute_ndcg(rxn_labels, k=10))

    if best_ranks:
        best_ranks_np = np.asarray(best_ranks)
        results["sr"] = {k: float(np.mean(best_ranks_np <= k)) for k in k_values}
        results["mrr"] = float(np.mean(1.0 / best_ranks_np))
        results["mean_rank"] = float(np.mean(best_ranks_np))
        results["positive_reaction_count"] = int(best_ranks_np.size)

    if per_reaction_auc:
        results["reaction_auc"] = float(np.mean(per_reaction_auc))
    if per_reaction_ap:
        results["reaction_ap"] = float(np.mean(per_reaction_ap))
    if dcg_values:
        results["dcg10"] = float(np.mean(dcg_values))
    if ndcg_values:
        results["ndcg10"] = float(np.mean(ndcg_values))

    results["reaction_count"] = len(rxn_groups)
    return results


def compute_global_binary_metrics(scores, labels):
    """计算全局样本级二分类指标。"""
    scores = np.asarray(scores).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(np.int32)
    predictions = (scores >= 0.5).astype(np.int32)

    metrics = {
        "auc": 0.0,
        "ap": 0.0,
        "accuracy": float(np.mean(predictions == labels)) if labels.size > 0 else 0.0,
        "balanced_accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    try:
        metrics["auc"] = float(roc_auc_score(labels, scores))
    except Exception:
        pass

    try:
        metrics["ap"] = float(average_precision_score(labels, scores))
    except Exception:
        pass

    try:
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(labels, predictions))
    except Exception:
        pass

    metrics["precision"] = float(precision_score(labels, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(labels, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(labels, predictions, zero_division=0))
    return metrics


def main_classification():
    """二分类训练入口"""
    parser = argparse.ArgumentParser(description='AlphaEnzyme specificity classifier training')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config path')
    # 命令行参数 (会覆盖 YAML 配置)
    parser.add_argument('--json_dir', type=str, default=None,
                        help='Directory of per-pair JSON files')
    parser.add_argument('--embedding_dir', type=str, default=None,
                        help='Directory of Pairformer embeddings')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    # 模型超参数
    parser.add_argument('--pairformer_blocks', type=int, default=None, help='Number of Pairformer blocks')
    parser.add_argument('--n_heads', type=int, default=None, help='Number of attention heads')
    parser.add_argument('--hidden_size', type=int, default=None, help='Hidden size for the classifier')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length')
    parser.add_argument('--precision', type=str, default=None, help='Mixed precision (bf16, fp16, or no)')
    parser.add_argument('--use_amp', type=bool, default=None, help='Enable automatic mixed precision')
    parser.add_argument('--seed', type=int, default=None, help='Global random seed')
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help='Enable deterministic training')
    parser.add_argument('--non_deterministic', dest='deterministic', action='store_false',
                        help='Disable deterministic training')
    parser.add_argument('--compressed', action='store_true', default=False,
                        help='Use the compressed token-level Pairformer representation')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a checkpoint')
    parser.set_defaults(deterministic=None)
    args = parser.parse_args()

    # 如果提供了配置文件，从 YAML 加载配置
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = EasyDict(yaml.safe_load(f))

    # 获取 log_dir (需要在设置 logger 之前)
    log_dir = args.log_dir or (
        config.training.log_dir if config and 'training' in config
        else os.path.join(OUTPUT_ROOT, 'logs', 'training')
    )

    # 设置日志
    logger = setup_logger("train_simple", log_dir)

    # 如果有 config，记录加载信息
    if config:
        logger.info(f"Loading config from: {args.config}")
        logger.info(f"Config loaded: {config}")

    # 辅助函数：命令行参数优先，否则使用 YAML 配置
    def get_param(name, default=None):
        val = getattr(args, name)
        if val is not None:
            return val
        if config and name in config:
            return config[name]
        if config and 'training' in config and name in config.training:
            return config.training[name]
        if config and 'model' in config and name in config.model:
            return config.model[name]
        return default

    # 获取所有参数
    json_dir = args.json_dir or (config.data.train_json if config else None)
    valid_json_dir = config.data.valid_json if config else None
    test_json_dir = config.data.get('test_json') if config and 'data' in config else None
    embedding_dir = args.embedding_dir or (config.data.train_embedding_dir if config else None)
    valid_embedding_dir = config.data.valid_embedding_dir if config else None
    test_embedding_dir = config.data.get('test_embedding_dir') if config and 'data' in config else None
    output_dir = args.output_dir or (
        config.training.output_dir if config
        else os.path.join(OUTPUT_ROOT, 'outputs', 'checkpoint', 'enzyme_classification')
    )
    batch_size = args.batch_size or (config.training.batch_size if config else 4)
    lr = args.lr or (config.training.lr if config else 1e-4)
    epochs = args.epochs or (config.training.epochs if config else 10)
    hidden_size = args.hidden_size or (config.model.get('hidden_size', 512) if config else 512)
    dropout = args.dropout if args.dropout is not None else (config.model.get('dropout', 0.0) if config else 0.0)
    pairformer_blocks = args.pairformer_blocks or (config.model.get('pairformer_blocks', 2) if config else 2)
    n_heads = args.n_heads or (config.model.get('n_heads_single', config.model.get('n_heads', 16)) if config else 16)
    n_heads_pair = config.model.get('n_heads_pair', 4) if config else 4
    pooling_mode = config.model.get('pooling_mode', 'ligand_only') if config else 'ligand_only'
    global_pool_scale_init = float(config.model.get('global_pool_scale_init', 0.5) if config else 0.5)
    main_input_mode = str(config.model.get('main_input_mode', 'sz') if config else 'sz').lower()
    # 处理 blocks_per_ckpt
    # None 或 'None' 表示不使用 gradient checkpointing
    # 整数值表示每 N 个 block 做一次 checkpoint
    blocks_per_ckpt_raw = config.model.get('blocks_per_ckpt', None) if config and 'model' in config else None
    if blocks_per_ckpt_raw is None or blocks_per_ckpt_raw == 'None':
        blocks_per_ckpt = None  # 不使用 gradient checkpointing
    else:
        blocks_per_ckpt = int(blocks_per_ckpt_raw)
    max_seq_length = args.max_seq_length or (config.data.get('max_seq_length') if config and 'data' in config else None)
    use_amp = args.use_amp if args.use_amp is not None else (config.training.get('use_amp', True) if config else True)
    precision = args.precision or (config.training.get('precision', 'bf16') if config else 'bf16')
    seed = args.seed if args.seed is not None else int(config.training.get('seed', 42) if config else 42)
    deterministic = (
        args.deterministic
        if args.deterministic is not None
        else bool(config.training.get('deterministic', True) if config else True)
    )
    use_compressed = args.compressed or (config.model.get('use_compressed', False) if config else False)
    resume_from = args.resume_from or (
        config.get('training', {}).get('resume_from') if config else None
    )

    # Eval 配置
    eval_every_epoch = config.get('evaluation', {}).get('eval_every_epoch', False) if config else False
    test_every_n_epochs_raw = config.get('evaluation', {}).get('test_every_n_epochs', 0) if config else 0
    if test_every_n_epochs_raw in [None, 'None', 0, '0']:
        test_every_n_epochs = 0
    else:
        test_every_n_epochs = int(test_every_n_epochs_raw)

    # EMA 配置
    use_ema = config.get('ema', {}).get('use_ema', False) if config else False
    ema_decay = config.get('ema', {}).get('decay', 0.999) if config else 0.999
    ema_update_interval = config.get('ema', {}).get('update_interval', 1) if config else 1

    # 日志配置
    use_swanlab = bool(config.get('logging', {}).get('use_swanlab', False)) if config else False
    if use_swanlab and swanlab is None:
        use_swanlab = False
    swanlab_project = os.environ.get(
        'SWANLAB_PROJECT',
        config.get('logging', {}).get('swanlab_project', 'Alphaenzyme') if config else 'Alphaenzyme'
    )
    swanlab_workspace = os.environ.get(
        'SWANLAB_WORKSPACE',
        config.get('logging', {}).get('swanlab_workspace') if config else None
    )

    # 显存管理配置
    empty_cache_interval = config.get('training', {}).get('empty_cache_interval', 0) if config else 0
    if empty_cache_interval is None or empty_cache_interval == 'None':
        empty_cache_interval = 0  # 0 表示不进行定期清理

    # 学习率调度器配置
    use_scheduler = config.get('training', {}).get('use_scheduler', False) if config else False
    warmup_ratio = config.get('training', {}).get('warmup_ratio', 0.1) if config else 0.1
    warmup_steps = config.get('training', {}).get('warmup_steps', None) if config else None
    scheduler_type = config.get('training', {}).get('scheduler_type', 'cosine') if config else 'cosine'
    dropout_warmup_ratio = config.get('training', {}).get('dropout_warmup_ratio', 0.0) if config else 0.0
    label_smoothing = float(config.get('training', {}).get('label_smoothing', 0.0) if config else 0.0)
    early_stopping_patience_raw = config.get('training', {}).get('early_stopping_patience', None) if config else None
    early_stopping_metric = str(config.get('training', {}).get('early_stopping_metric', 'loss') if config else 'loss').lower()
    if early_stopping_patience_raw in [None, 'None', 0, '0']:
        early_stopping_patience = None
    else:
        early_stopping_patience = int(early_stopping_patience_raw)

    # 优化器配置
    optimizer_type = config.get('training', {}).get('optimizer', 'adam') if config else 'adam'
    weight_decay = config.get('training', {}).get('weight_decay', 0.01) if config else 0.01
    beta1 = config.get('training', {}).get('beta1', 0.9) if config else 0.9
    beta2 = config.get('training', {}).get('beta2', 0.999) if config else 0.999

    # 梯度管理配置
    max_grad_norm = config.get('training', {}).get('max_grad_norm', 1.0) if config else 1.0

    # Checkpoint 保存配置
    save_every_n_epochs = config.get('training', {}).get('save_every_n_epochs', 5) if config else 5

    # DRFP 配置
    use_drfp = config.get('model', {}).get('use_drfp', True) if config else True
    drfp_dim = config.get('model', {}).get('drfp_dim', 2048) if config else 2048
    drfp_injection = config.get('model', {}).get('drfp_injection', 'z_pair_pre_pairformer_global') if config else 'z_pair_pre_pairformer_global'
    drfp_pair_scale_init = config.get('model', {}).get('drfp_pair_scale_init', 1.0) if config else 1.0

    # Pairwise / Hybrid loss 配置
    loss_type = config.get('training', {}).get('loss_type', 'bce') if config else 'bce'
    bce_weight = float(config.get('training', {}).get('bce_weight', 1.0) if config else 1.0)
    pairwise_weight = float(config.get('training', {}).get('pairwise_weight', 1.0) if config else 1.0)
    listwise_weight = float(config.get('training', {}).get('listwise_weight', 1.0) if config else 1.0)
    listwise_temperature = float(config.get('training', {}).get('listwise_temperature', 1.0) if config else 1.0)
    retrieval_prior_mode = str(
        config.get('training', {}).get('retrieval_prior_mode', 'none') if config else 'none'
    ).lower()
    retrieval_prior_scale = float(config.get('training', {}).get('retrieval_prior_scale', 0.0) if config else 0.0)
    retrieval_prior_threshold = float(
        config.get('training', {}).get('retrieval_prior_threshold', 0.0) if config else 0.0
    )
    pair_batching_strategy = str(
        config.get('training', {}).get('pair_batching_strategy', 'balanced_chunks') if config else 'balanced_chunks'
    ).lower()
    sampled_batches_per_reaction = int(
        config.get('training', {}).get('sampled_batches_per_reaction', 1) if config else 1
    )
    min_positive_fraction = float(
        config.get('training', {}).get('min_positive_fraction', 0.15) if config else 0.15
    )
    max_positive_fraction = float(
        config.get('training', {}).get('max_positive_fraction', 0.70) if config else 0.70
    )
    num_workers = int(config.get('training', {}).get('num_workers', 4) if config else 4)
    ranking_loss_types = ('pairwise', 'hybrid', 'listwise', 'hybrid_listwise')
    uses_grouped_reaction_batches = loss_type in ranking_loss_types
    logger.info(f"Loss type: {loss_type}")

    # ==================== 初始化 Accelerator ====================
    set_global_reproducibility(seed=seed, deterministic=deterministic, logger=logger)

    # 使用 Accelerate 进行分布式训练
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=False)]
    accelerator = Accelerator(
        kwargs_handlers=kwargs_handlers,
        mixed_precision=precision,
        log_with=None,  # 使用 swanlab 记录，不使用 wandb
        step_scheduler_with_optimizer=False
    )

    # 获取分布式信息
    device = accelerator.device
    is_distributed = accelerator.num_processes > 1
    logger.info(f"Using device: {device}, num_processes: {accelerator.num_processes}, is_distributed: {is_distributed}")

    # DRFP 数据路径（优先从 data 配置读取，其次从 model 配置读取，最后使用默认路径）
    drfp_path = (config.get('data', {}).get('drfp_path') or
                 config.get('model', {}).get('drfp_path')) if config else None
    if not drfp_path:
        drfp_path = os.path.join(DATA_ROOT, 'inputs', 'specificity', 'features', 'drfp', 'reaction_fingerprints.pkl')
    # CSV 路径：优先从 data 配置读取，否则按约定路径推断
    train_csv_path = (config.get('data', {}).get('csv_path') if config else None) or \
                     os.path.join(os.path.dirname(json_dir), 'csv', 'train.csv')
    valid_csv_path = (config.get('data', {}).get('valid_csv_path') if config else None) or \
                     (os.path.join(os.path.dirname(json_dir), 'csv', 'valid.csv') if valid_json_dir else None)
    test_csv_path = (config.get('data', {}).get('test_csv_path') if config else None) or \
                    (os.path.join(test_json_dir, 'csv', 'valid.csv') if test_json_dir else None)

    if test_json_dir is None:
        default_test_json_dir = os.path.join(DATA_ROOT, 'inputs', 'specificity', 'features', 'json', 'enzyme_405')
        if os.path.isdir(default_test_json_dir):
            test_json_dir = default_test_json_dir
    if test_embedding_dir is None:
        default_test_embedding_dir = os.path.join(DATA_ROOT, 'inputs', 'specificity', 'features', 'pairformer', 'test')
        if os.path.isdir(default_test_embedding_dir):
            test_embedding_dir = default_test_embedding_dir
    if test_csv_path is None and test_json_dir:
        default_test_csv_path = os.path.join(test_json_dir, 'csv', 'valid.csv')
        if os.path.exists(default_test_csv_path):
            test_csv_path = default_test_csv_path

    external_missing_policy = str(
        config.get('data', {}).get('external_missing_policy', 'error') if config else 'error'
    ).lower()

    def resolve_external_bank_specs(split_name):
        if not config:
            return []
        data_cfg = config.get('data', {})
        split_key = f'{split_name}_external_feature_banks'
        if split_key in data_cfg:
            return normalize_bank_specs(data_cfg.get(split_key))
        legacy_path = data_cfg.get(f'{split_name}_external_feature_bank_path')
        if legacy_path:
            return normalize_bank_specs(legacy_path)
        common_key = data_cfg.get('external_feature_banks')
        if common_key:
            return normalize_bank_specs(common_key)
        return []

    train_external_bank_specs = resolve_external_bank_specs('train')
    valid_external_bank_specs = resolve_external_bank_specs('valid')
    test_external_bank_specs = resolve_external_bank_specs('test')

    context_missing_policy = str(
        config.get('data', {}).get('context_missing_policy', 'error') if config else 'error'
    ).lower()

    def resolve_split_data_value(split_name, key_suffix, default=None):
        if not config:
            return default
        data_cfg = config.get('data', {})
        split_key = f"{split_name}_{key_suffix}"
        if split_key in data_cfg:
            return data_cfg.get(split_key)
        if key_suffix in data_cfg:
            return data_cfg.get(key_suffix)
        return default

    train_context_cache_path = resolve_split_data_value('train', 'context_cache_path')
    valid_context_cache_path = resolve_split_data_value('valid', 'context_cache_path')
    test_context_cache_path = resolve_split_data_value('test', 'context_cache_path')
    train_context_bank_path = resolve_split_data_value('train', 'context_bank_path')
    valid_context_bank_path = resolve_split_data_value('valid', 'context_bank_path', default=train_context_bank_path)
    test_context_bank_path = resolve_split_data_value('test', 'context_bank_path', default=train_context_bank_path)

    def build_loader_kwargs(shuffle=False, collate_fn=None):
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed + accelerator.process_index)

        def seed_worker(worker_id):
            worker_seed = (seed + accelerator.process_index * 1000 + worker_id) % (2**32)
            random.seed(worker_seed)
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        kwargs = {
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": True,
            "generator": loader_generator,
            "worker_init_fn": seed_worker,
        }
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        if num_workers > 0:
            kwargs["prefetch_factor"] = 2
        return kwargs

    # 创建数据集
    # compressed 模式显存压力小，不限制序列长度
    if use_compressed and max_seq_length is not None:
        logger.info(f"Compressed mode: disabling max_seq_length filtering (use full sequences)")
        max_seq_length = None

    if max_seq_length is not None:
        logger.info(f"Filtering sequences with length > {max_seq_length}")

    # 根据 use_drfp 决定是否加载 DRFP 数据
    if uses_grouped_reaction_batches:
        from alphaenzyme.data.pairwise_dataset import PairwiseReactionDataset
        logger.info(
            f"Using PairwiseReactionDataset (batch_size={batch_size}, "
            f"batching_strategy={pair_batching_strategy})"
        )
        dataset = PairwiseReactionDataset(
            json_dir,
            embedding_dir,
            csv_path=train_csv_path,
            drfp_path=drfp_path if use_drfp else None,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            batching_strategy=pair_batching_strategy,
            sampled_batches_per_reaction=sampled_batches_per_reaction,
            min_positive_fraction=min_positive_fraction,
            max_positive_fraction=max_positive_fraction,
            random_seed=seed,
            external_feature_banks=train_external_bank_specs,
            external_missing_policy=external_missing_policy,
            context_cache_path=train_context_cache_path,
            context_bank_path=train_context_bank_path,
            context_missing_policy=context_missing_policy,
        )
    elif use_drfp:
        logger.info(f"DRFP enabled: loading DRFP data from {drfp_path}")
        dataset = BinaryClassificationDataset(
            json_dir,
            embedding_dir,
            csv_path=train_csv_path,
            drfp_path=drfp_path,
            max_seq_length=max_seq_length,
            external_feature_banks=train_external_bank_specs,
            external_missing_policy=external_missing_policy,
            context_cache_path=train_context_cache_path,
            context_bank_path=train_context_bank_path,
            context_missing_policy=context_missing_policy,
        )
    else:
        logger.info(f"DRFP disabled: not loading DRFP data")
        dataset = BinaryClassificationDataset(
            json_dir,
            embedding_dir,
            csv_path=None,
            drfp_path=None,
            max_seq_length=max_seq_length,
            external_feature_banks=train_external_bank_specs,
            external_missing_policy=external_missing_policy,
            context_cache_path=train_context_cache_path,
            context_bank_path=train_context_bank_path,
            context_missing_policy=context_missing_policy,
        )
    logger.info(f"Loaded {len(dataset)} training samples")

    # 创建验证集 (如果提供了 valid_embedding_dir)
    valid_dataset = None
    valid_dataloader = None
    if valid_json_dir and valid_embedding_dir:
        logger.info(f"Creating validation dataset...")
        if uses_grouped_reaction_batches:
            from alphaenzyme.data.pairwise_dataset import PairwiseReactionDataset
            valid_dataset = PairwiseReactionDataset(
                valid_json_dir,
                valid_embedding_dir,
                csv_path=valid_csv_path,
                drfp_path=drfp_path if use_drfp else None,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                batching_strategy='balanced_chunks',
                random_seed=seed + 1,
                external_feature_banks=valid_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=valid_context_cache_path,
                context_bank_path=valid_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            valid_dataloader = DataLoader(valid_dataset, batch_size=1, **build_loader_kwargs(shuffle=False))
        elif use_drfp:
            valid_dataset = BinaryClassificationDataset(
                valid_json_dir,
                valid_embedding_dir,
                csv_path=valid_csv_path,
                drfp_path=drfp_path,
                max_seq_length=max_seq_length,
                external_feature_banks=valid_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=valid_context_cache_path,
                context_bank_path=valid_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, **build_loader_kwargs(shuffle=False, collate_fn=binary_collate_fn))
        else:
            valid_dataset = BinaryClassificationDataset(
                valid_json_dir,
                valid_embedding_dir,
                csv_path=None,
                drfp_path=None,
                max_seq_length=max_seq_length,
                external_feature_banks=valid_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=valid_context_cache_path,
                context_bank_path=valid_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, **build_loader_kwargs(shuffle=False, collate_fn=binary_collate_fn))
        logger.info(f"Loaded {len(valid_dataset)} validation samples")
    else:
        logger.info("No validation dataset configured (missing valid_json or valid_embedding_dir)")

    test_dataset = None
    test_dataloader = None
    if test_every_n_epochs > 0 and test_json_dir and test_embedding_dir:
        logger.info(f"Creating test monitor dataset (every {test_every_n_epochs} epoch)...")
        if uses_grouped_reaction_batches:
            from alphaenzyme.data.pairwise_dataset import PairwiseReactionDataset
            test_dataset = PairwiseReactionDataset(
                test_json_dir,
                test_embedding_dir,
                csv_path=test_csv_path,
                drfp_path=drfp_path if use_drfp else None,
                max_seq_length=max_seq_length,
                batch_size=batch_size,
                batching_strategy='balanced_chunks',
                random_seed=seed + 2,
                external_feature_banks=test_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=test_context_cache_path,
                context_bank_path=test_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=1, **build_loader_kwargs(shuffle=False))
        elif use_drfp:
            test_dataset = BinaryClassificationDataset(
                test_json_dir,
                test_embedding_dir,
                csv_path=test_csv_path,
                drfp_path=drfp_path,
                max_seq_length=max_seq_length,
                external_feature_banks=test_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=test_context_cache_path,
                context_bank_path=test_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **build_loader_kwargs(shuffle=False, collate_fn=binary_collate_fn))
        else:
            test_dataset = BinaryClassificationDataset(
                test_json_dir,
                test_embedding_dir,
                csv_path=None,
                drfp_path=None,
                max_seq_length=max_seq_length,
                external_feature_banks=test_external_bank_specs,
                external_missing_policy=external_missing_policy,
                context_cache_path=test_context_cache_path,
                context_bank_path=test_context_bank_path,
                context_missing_policy=context_missing_policy,
            )
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, **build_loader_kwargs(shuffle=False, collate_fn=binary_collate_fn))
        logger.info(f"Loaded {len(test_dataset)} test monitor samples")
    elif test_every_n_epochs > 0:
        logger.info("Test monitor requested but missing test_json or test_embedding_dir, disabling test logging")

    train_external_feature_dim = int(getattr(dataset, "external_feature_dim", 0))
    valid_external_feature_dim = int(getattr(valid_dataset, "external_feature_dim", 0)) if valid_dataset is not None else 0
    test_external_feature_dim = int(getattr(test_dataset, "external_feature_dim", 0)) if test_dataset is not None else 0
    external_feature_dim = max(train_external_feature_dim, valid_external_feature_dim, test_external_feature_dim)
    if valid_dataset is not None and valid_external_feature_dim not in (0, external_feature_dim):
        raise ValueError(f"Validation external_feature_dim mismatch: train/test={external_feature_dim}, valid={valid_external_feature_dim}")
    if test_dataset is not None and test_external_feature_dim not in (0, external_feature_dim):
        raise ValueError(f"Test external_feature_dim mismatch: train/valid={external_feature_dim}, test={test_external_feature_dim}")
    use_extra_features = external_feature_dim > 0
    if use_extra_features and not use_compressed:
        logger.info("External feature fusion is only enabled for compressed models; disabling extra features for full model path.")
        use_extra_features = False
    extra_hidden_size = int(config.get('model', {}).get('extra_hidden_size', hidden_size) if config else hidden_size)
    extra_fusion_mode = str(config.get('model', {}).get('extra_fusion_mode', 'delta') if config else 'delta').lower()
    extra_logit_scale_init = float(config.get('model', {}).get('extra_logit_scale_init', 0.5) if config else 0.5)
    extra_token_injection = bool(config.get('model', {}).get('extra_token_injection', False) if config else False)
    extra_branch_scale_init = float(config.get('model', {}).get('extra_branch_scale_init', 1.0) if config else 1.0)
    train_external_feature_meta = list(getattr(dataset, "external_feature_meta", []) or [])
    extra_feature_branch_dims = [int(meta.get("dim", 0)) for meta in train_external_feature_meta if int(meta.get("dim", 0)) > 0]
    extra_feature_branch_names = [str(meta.get("name")) for meta in train_external_feature_meta if int(meta.get("dim", 0)) > 0]
    if use_extra_features and not extra_feature_branch_dims:
        extra_feature_branch_dims = [external_feature_dim]
        extra_feature_branch_names = ["external_feature"]
    logger.info(
        f"External feature setup: use_extra_features={use_extra_features}, dim={external_feature_dim}, "
        f"fusion={extra_fusion_mode}, extra_hidden_size={extra_hidden_size}, scale_init={extra_logit_scale_init}"
    )
    if use_extra_features:
        logger.info(
            f"  extra_feature_branch_names={extra_feature_branch_names}, "
            f"extra_feature_branch_dims={extra_feature_branch_dims}, "
            f"extra_token_injection={extra_token_injection}, "
            f"extra_branch_scale_init={extra_branch_scale_init}"
        )

    train_context_dim = int(getattr(dataset, "context_dim", 0))
    valid_context_dim = int(getattr(valid_dataset, "context_dim", 0)) if valid_dataset is not None else 0
    test_context_dim = int(getattr(test_dataset, "context_dim", 0)) if test_dataset is not None else 0
    context_feature_dim = max(train_context_dim, valid_context_dim, test_context_dim)
    train_context_len = int(getattr(dataset, "context_len", 0))
    valid_context_len = int(getattr(valid_dataset, "context_len", 0)) if valid_dataset is not None else 0
    test_context_len = int(getattr(test_dataset, "context_len", 0)) if test_dataset is not None else 0
    context_len = max(train_context_len, valid_context_len, test_context_len)
    train_overall_corr_dim = 2 if train_context_dim > 0 else 0
    valid_overall_corr_dim = 2 if valid_context_dim > 0 else 0
    test_overall_corr_dim = 2 if test_context_dim > 0 else 0
    overall_corr_dim = max(train_overall_corr_dim, valid_overall_corr_dim, test_overall_corr_dim)
    use_context_features = context_feature_dim > 0 and context_len > 0
    context_scale_init = float(config.get('model', {}).get('context_scale_init', 1.0) if config else 1.0)
    context_score_scale_init = float(config.get('model', {}).get('context_score_scale_init', 0.25) if config else 0.25)
    overall_corr_scale_init = float(config.get('model', {}).get('overall_corr_scale_init', 0.5) if config else 0.5)
    use_context_score_bias = bool(config.get('model', {}).get('use_context_score_bias', True) if config else True)
    use_overall_corr = bool(config.get('model', {}).get('use_overall_corr', True) if config else True)
    logger.info(
        f"Context feature setup: use_context_features={use_context_features}, len={context_len}, "
        f"dim={context_feature_dim}, overall_corr_dim={overall_corr_dim}, "
        f"context_scale_init={context_scale_init}, context_score_scale_init={context_score_scale_init}, "
        f"overall_corr_scale_init={overall_corr_scale_init}, "
        f"use_context_score_bias={use_context_score_bias}, use_overall_corr={use_overall_corr}"
    )

    # DataLoader: Pairwise 模式下 batch 已由 Dataset 构造好，batch_size=1
    if uses_grouped_reaction_batches:
        dataloader = DataLoader(dataset, batch_size=1, **build_loader_kwargs(shuffle=False))
        # 统计 label 分布：从 batch_metadata 获取
        logger.info("Computing label distribution...")
        total_pos = sum(m['n_pos'] for m in dataset.batch_metadata)
        total_neg = sum(m['n_neg'] for m in dataset.batch_metadata)
        label_counts = {0: total_neg, 1: total_pos}
        logger.info(f"Label distribution: {label_counts}")
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, **build_loader_kwargs(shuffle=True, collate_fn=binary_collate_fn))
        # 统计label分布 (从JSON文件读取，不加载embedding)
        logger.info("Computing label distribution...")
        label_counts = {0: 0, 1: 0}
        for sample in dataset.samples:
            with open(sample['json_path'], 'r') as f:
                data = json.load(f)
            label = data.get('label', 0)
            label_counts[label] = label_counts.get(label, 0) + 1
        logger.info(f"Label distribution: {label_counts}")

    if len(dataloader) == 0:
        raise RuntimeError(
            f"No training batches were constructed for loss_type={loss_type}. "
            f"Please check reaction mapping / labels / embedding coverage for {json_dir}"
        )

    # ==================== SwanLab 初始化 ====================
    # 读取任务类型 (默认 classification)
    task_type = config.get('task_type', 'classification') if config else 'classification'
    use_msa = config.data.get('use_msa', False) if config and 'data' in config else False

    # 从数据路径提取训练数据名称（使用 train_embedding_dir 的后两个路径）
    if embedding_dir:
        # 例如：data/specificity_50w/features/pairformer_compressed/train
        # 提取：pairformer_compressed/train
        path_parts = embedding_dir.rstrip('/').split('/')
        if len(path_parts) >= 2:
            train_data_name = f"{path_parts[-2]}/{path_parts[-1]}"
        else:
            train_data_name = os.path.basename(embedding_dir)
    else:
        train_data_name = "unknown"

    # 生成简洁的 experiment_name（包含关键信息：任务类型、是否使用MSA、是否使用DRFP、学习率、batch size、dropout）
    msa_suffix = "msa" if use_msa else "nomsa"
    drfp_suffix = "drfp" if use_drfp else "nodrfp"
    model_mode_suffix = "cmp" if use_compressed else "full"
    arch_suffix = f"h{hidden_size}"
    if use_compressed:
        arch_suffix += f"_l{config.model.get('n_transformer_layers', 4) if config else 4}"
    else:
        arch_suffix += f"_pb{pairformer_blocks}"
    loss_suffix = str(loss_type).lower()
    if use_extra_features and train_external_bank_specs:
        bank_name_tokens = []
        for spec in train_external_bank_specs[:2]:
            bank_name = str(spec.get("name") or os.path.splitext(os.path.basename(spec["path"]))[0]).lower()
            bank_name_tokens.append(bank_name.replace("enzymecage_true_esmc_", "").replace("train_", ""))
        bank_suffix = "-".join(bank_name_tokens) if bank_name_tokens else "ext"
        extra_suffix = f"{bank_suffix}_{external_feature_dim}d_{extra_fusion_mode}"
    else:
        extra_suffix = "noext"
    if use_context_features:
        context_suffix = f"ctx{context_len}x{context_feature_dim}"
        if not use_context_score_bias:
            context_suffix += "_noscore"
        if not use_overall_corr:
            context_suffix += "_nocorr"
    else:
        context_suffix = "noctx"
    experiment_name = (
        f"{task_type}_{msa_suffix}_{drfp_suffix}_{model_mode_suffix}_{loss_suffix}_"
        f"{arch_suffix}_{extra_suffix}_{context_suffix}_lr{lr}_bs{batch_size}_do{dropout}"
    )

    # 将 output_dir 的末尾目录名替换为 experiment_name，使 checkpoint 文件夹与 swanlab 实验名一致
    output_dir = os.path.join(os.path.dirname(output_dir.rstrip('/')), experiment_name)

    # 准备超参数配置（用于 SwanLab config）
    hparams_config = {
        "task_type": task_type,
        "train_data": train_data_name,
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "max_seq_length": max_seq_length if max_seq_length else None,
        "hidden_size": hidden_size,
        "pairformer_blocks": pairformer_blocks,
        "n_heads": n_heads,
        "n_heads_pair": n_heads_pair,
        "dropout": dropout,
        "blocks_per_ckpt": blocks_per_ckpt,
        "use_amp": use_amp,
        "precision": precision,
        "seed": seed,
        "deterministic": deterministic,
        "use_msa": use_msa,
        "use_ema": use_ema,
        "ema_decay": ema_decay if use_ema else None,
        "ema_update_interval": ema_update_interval if use_ema else None,
        "eval_every_epoch": eval_every_epoch,
        "valid_samples": len(valid_dataset) if valid_dataset else 0,
        "test_samples": len(test_dataset) if test_dataset else 0,
        "test_every_n_epochs": test_every_n_epochs,
        "include_product": False,  # 当前固定 false
        "condition_ph": 7.0,  # 当前固定 7.0
        "condition_temp": 25.0,  # 当前固定 25°C
        "train_samples": len(dataset),
        "label_0_count": label_counts.get(0, 0),
        "label_1_count": label_counts.get(1, 0),
        "optimizer": optimizer_type,
        "weight_decay": weight_decay,
        "beta1": beta1,
        "beta2": beta2,
        "use_scheduler": use_scheduler,
        "warmup_ratio": warmup_ratio if use_scheduler else None,
        "warmup_steps": warmup_steps if use_scheduler else None,
        "scheduler_type": scheduler_type if use_scheduler else None,
        "max_grad_norm": max_grad_norm,
        "dropout_warmup_ratio": dropout_warmup_ratio,
        "label_smoothing": label_smoothing if loss_type in ('bce', 'hybrid', 'hybrid_listwise') else None,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_metric": early_stopping_metric if early_stopping_patience is not None else None,
        "loss_type": loss_type,
        "bce_weight": bce_weight if loss_type in ('hybrid', 'hybrid_listwise') else None,
        "pairwise_weight": pairwise_weight if loss_type == 'hybrid' else (1.0 if loss_type == 'pairwise' else None),
        "listwise_weight": listwise_weight if loss_type == 'hybrid_listwise' else (1.0 if loss_type == 'listwise' else None),
        "listwise_temperature": listwise_temperature if loss_type in ('listwise', 'hybrid_listwise') else None,
        "retrieval_prior_mode": retrieval_prior_mode,
        "retrieval_prior_scale": retrieval_prior_scale,
        "retrieval_prior_threshold": retrieval_prior_threshold,
        "pair_batching_strategy": pair_batching_strategy if uses_grouped_reaction_batches else None,
        "sampled_batches_per_reaction": sampled_batches_per_reaction if uses_grouped_reaction_batches else None,
        "min_positive_fraction": min_positive_fraction if uses_grouped_reaction_batches else None,
        "max_positive_fraction": max_positive_fraction if uses_grouped_reaction_batches else None,
        "use_drfp": use_drfp,
        "drfp_dim": drfp_dim if use_drfp else None,
        "drfp_injection": drfp_injection if use_drfp else None,
        "drfp_pair_scale_init": drfp_pair_scale_init if use_drfp else None,
        "pooling_mode": pooling_mode,
        "global_pool_scale_init": global_pool_scale_init if pooling_mode == "ligand_plus_global" else None,
        "main_input_mode": main_input_mode,
        "use_extra_features": use_extra_features,
        "extra_feature_dim": external_feature_dim if use_extra_features else None,
        "extra_hidden_size": extra_hidden_size if use_extra_features else None,
        "extra_fusion_mode": extra_fusion_mode if use_extra_features else None,
        "extra_logit_scale_init": extra_logit_scale_init if use_extra_features else None,
        "extra_feature_branch_names": extra_feature_branch_names if use_extra_features else None,
        "extra_feature_branch_dims": extra_feature_branch_dims if use_extra_features else None,
        "extra_token_injection": extra_token_injection if use_extra_features else None,
        "extra_branch_scale_init": extra_branch_scale_init if use_extra_features else None,
        "use_context_features": use_context_features,
        "context_feature_dim": context_feature_dim if use_context_features else None,
        "context_len": context_len if use_context_features else None,
        "context_scale_init": context_scale_init if use_context_features else None,
        "context_score_scale_init": context_score_scale_init if use_context_features else None,
        "overall_corr_dim": overall_corr_dim if use_context_features else None,
        "overall_corr_scale_init": overall_corr_scale_init if use_context_features else None,
        "use_context_score_bias": use_context_score_bias if use_context_features else None,
        "use_overall_corr": use_overall_corr if use_context_features else None,
        "train_external_feature_banks": train_external_bank_specs if train_external_bank_specs else None,
        "valid_external_feature_banks": valid_external_bank_specs if valid_external_bank_specs else None,
        "test_external_feature_banks": test_external_bank_specs if test_external_bank_specs else None,
        "external_missing_policy": external_missing_policy if use_extra_features else None,
        "train_context_cache_path": train_context_cache_path if use_context_features else None,
        "valid_context_cache_path": valid_context_cache_path if use_context_features else None,
        "test_context_cache_path": test_context_cache_path if use_context_features else None,
        "context_bank_path": train_context_bank_path if use_context_features else None,
        "context_missing_policy": context_missing_policy if use_context_features else None,
        "num_workers": num_workers,
    }

    # 初始化 SwanLab（只在主进程初始化，将超参数作为 config 传入）
    if use_swanlab and accelerator.is_main_process:
        swanlab_init_kwargs = {
            "project": swanlab_project,
            "experiment_name": experiment_name,
            "config": hparams_config,
        }
        if swanlab_workspace:
            swanlab_init_kwargs["workspace"] = swanlab_workspace
        swanlab.init(**swanlab_init_kwargs)

    # 创建模型
    model_configs = EasyDict({
        "c_s": 384,
        "c_z": 128,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "pairformer_blocks": pairformer_blocks,
        "n_heads": n_heads,
        "n_heads_pair": n_heads_pair,
        "blocks_per_ckpt": blocks_per_ckpt,
        "use_drfp": use_drfp,
        "drfp_dim": drfp_dim,
        "drfp_injection": drfp_injection,
        "drfp_pair_scale_init": drfp_pair_scale_init,
        "pooling_mode": pooling_mode,
        "global_pool_scale_init": global_pool_scale_init,
        "main_input_mode": main_input_mode,
        "use_extra_features": use_extra_features,
        "extra_feature_dim": external_feature_dim,
        "extra_hidden_size": extra_hidden_size,
        "extra_fusion_mode": extra_fusion_mode,
        "extra_logit_scale_init": extra_logit_scale_init,
        "extra_feature_branch_dims": extra_feature_branch_dims,
        "extra_feature_branch_names": extra_feature_branch_names,
        "extra_token_injection": extra_token_injection,
        "extra_branch_scale_init": extra_branch_scale_init,
        "context_dim": context_feature_dim,
        "context_len": context_len,
        "context_scale_init": context_scale_init,
        "context_score_scale_init": context_score_scale_init,
        "overall_corr_dim": overall_corr_dim,
        "overall_corr_scale_init": overall_corr_scale_init,
        "use_context_score_bias": use_context_score_bias,
        "use_overall_corr": use_overall_corr,
    })
    logger.info(f"Model config: c_s=384, c_z=128, hidden_size={hidden_size}, dropout={dropout}")
    logger.info(f"  pairformer_blocks={pairformer_blocks}, n_heads={n_heads}, n_heads_pair={n_heads_pair}")
    logger.info(f"  blocks_per_ckpt={blocks_per_ckpt} (Gradient Checkpointing)")
    logger.info(
        f"  use_drfp={use_drfp}, drfp_dim={drfp_dim}, "
        f"drfp_injection={drfp_injection}, drfp_pair_scale_init={drfp_pair_scale_init}"
    )
    logger.info(
        f"  pooling_mode={pooling_mode}, global_pool_scale_init={global_pool_scale_init}, "
        f"main_input_mode={main_input_mode}"
    )
    logger.info(
        f"  use_extra_features={use_extra_features}, extra_feature_dim={external_feature_dim}, "
        f"extra_hidden_size={extra_hidden_size}, extra_fusion_mode={extra_fusion_mode}, "
        f"extra_logit_scale_init={extra_logit_scale_init}"
    )
    if use_extra_features:
        logger.info(
            f"  extra_feature_branch_names={extra_feature_branch_names}, "
            f"extra_feature_branch_dims={extra_feature_branch_dims}, "
            f"extra_token_injection={extra_token_injection}, "
            f"extra_branch_scale_init={extra_branch_scale_init}"
        )
    logger.info(
        f"  use_context_features={use_context_features}, context_len={context_len}, "
        f"context_dim={context_feature_dim}, overall_corr_dim={overall_corr_dim}, "
        f"context_scale_init={context_scale_init}, context_score_scale_init={context_score_scale_init}, "
        f"overall_corr_scale_init={overall_corr_scale_init}, "
        f"use_context_score_bias={use_context_score_bias}, use_overall_corr={use_overall_corr}"
    )
    if empty_cache_interval > 0:
        logger.info(f"  empty_cache_interval={empty_cache_interval} (清理显存缓存频率)")
    else:
        logger.info(f"  empty_cache_interval=0 (禁用定期显存清理)")
    if use_compressed:
        model_configs['n_transformer_layers'] = config.model.get('n_transformer_layers', 4) if config else 4
        model_configs['n_heads_single'] = n_heads
        if use_context_features:
            model = EnzymeClassificationHeadCompressedDirectContext(model_configs)
            logger.info("Using EnzymeClassificationHeadCompressedDirectContext (z: [N, C_z], direct context enabled)")
        else:
            model = EnzymeClassificationHeadCompressed(model_configs)
            logger.info("Using EnzymeClassificationHeadCompressed (z: [N, C_z])")
    else:
        model = EnzymeClassificationHead(model_configs)
        logger.info("Using EnzymeClassificationHead (z: [N, N, C_z])")
    model = model.to(device)
    model.train()
    logger.info(f"Model created")

    # 加载 checkpoint（用于继续训练）
    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Resumed from {resume_from}")
        if missing_keys:
            logger.info(f"  Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.info(f"  Unexpected keys: {unexpected_keys}")

    # 损失函数
    pairwise_criterion = None
    bce_criterion = None
    listwise_criterion = None
    if loss_type in ('pairwise', 'hybrid'):
        from alphaenzyme.loss.enzyme_kinetics_loss import PairwiseRankingLoss
        pairwise_criterion = PairwiseRankingLoss()
    if loss_type in ('listwise', 'hybrid_listwise'):
        from alphaenzyme.loss.enzyme_kinetics_loss import ReactionListNetLoss
        listwise_criterion = ReactionListNetLoss({"temperature": listwise_temperature})
    if loss_type in ('bce', 'hybrid', 'hybrid_listwise'):
        bce_criterion = BinaryClassificationLoss({"label_smoothing": label_smoothing})

    if loss_type == 'pairwise':
        criterion = pairwise_criterion
        logger.info(
            f"Using PairwiseRankingLoss (retrieval_prior_mode={retrieval_prior_mode}, "
            f"scale={retrieval_prior_scale}, threshold={retrieval_prior_threshold})"
        )
    elif loss_type == 'hybrid':
        criterion = None
        logger.info(
            f"Using HybridLoss (bce_weight={bce_weight}, pairwise_weight={pairwise_weight}, "
            f"retrieval_prior_mode={retrieval_prior_mode}, scale={retrieval_prior_scale}, "
            f"threshold={retrieval_prior_threshold}, label_smoothing={label_smoothing})"
        )
    elif loss_type == 'listwise':
        criterion = listwise_criterion
        logger.info(f"Using ReactionListNetLoss (temperature={listwise_temperature})")
    elif loss_type == 'hybrid_listwise':
        criterion = None
        logger.info(
            f"Using HybridListwiseLoss (bce_weight={bce_weight}, listwise_weight={listwise_weight}, "
            f"temperature={listwise_temperature}, label_smoothing={label_smoothing})"
        )
    else:
        criterion = bce_criterion
        logger.info(f"Using BinaryClassificationLoss (label_smoothing={label_smoothing})")

    # 创建优化器（支持 Adam 和 AdamW）
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay
        )
        logger.info(f"Using AdamW optimizer with weight_decay={weight_decay}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        logger.info(f"Using Adam optimizer")

    # 创建学习率调度器
    scheduler = None
    if use_scheduler:
        # 计算训练步数
        if uses_grouped_reaction_batches:
            # pairwise: len(dataset) = batch 数，每个 batch 一步
            per_steps_one_epoch = len(dataset) // accelerator.num_processes
        else:
            per_steps_one_epoch = len(dataset) // batch_size // accelerator.num_processes
        num_training_steps = per_steps_one_epoch * epochs

        # 计算 warmup 步数（优先使用 warmup_steps，否则使用 warmup_ratio）
        if warmup_steps is not None:
            num_warmup_steps = warmup_steps
            logger.info(f"Using warmup_steps={warmup_steps} (manually specified)")
        else:
            num_warmup_steps = int(num_training_steps * warmup_ratio)
            logger.info(f"Using warmup_ratio={warmup_ratio} ({num_warmup_steps} steps = {warmup_ratio*100:.1f}% of training)")

        if scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(f"Using Cosine scheduler with warmup")
            logger.info(f"  Warmup steps: {num_warmup_steps} / {num_training_steps} total steps")
            logger.info(f"  Steps per epoch: {per_steps_one_epoch}")
        else:
            logger.info(f"Scheduler type '{scheduler_type}' not implemented, using fixed lr")
    else:
        logger.info("Learning rate scheduler disabled, using fixed lr")

    # ==================== 使用 Accelerator 包装模型、优化器和 DataLoader ====================
    # 重要：必须用 accelerator.prepare() 包装 DataLoader，这样会自动：
    # 1. 使用 DistributedSampler 对数据进行分片
    # 2. 确保每个 rank 处理不同的数据子集（总样本数 / GPU数量）
    # 3. 自动处理 shuffle 和数据分布
    # 准备模型、优化器、数据加载器和调度器
    prepare_items = [model, optimizer, dataloader]
    if valid_dataloader is not None:
        prepare_items.append(valid_dataloader)
    if test_dataloader is not None:
        prepare_items.append(test_dataloader)
    if scheduler is not None:
        prepare_items.append(scheduler)

    prepared = accelerator.prepare(*prepare_items)
    prepared_idx = 0
    model = prepared[prepared_idx]
    prepared_idx += 1
    optimizer = prepared[prepared_idx]
    prepared_idx += 1
    dataloader = prepared[prepared_idx]
    prepared_idx += 1
    if valid_dataloader is not None:
        valid_dataloader = prepared[prepared_idx]
        prepared_idx += 1
    if test_dataloader is not None:
        test_dataloader = prepared[prepared_idx]
        prepared_idx += 1
    if scheduler is not None:
        scheduler = prepared[prepared_idx]

    # 输出数据分片信息
    if accelerator.is_main_process:
        logger.info(f"Data sharding info:")
        logger.info(f"  Total training samples: {len(dataset)}")
        logger.info(f"  Num processes (GPUs): {accelerator.num_processes}")
        logger.info(f"  Samples per rank: ~{len(dataset) // accelerator.num_processes}")
        logger.info(f"  Batch size per rank: {batch_size}")
        logger.info(f"  Batches per rank per epoch: {len(dataloader)}")
        logger.info(f"  Effective global batch size: {batch_size * accelerator.num_processes}")

    # Accelerator 会自动处理混合精度，不需要手动获取 scaler
    # 在使用 accelerator.backward() 时会自动处理梯度缩放
    logger.info(f"Using Accelerator with precision: {precision}")

    # ==================== EMA 初始化 (在 accelerator.prepare 之后) ====================
    # 分布式训练中，每个进程都有模型副本，EMA 需要从 unwrapped model 获取原始模型
    ema_wrapper = None
    if use_ema:
        # 需要从 DDP unwrapped model 获取原始模型
        unwrapped_model = model.module if hasattr(model, 'module') else model
        ema_config = {"decay": ema_decay, "mutable_param_keywords": None}
        ema_wrapper = EMAWrapperV1(ema_config)
        ema_wrapper.model = unwrapped_model
        ema_wrapper.register()
        logger.info(f"EMA enabled with decay={ema_decay}, update_interval={ema_update_interval}")

    # ==================== 显存分析 ====================
    def get_memory_stats():
        """获取显存统计信息（单位：GB）"""
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved

    def get_model_memory():
        """计算模型参数占用的显存"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
        return param_size

    def get_optimizer_memory():
        """计算优化器状态占用的显存（Adam 有 2 个状态：momentum 和 variance）"""
        optimizer_size = 0
        for param_group in optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad:
                    # Adam 为每个参数存储 2 个状态（momentum + variance）
                    optimizer_size += p.numel() * p.element_size() * 2
        return optimizer_size / 1024**3

    # 初始显存（只有模型）
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mem_before_optimizer, _ = get_memory_stats()
    model_mem = get_model_memory()
    logger.info(f"\n{'='*70}")
    logger.info(f"Memory Analysis:")
    logger.info(f"{'='*70}")
    logger.info(f"Model parameters: {model_mem:.3f} GB")

    # 优化器初始化后
    optimizer_mem = get_optimizer_memory()
    mem_after_optimizer, _ = get_memory_stats()
    logger.info(f"Optimizer states: {optimizer_mem:.3f} GB")
    logger.info(f"Total (Model + Optimizer): {mem_after_optimizer:.3f} GB")
    logger.info(f"{'='*70}\n")

    # Dropout warmup 计算
    per_steps_one_epoch_est = len(dataloader)
    total_training_steps = per_steps_one_epoch_est * epochs
    dropout_warmup_steps = int(total_training_steps * dropout_warmup_ratio)
    if dropout_warmup_ratio > 0:
        logger.info(f"Dropout warmup: disabled for first {dropout_warmup_steps} steps "
                     f"({dropout_warmup_ratio*100:.0f}% of {total_training_steps} total steps)")
        # 初始化时关闭 dropout
        set_dropout_rate(model, 0.0)

    best_early_stop_value = None
    best_early_stop_epoch = None
    epochs_without_improve = 0
    if early_stopping_patience is not None:
        if not eval_every_epoch or valid_dataloader is None:
            logger.info("Early stopping configured but disabled because validation is not running every epoch.")
            early_stopping_patience = None
        else:
            logger.info(
                f"Early stopping enabled: patience={early_stopping_patience}, metric={early_stopping_metric}"
            )

    reaction_cache = {}
    csv_mapping_cache = {}

    def compute_retrieval_prior_pair_weights(batch_s, batch_is_ligand, batch_single_mask, batch_labels):
        if retrieval_prior_mode in ('none', '') or retrieval_prior_scale <= 0:
            return None, {}
        if retrieval_prior_mode != 'batch_positive_similarity':
            raise ValueError(f"Unsupported retrieval_prior_mode: {retrieval_prior_mode}")

        pos_mask = batch_labels > 0.5
        neg_mask = ~pos_mask
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return None, {}

        protein_mask = batch_single_mask.float() * (1.0 - batch_is_ligand.float())
        protein_token_count = protein_mask.sum(dim=1, keepdim=True)
        fallback_mask = protein_token_count <= 0
        if fallback_mask.any():
            protein_mask = torch.where(fallback_mask, batch_single_mask.float(), protein_mask)
            protein_token_count = protein_mask.sum(dim=1, keepdim=True)

        pooled = (batch_s.float() * protein_mask.unsqueeze(-1)).sum(dim=1)
        pooled = pooled / protein_token_count.clamp_min(1.0)
        pooled = F.normalize(pooled, dim=-1, eps=1e-6)

        pos_repr = pooled[pos_mask]
        neg_repr = pooled[neg_mask]
        similarity = pos_repr @ neg_repr.transpose(0, 1)
        pair_weights = 1.0 + retrieval_prior_scale * torch.relu(similarity - retrieval_prior_threshold)
        stats = {
            "retrieval_prior/mean_pair_weight": float(pair_weights.mean().item()),
            "retrieval_prior/max_pair_weight": float(pair_weights.max().item()),
            "retrieval_prior/mean_pos_neg_sim": float(similarity.mean().item()),
        }
        return pair_weights, stats

    def compute_total_classification_loss(logits, labels, batch_s, batch_is_ligand, batch_single_mask):
        pair_weights = None
        retrieval_stats = {}
        if loss_type in ('pairwise', 'hybrid'):
            pair_weights, retrieval_stats = compute_retrieval_prior_pair_weights(
                batch_s, batch_is_ligand, batch_single_mask, labels
            )

        loss_breakdown = {}
        if loss_type == 'pairwise':
            pairwise_loss = pairwise_criterion(logits, labels.float(), pair_weights=pair_weights)
            loss_breakdown["pairwise_loss"] = float(pairwise_loss.detach().item())
            loss_breakdown.update(retrieval_stats)
            return pairwise_loss, loss_breakdown

        if loss_type == 'hybrid':
            bce_loss = bce_criterion(logits, labels)
            pairwise_loss = pairwise_criterion(logits, labels.float(), pair_weights=pair_weights)
            total_loss = bce_weight * bce_loss + pairwise_weight * pairwise_loss
            loss_breakdown["bce_loss"] = float(bce_loss.detach().item())
            loss_breakdown["pairwise_loss"] = float(pairwise_loss.detach().item())
            loss_breakdown.update(retrieval_stats)
            return total_loss, loss_breakdown

        if loss_type == 'listwise':
            listwise_loss = listwise_criterion(logits, labels.float())
            loss_breakdown["listwise_loss"] = float(listwise_loss.detach().item())
            return listwise_loss, loss_breakdown

        if loss_type == 'hybrid_listwise':
            bce_loss = bce_criterion(logits, labels)
            listwise_loss = listwise_criterion(logits, labels.float())
            total_loss = bce_weight * bce_loss + listwise_weight * listwise_loss
            loss_breakdown["bce_loss"] = float(bce_loss.detach().item())
            loss_breakdown["listwise_loss"] = float(listwise_loss.detach().item())
            return total_loss, loss_breakdown

        bce_loss = bce_criterion(logits, labels)
        loss_breakdown["bce_loss"] = float(bce_loss.detach().item())
        return bce_loss, loss_breakdown

    def evaluate_split(split_name, eval_dataloader, split_json_dir, split_csv_path=None):
        if eval_dataloader is None:
            return None

        split_correct = 0
        split_total = 0
        split_loss = 0.0
        all_logits = []
        all_labels = []
        all_sample_ids = []

        with torch.no_grad():
            for eval_batch in eval_dataloader:
                s = eval_batch['s'].to(device)
                z = eval_batch['z'].to(device)
                is_ligand = eval_batch['is_ligand'].to(device)
                single_mask = eval_batch['single_mask'].to(device)
                labels = eval_batch['labels'].to(device)
                drfp = eval_batch['drfp'].to(device)
                extra_features = eval_batch['extra_features'].to(device)
                context_features = eval_batch['context_features'].to(device)
                context_mask = eval_batch['context_mask'].to(device)
                context_scores = eval_batch['context_scores'].to(device)
                overall_corr = eval_batch['overall_corr'].to(device)

                if uses_grouped_reaction_batches:
                    s = s.squeeze(0)
                    z = z.squeeze(0)
                    is_ligand = is_ligand.squeeze(0)
                    single_mask = single_mask.squeeze(0)
                    labels = labels.squeeze(0)
                    drfp = drfp.squeeze(0)
                    extra_features = extra_features.squeeze(0)
                    context_features = context_features.squeeze(0)
                    context_mask = context_mask.squeeze(0)
                    context_scores = context_scores.squeeze(0)
                    overall_corr = overall_corr.squeeze(0)

                with accelerator.autocast():
                    outputs = model(
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
                    logits = outputs['logits']
                    loss, _ = compute_total_classification_loss(logits, labels, s, is_ligand, single_mask)

                if is_distributed:
                    loss_tensor = loss.detach().clone()
                    all_losses = accelerator.gather(loss_tensor)
                    batch_loss = all_losses.mean().item()
                    all_logits.append(gather_tensor_for_metrics(accelerator, logits, is_distributed))
                    all_labels.append(gather_tensor_for_metrics(accelerator, labels, is_distributed))
                else:
                    batch_loss = loss.item()
                    all_logits.append(logits.detach().cpu())
                    all_labels.append(labels.detach().cpu())

                raw_sample_ids = eval_batch['sample_ids']
                if uses_grouped_reaction_batches and raw_sample_ids and isinstance(raw_sample_ids[0], (list, tuple)):
                    raw_sample_ids = [sid for sub in raw_sample_ids for sid in sub]
                all_sample_ids.extend(gather_sample_ids(accelerator, raw_sample_ids, is_distributed))

                split_loss += batch_loss
                preds = (logits > 0).long()
                batch_correct_tensor = (preds == labels).sum()
                batch_total_tensor = torch.tensor(labels.size(0), device=labels.device)
                if is_distributed:
                    batch_correct_tensor = accelerator.gather(batch_correct_tensor).sum()
                    batch_total_tensor = accelerator.gather(batch_total_tensor).sum()
                split_correct += batch_correct_tensor.item()
                split_total += batch_total_tensor.item()

        avg_loss = split_loss / len(eval_dataloader)
        accuracy = split_correct / split_total if split_total > 0 else 0.0

        global_metrics = {
            "auc": 0.0,
            "ap": 0.0,
            "accuracy": accuracy,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        reaction_metrics = {
            "sr": {k: 0.0 for k in [1, 3, 5, 10, 20, 50]},
            "reaction_auc": 0.0,
            "reaction_ap": 0.0,
            "dcg10": 0.0,
            "ndcg10": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "reaction_count": 0,
            "positive_reaction_count": 0,
        }

        if accelerator.is_main_process:
            all_logits_t = torch.cat(all_logits, dim=0)
            all_labels_t = torch.cat(all_labels, dim=0)
            probs = torch.sigmoid(all_logits_t).numpy()
            labels_np = all_labels_t.numpy()
            global_metrics = compute_global_binary_metrics(probs, labels_np)
            reaction_metrics = compute_reaction_ranking_metrics(
                probs,
                labels_np,
                all_sample_ids,
                split_json_dir,
                reaction_cache=reaction_cache,
                csv_path=split_csv_path,
                csv_mapping_cache=csv_mapping_cache,
            )

        metrics = {
            f"{split_name}/epoch_loss": avg_loss,
            f"{split_name}/epoch_accuracy": accuracy,
            f"{split_name}/epoch_auc": global_metrics["auc"],
            f"{split_name}/epoch_ap": global_metrics["ap"],
            f"{split_name}/epoch_balanced_accuracy": global_metrics["balanced_accuracy"],
            f"{split_name}/epoch_precision": global_metrics["precision"],
            f"{split_name}/epoch_recall": global_metrics["recall"],
            f"{split_name}/epoch_f1": global_metrics["f1"],
            f"{split_name}/epoch_success_rate": reaction_metrics["sr"][3],
            f"{split_name}/epoch_sr1": reaction_metrics["sr"][1],
            f"{split_name}/epoch_sr3": reaction_metrics["sr"][3],
            f"{split_name}/epoch_sr5": reaction_metrics["sr"][5],
            f"{split_name}/epoch_sr10": reaction_metrics["sr"][10],
            f"{split_name}/epoch_sr20": reaction_metrics["sr"][20],
            f"{split_name}/epoch_sr50": reaction_metrics["sr"][50],
            f"{split_name}/epoch_reaction_auc": reaction_metrics["reaction_auc"],
            f"{split_name}/epoch_reaction_ap": reaction_metrics["reaction_ap"],
            f"{split_name}/epoch_dcg10": reaction_metrics["dcg10"],
            f"{split_name}/epoch_ndcg10": reaction_metrics["ndcg10"],
            f"{split_name}/epoch_mrr": reaction_metrics["mrr"],
            f"{split_name}/epoch_mean_rank": reaction_metrics["mean_rank"],
        }
        summary = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "global": global_metrics,
            "reaction": reaction_metrics,
            "metrics": metrics,
        }
        return summary

    # 训练
    logger.info("Starting classification training...")
    logger.info(f"Config: batch_size={batch_size}, lr={lr}, epochs={epochs}, seed={seed}, deterministic={deterministic}")
    os.makedirs(output_dir, exist_ok=True)

    def _to_plain_config(x):
        if isinstance(x, dict):
            return {k: _to_plain_config(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_plain_config(v) for v in x]
        return x

    resolved_config = {
        "task_type": task_type,
        "data": {
            "train_json": json_dir,
            "valid_json": valid_json_dir,
            "test_json": test_json_dir,
            "train_embedding_dir": embedding_dir,
            "valid_embedding_dir": valid_embedding_dir,
            "test_embedding_dir": test_embedding_dir,
            "csv_path": train_csv_path,
            "valid_csv_path": valid_csv_path,
            "test_csv_path": test_csv_path,
            "drfp_path": drfp_path if use_drfp else None,
            "max_seq_length": max_seq_length,
            "external_missing_policy": external_missing_policy if use_extra_features else None,
            "train_external_feature_banks": train_external_bank_specs if train_external_bank_specs else None,
            "valid_external_feature_banks": valid_external_bank_specs if valid_external_bank_specs else None,
            "test_external_feature_banks": test_external_bank_specs if test_external_bank_specs else None,
            "context_missing_policy": context_missing_policy if use_context_features else None,
            "train_context_cache_path": train_context_cache_path if use_context_features else None,
            "valid_context_cache_path": valid_context_cache_path if use_context_features else None,
            "test_context_cache_path": test_context_cache_path if use_context_features else None,
            "train_context_bank_path": train_context_bank_path if use_context_features else None,
            "valid_context_bank_path": valid_context_bank_path if use_context_features else None,
            "test_context_bank_path": test_context_bank_path if use_context_features else None,
        },
        "model": _to_plain_config(model_configs),
        "training": {
            "batch_size": batch_size,
            "lr": lr,
            "epochs": epochs,
            "loss_type": loss_type,
            "bce_weight": bce_weight,
            "pairwise_weight": pairwise_weight,
            "listwise_weight": listwise_weight,
            "listwise_temperature": listwise_temperature,
            "retrieval_prior_mode": retrieval_prior_mode,
            "retrieval_prior_scale": retrieval_prior_scale,
            "retrieval_prior_threshold": retrieval_prior_threshold,
            "optimizer": optimizer_type,
            "weight_decay": weight_decay,
            "beta1": beta1,
            "beta2": beta2,
            "precision": precision,
            "use_amp": use_amp,
            "seed": seed,
            "deterministic": deterministic,
            "use_scheduler": use_scheduler,
            "warmup_ratio": warmup_ratio,
            "warmup_steps": warmup_steps,
            "scheduler_type": scheduler_type,
            "num_workers": num_workers,
            "save_every_n_epochs": save_every_n_epochs,
            "output_dir": output_dir,
            "log_dir": log_dir,
        },
        "evaluation": {
            "eval_every_epoch": eval_every_epoch,
            "test_every_n_epochs": test_every_n_epochs,
            "official_eval_precision": "fp32",
            "official_eval_deduplicate_reaction_enzymes": True,
        },
        "ema": {
            "use_ema": use_ema,
            "decay": ema_decay if use_ema else None,
            "update_interval": ema_update_interval if use_ema else None,
        },
        "logging": {
            "use_swanlab": use_swanlab,
            "swanlab_project": swanlab_project if use_swanlab else None,
            "swanlab_workspace": swanlab_workspace if use_swanlab else None,
        },
    }
    with open(os.path.join(output_dir, "resolved_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_config, f, allow_unicode=True, sort_keys=False)
    with open(os.path.join(output_dir, "resolved_hparams.json"), "w", encoding="utf-8") as f:
        json.dump(hparams_config, f, indent=2, ensure_ascii=False)

    # 全局 step 计数 (用于 SwanLab)
    global_step = 0
    first_batch_analyzed = False

    for epoch in range(epochs):
        # Pairwise 模式：每个 epoch 打乱 batch 顺序
        if uses_grouped_reaction_batches:
            dataset.shuffle_batches()

        total_loss = 0
        correct = 0
        total = 0
        train_all_logits = []
        train_all_labels = []
        train_all_sample_ids = []

        # 用于计算最近 100 个 batch 的平均准确率
        recent_correct = 0
        recent_total = 0
        recent_window = 100
        recent_acc = 0.0  # 用于 SwanLab 记录

        # 初始化显存变量
        allocated = 0.0
        reserved = 0.0
        latest_loss_breakdown = {}

        for batch_idx, batch in enumerate(dataloader):
            # 定期清理显存碎片（根据配置决定是否清理）
            if empty_cache_interval > 0 and batch_idx > 0 and batch_idx % empty_cache_interval == 0:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 定期监控显存（每 50 个 batch 打印一次，只在主进程输出）
            if batch_idx > 0 and batch_idx % 50 == 0 and accelerator.is_main_process:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"  [Memory] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
            s = batch['s'].to(device)
            z = batch['z'].to(device)
            is_ligand = batch['is_ligand'].to(device)
            single_mask = batch['single_mask'].to(device)
            labels = batch['labels'].to(device)
            drfp = batch['drfp'].to(device)
            extra_features = batch['extra_features'].to(device)
            context_features = batch['context_features'].to(device)
            context_mask = batch['context_mask'].to(device)
            context_scores = batch['context_scores'].to(device)
            overall_corr = batch['overall_corr'].to(device)

            # Pairwise 模式: DataLoader(batch_size=1) 会加一个额外的 batch 维度
            if uses_grouped_reaction_batches:
                s = s.squeeze(0)
                z = z.squeeze(0)
                is_ligand = is_ligand.squeeze(0)
                single_mask = single_mask.squeeze(0)
                labels = labels.squeeze(0)
                drfp = drfp.squeeze(0)
                extra_features = extra_features.squeeze(0)
                context_features = context_features.squeeze(0)
                context_mask = context_mask.squeeze(0)
                context_scores = context_scores.squeeze(0)
                overall_corr = overall_corr.squeeze(0)

            # 第一个 batch 分析显存
            if not first_batch_analyzed:
                mem_after_data, _ = get_memory_stats()
                data_mem = mem_after_data - mem_after_optimizer
                logger.info(f"\n{'='*70}")
                logger.info(f"First Batch Memory Analysis (batch_size={batch_size}):")
                logger.info(f"{'='*70}")
                logger.info(f"Input data: {data_mem:.3f} GB")
                logger.info(f"  - s shape: {s.shape}, z shape: {z.shape}")
                logger.info(f"  - extra_features shape: {extra_features.shape}")
                logger.info(f"  - context_features shape: {context_features.shape}")
                logger.info(f"  - Max sequence length in batch: {s.shape[1]}")

            # Dropout warmup: 前 dropout_warmup_ratio 比例的步骤关闭 dropout
            if dropout_warmup_ratio > 0 and global_step == dropout_warmup_steps:
                set_dropout_rate(model, 1.0)
                if accelerator.is_main_process:
                    logger.info(f"[Step {global_step}] Dropout warmup finished, enabling dropout (rate={dropout})")

            # 前向传播
            optimizer.zero_grad()

            # 重置峰值显存统计
            if not first_batch_analyzed:
                torch.cuda.reset_peak_memory_stats()

            # 混合精度前向传播 (使用 accelerator.autocast)
            with accelerator.autocast():
                outputs = model(
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
                logits = outputs['logits']
                loss, latest_loss_breakdown = compute_total_classification_loss(
                    logits, labels, s, is_ligand, single_mask
                )

                # 第一个 batch 分析激活值（在 forward 之后测量）
                if not first_batch_analyzed:
                    mem_after_forward, _ = get_memory_stats()
                    peak_forward = torch.cuda.max_memory_allocated() / 1024**3
                    activation_mem = mem_after_forward - mem_after_data - mem_after_optimizer

            # 第一个 batch 分析激活值（在 autocast 外部打印）
            if not first_batch_analyzed:
                logger.info(f"Activations (forward): {activation_mem:.3f} GB")
                logger.info(f"Peak during forward: {peak_forward:.3f} GB")

            # 反向传播 (使用 accelerator.backward)
            # 注意：DDP 会在反向传播时自动对梯度进行 all-reduce 并除以 world_size
            # 因此梯度已经是所有进程的平均值，不需要手动处理
            if not first_batch_analyzed:
                torch.cuda.reset_peak_memory_stats()

            accelerator.backward(loss)

            # 收集 logits 和 labels 用于 train AUC/SuccessRate 计算
            if is_distributed:
                train_all_logits.append(gather_tensor_for_metrics(accelerator, logits, is_distributed))
                train_all_labels.append(gather_tensor_for_metrics(accelerator, labels, is_distributed))
            else:
                train_all_logits.append(logits.detach().cpu())
                train_all_labels.append(labels.detach().cpu())
            # Pairwise 模式: sample_ids 是 [[id1, id2, ...]] 的 nested list
            raw_sample_ids = batch['sample_ids']
            if uses_grouped_reaction_batches and raw_sample_ids and isinstance(raw_sample_ids[0], (list, tuple)):
                raw_sample_ids = [sid for sub in raw_sample_ids for sid in sub]
            train_all_sample_ids.extend(gather_sample_ids(accelerator, raw_sample_ids, is_distributed))

            # 第一个 batch 分析梯度
            if not first_batch_analyzed:
                mem_after_backward, _ = get_memory_stats()
                peak_backward = torch.cuda.max_memory_allocated() / 1024**3
                peak_mem = max(peak_forward, peak_backward)
                gradient_mem = mem_after_backward - mem_after_forward
                logger.info(f"Gradients (backward): {gradient_mem:.3f} GB")
                logger.info(f"Peak during backward: {peak_backward:.3f} GB")
                logger.info(f"\nTotal memory breakdown:")
                logger.info(f"  Model parameters:  {model_mem:.3f} GB")
                logger.info(f"  Optimizer states:  {optimizer_mem:.3f} GB")
                logger.info(f"  Input data:        {data_mem:.3f} GB")
                logger.info(f"  Activations:       {activation_mem:.3f} GB")
                logger.info(f"  Gradients:         {gradient_mem:.3f} GB")
                logger.info(f"  Total allocated:   {mem_after_backward:.3f} GB")
                logger.info(f"  Peak forward:      {peak_forward:.3f} GB")
                logger.info(f"  Peak backward:     {peak_backward:.3f} GB")
                logger.info(f"  Overall peak:      {peak_mem:.3f} GB  <-- 真实峰值显存 (with AMP)")
                logger.info(f"{'='*70}\n")
                first_batch_analyzed = True

            # 梯度裁剪（防止梯度爆炸）
            if max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 优化器步骤 (使用 accelerator 处理混合精度)
            optimizer.step()

            # 学习率调度器步骤
            if scheduler is not None:
                scheduler.step()

            # ==================== EMA 更新 ====================
            if use_ema and ema_wrapper is not None and batch_idx % ema_update_interval == 0:
                ema_wrapper.update()

            # 统计准确率
            preds = (logits > 0).long()
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)

            correct += batch_correct
            total += batch_total

            # 更新最近 100 个 batch 的统计
            recent_correct += batch_correct
            recent_total += batch_total

            # recent_correct/recent_total 在每次打印时（每 100 batch）重置为 0
            # 因此统计的是"自上次打印以来"的准确率，而非严格滑动窗口

            # ==================== 分布式训练：收集所有进程的 loss 用于日志记录 ====================
            # 注意：这里的 gather 操作仅用于日志记录，不影响训练
            # 反向传播时，DDP 已经自动对梯度进行了 all-reduce 平均
            # 但每个进程的 loss 值是基于各自的 batch 计算的，需要 gather 后平均才能得到全局 loss
            if is_distributed:
                # 将 loss 转换为 tensor 以便 gather
                loss_tensor = loss.detach().clone()
                # gather 所有进程的 loss
                all_losses = accelerator.gather(loss_tensor)
                # 计算全局平均 loss（用于日志记录）
                loss_value = all_losses.mean().item()
            else:
                loss_value = loss.item()

            total_loss += loss_value

            # 保存 loss 值用于后续打印（因为后面会删除 loss 变量）
            # loss_value 已经在上面计算好了

            # 显式删除中间变量以帮助释放显存（必须在使用之后）
            del outputs, logits, loss

            # 实时输出 loss 和 accuracy
            if batch_idx % 100 == 0 and accelerator.is_main_process:
                running_acc = correct / total if total > 0 else 0.0
                # 计算最近 100 个 batch 的平均准确率
                if batch_idx < recent_window:
                    recent_acc = recent_correct / recent_total if recent_total > 0 else 0.0
                    logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss_value:.4f}, Recent {batch_idx+1} Batches Acc: {recent_acc:.4f}, Running Acc: {running_acc:.4f}")
                else:
                    recent_acc = recent_correct / recent_total if recent_total > 0 else 0.0
                    logger.info(f"  Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss_value:.4f}, Recent {recent_window} Batches Acc: {recent_acc:.4f}, Running Acc: {running_acc:.4f}")
                # 重置最近统计
                recent_correct = 0
                recent_total = 0

            # ==================== SwanLab 批量指标记录 ====================
            # 获取当前显存统计
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3

            # 获取当前 batch 的序列长度
            batch_seq_length = s.shape[1]  # 当前 batch 的序列长度（padding 后所有序列长度相同）

            # 记录当前 batch 的指标 (每 10 个 batch 记录一次以减少开销)
            if batch_idx % 10 == 0:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                train_metrics = {
                    "train/loss": loss_value,
                    "train/accuracy": correct / total if total > 0 else 0.0,
                    "train/recent_accuracy": recent_acc if batch_idx > 0 else 0.0,
                    "train/learning_rate": current_lr,  # 新增：记录学习率
                    "memory/allocated_gb": allocated if torch.cuda.is_available() else 0.0,
                    "memory/reserved_gb": reserved if torch.cuda.is_available() else 0.0,
                    "data/sequence_length": batch_seq_length,  # 当前 batch 的序列长度
                }
                if "bce_loss" in latest_loss_breakdown:
                    train_metrics["train/bce_loss"] = latest_loss_breakdown["bce_loss"]
                if "pairwise_loss" in latest_loss_breakdown:
                    train_metrics["train/pairwise_loss"] = latest_loss_breakdown["pairwise_loss"]
                for key in [
                    "retrieval_prior/mean_pair_weight",
                    "retrieval_prior/max_pair_weight",
                    "retrieval_prior/mean_pos_neg_sim",
                ]:
                    if key in latest_loss_breakdown:
                        train_metrics[key] = latest_loss_breakdown[key]
                if use_swanlab and accelerator.is_main_process:
                    swanlab.log(train_metrics, step=global_step)

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        # 计算 train 全局指标和 reaction-level 排名指标 (只在主进程)
        train_global_metrics = {
            "auc": 0.0,
            "ap": 0.0,
            "accuracy": accuracy,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
        train_reaction_metrics = {
            "sr": {k: 0.0 for k in [1, 3, 5, 10, 20, 50]},
            "reaction_auc": 0.0,
            "reaction_ap": 0.0,
            "dcg10": 0.0,
            "ndcg10": 0.0,
            "mrr": 0.0,
            "mean_rank": 0.0,
            "reaction_count": 0,
            "positive_reaction_count": 0,
        }
        train_auc = 0.0
        train_success_rate = 0.0
        train_sr = {k: 0.0 for k in [1, 3, 5, 10, 20, 50]}
        if accelerator.is_main_process:
            all_logits_t = torch.cat(train_all_logits, dim=0)
            all_labels_t = torch.cat(train_all_labels, dim=0)
            probs = torch.sigmoid(all_logits_t).numpy()
            labels_np = all_labels_t.numpy()
            train_global_metrics = compute_global_binary_metrics(probs, labels_np)
            train_reaction_metrics = compute_reaction_ranking_metrics(
                probs,
                labels_np,
                train_all_sample_ids,
                json_dir,
                reaction_cache=reaction_cache,
                csv_path=train_csv_path,
                csv_mapping_cache=csv_mapping_cache,
            )
            train_auc = train_global_metrics["auc"]
            train_sr = train_reaction_metrics["sr"]
            train_success_rate = train_sr[3]

        # 只在主进程输出 epoch 总结
        if accelerator.is_main_process:
            logger.info(
                f"Epoch {epoch}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, "
                f"AUC = {train_auc:.4f}, AP = {train_global_metrics['ap']:.4f}, "
                f"RxnAUC = {train_reaction_metrics['reaction_auc']:.4f}, "
                f"RxnAP = {train_reaction_metrics['reaction_ap']:.4f}, "
                f"DCG@10 = {train_reaction_metrics['dcg10']:.4f}, "
                f"NDCG@10 = {train_reaction_metrics['ndcg10']:.4f}, "
                f"SuccessRate = {train_success_rate:.4f}, "
                f"SR@1 = {train_sr[1]:.4f}, SR@3 = {train_sr[3]:.4f}, SR@5 = {train_sr[5]:.4f}, "
                f"SR@10 = {train_sr[10]:.4f}, SR@20 = {train_sr[20]:.4f}, SR@50 = {train_sr[50]:.4f}"
            )

        # ==================== SwanLab Epoch 指标记录 ====================
        # 获取峰值显存统计
        peak_forward_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0

        epoch_metrics = {
            "train/epoch_loss": avg_loss,
            "train/epoch_accuracy": accuracy,
            "train/epoch_auc": train_auc,
            "train/epoch_ap": train_global_metrics["ap"],
            "train/epoch_balanced_accuracy": train_global_metrics["balanced_accuracy"],
            "train/epoch_precision": train_global_metrics["precision"],
            "train/epoch_recall": train_global_metrics["recall"],
            "train/epoch_f1": train_global_metrics["f1"],
            "train/epoch_success_rate": train_success_rate,
            "train/epoch_sr1": train_sr[1],
            "train/epoch_sr3": train_sr[3],
            "train/epoch_sr5": train_sr[5],
            "train/epoch_sr10": train_sr[10],
            "train/epoch_sr20": train_sr[20],
            "train/epoch_sr50": train_sr[50],
            "train/epoch_reaction_auc": train_reaction_metrics["reaction_auc"],
            "train/epoch_reaction_ap": train_reaction_metrics["reaction_ap"],
            "train/epoch_dcg10": train_reaction_metrics["dcg10"],
            "train/epoch_ndcg10": train_reaction_metrics["ndcg10"],
            "train/epoch_mrr": train_reaction_metrics["mrr"],
            "train/epoch_mean_rank": train_reaction_metrics["mean_rank"],
            "memory/peak_forward_gb": peak_forward_gb,
        }
        if use_swanlab and accelerator.is_main_process:
            swanlab.log(epoch_metrics, step=global_step)

        # ==================== 验证/测试评估 (所有进程参与，仅主进程输出日志) ====================
        if eval_every_epoch and valid_dataloader is not None:
            if accelerator.is_main_process:
                logger.info(f"Running validation on epoch {epoch}...")

            if use_ema and ema_wrapper is not None:
                ema_wrapper.apply_shadow()
                if accelerator.is_main_process:
                    logger.info("  Using EMA weights for validation/test")
            model.eval()

            valid_summary = evaluate_split("valid", valid_dataloader, valid_json_dir, split_csv_path=valid_csv_path)
            valid_accuracy = valid_summary["accuracy"]
            valid_auc = valid_summary["global"]["auc"]
            valid_sr = valid_summary["reaction"]["sr"]
            valid_success_rate = valid_sr[3]

            # 只在主进程输出验证结果
            if accelerator.is_main_process:
                logger.info(
                    f"  Validation - Loss: {valid_summary['loss']:.4f}, Accuracy: {valid_accuracy:.4f}, "
                    f"AUC: {valid_auc:.4f}, AP: {valid_summary['global']['ap']:.4f}, "
                    f"RxnAUC: {valid_summary['reaction']['reaction_auc']:.4f}, "
                    f"RxnAP: {valid_summary['reaction']['reaction_ap']:.4f}, "
                    f"DCG@10: {valid_summary['reaction']['dcg10']:.4f}, "
                    f"NDCG@10: {valid_summary['reaction']['ndcg10']:.4f}, "
                    f"SuccessRate: {valid_success_rate:.4f}, "
                    f"SR@1: {valid_sr[1]:.4f}, SR@3: {valid_sr[3]:.4f}, SR@5: {valid_sr[5]:.4f}, "
                    f"SR@10: {valid_sr[10]:.4f}, SR@20: {valid_sr[20]:.4f}, SR@50: {valid_sr[50]:.4f}"
                )

            if use_swanlab and accelerator.is_main_process:
                swanlab.log(valid_summary["metrics"], step=global_step)

            if test_dataloader is not None and test_every_n_epochs > 0 and (epoch + 1) % test_every_n_epochs == 0:
                if accelerator.is_main_process:
                    logger.info(f"Running test monitor on epoch {epoch}...")
                test_summary = evaluate_split("test", test_dataloader, test_json_dir, split_csv_path=test_csv_path)
                if accelerator.is_main_process:
                    test_sr = test_summary["reaction"]["sr"]
                    logger.info(
                        f"  Test - Loss: {test_summary['loss']:.4f}, Accuracy: {test_summary['accuracy']:.4f}, "
                        f"AUC: {test_summary['global']['auc']:.4f}, AP: {test_summary['global']['ap']:.4f}, "
                        f"RxnAUC: {test_summary['reaction']['reaction_auc']:.4f}, "
                        f"RxnAP: {test_summary['reaction']['reaction_ap']:.4f}, "
                        f"DCG@10: {test_summary['reaction']['dcg10']:.4f}, "
                        f"NDCG@10: {test_summary['reaction']['ndcg10']:.4f}, "
                        f"SuccessRate: {test_sr[3]:.4f}, "
                        f"SR@1: {test_sr[1]:.4f}, SR@3: {test_sr[3]:.4f}, SR@5: {test_sr[5]:.4f}, "
                        f"SR@10: {test_sr[10]:.4f}, SR@20: {test_sr[20]:.4f}, SR@50: {test_sr[50]:.4f}"
                    )
                if use_swanlab and accelerator.is_main_process:
                    swanlab.log(test_summary["metrics"], step=global_step)

            should_stop = False
            if early_stopping_patience is not None:
                monitored_values = {
                    "loss": valid_summary["loss"],
                    "accuracy": valid_accuracy,
                    "auc": valid_auc,
                    "sr1": valid_sr[1],
                    "sr3": valid_sr[3],
                    "sr5": valid_sr[5],
                    "sr10": valid_sr[10],
                    "reaction_auc": valid_summary["reaction"]["reaction_auc"],
                    "reaction_ap": valid_summary["reaction"]["reaction_ap"],
                    "dcg10": valid_summary["reaction"]["dcg10"],
                    "ndcg10": valid_summary["reaction"]["ndcg10"],
                }
                if early_stopping_metric not in monitored_values:
                    raise ValueError(
                        f"Unsupported early_stopping_metric='{early_stopping_metric}'. "
                        f"Available: {sorted(monitored_values.keys())}"
                    )

                current_early_stop_value = monitored_values[early_stopping_metric]
                if accelerator.is_main_process:
                    is_better = (
                        best_early_stop_value is None or
                        (
                            current_early_stop_value < best_early_stop_value - 1e-8
                            if early_stopping_metric == "loss"
                            else current_early_stop_value > best_early_stop_value + 1e-8
                        )
                    )

                    if is_better:
                        best_early_stop_value = current_early_stop_value
                        best_early_stop_epoch = epoch + 1
                        epochs_without_improve = 0

                        unwrapped_model = model.module if hasattr(model, 'module') else model
                        best_path = os.path.join(output_dir, 'model_best.pt')
                        torch.save(unwrapped_model.state_dict(), best_path)
                        logger.info(
                            f"  Early stopping metric improved: {early_stopping_metric}="
                            f"{current_early_stop_value:.6f}; best model saved to {best_path}"
                        )
                    else:
                        epochs_without_improve += 1
                        logger.info(
                            f"  Early stopping metric did not improve for {epochs_without_improve}/"
                            f"{early_stopping_patience} epoch(s)"
                        )
                        if epochs_without_improve >= early_stopping_patience:
                            should_stop = True

                stop_flag = torch.tensor(int(should_stop), device=device)
                if is_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.broadcast(stop_flag, src=0)
                should_stop = bool(stop_flag.item())

            # 恢复训练模式
            if use_ema and ema_wrapper is not None:
                ema_wrapper.restore()
                logger.info("  Restored training weights from EMA")
            model.train()

            if early_stopping_patience is not None and should_stop:
                if accelerator.is_main_process:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best epoch: {best_early_stop_epoch}, "
                        f"best {early_stopping_metric}: {best_early_stop_value:.6f}"
                    )
                accelerator.wait_for_everyone()
                break

        # 重置峰值显存统计
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # 保存checkpoint (仅主进程)
        # 同步所有进程，确保所有进程都完成了当前 epoch
        accelerator.wait_for_everyone()
        if (epoch + 1) % save_every_n_epochs == 0 and accelerator.is_main_process:
            # 获取原始模型 (DDP 包装后的模型需要用 model.module 获取)
            unwrapped_model = model.module if hasattr(model, 'module') else model
            save_path = os.path.join(output_dir, f'model_epoch{epoch+1}.pt')
            if use_ema and ema_wrapper is not None:
                ema_wrapper.apply_shadow()
                torch.save(unwrapped_model.state_dict(), save_path)
                ema_wrapper.restore()
                logger.info(f"  Model (with EMA) saved to {save_path}")
            else:
                torch.save(unwrapped_model.state_dict(), save_path)
                logger.info(f"  Model saved to {save_path}")

    logger.info("Classification training complete!")

    # 保存最终模型 (仅主进程)
    # 同步所有进程
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 获取原始模型 (DDP 包装后的模型需要用 model.module 获取)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        save_path = os.path.join(output_dir, 'model_final.pt')
        if use_ema and ema_wrapper is not None:
            ema_wrapper.apply_shadow()
            torch.save(unwrapped_model.state_dict(), save_path)
            ema_wrapper.restore()
        else:
            torch.save(unwrapped_model.state_dict(), save_path)
        logger.info(f"Final model saved to {save_path}")


if __name__ == '__main__':
    # 检查是否有 --config 参数
    argv = sys.argv[1:]
    has_config = '--config' in argv or '-c' in argv

    if has_config:
        # 使用 YAML 配置模式
        main_classification()
    elif len(argv) > 0 and argv[0] == '--classification':
        # 旧的命令行模式（兼容）
        sys.argv.pop(1)  # 移除 --classification
        main_classification()
    else:
        # 默认使用二分类训练
        main_classification()
