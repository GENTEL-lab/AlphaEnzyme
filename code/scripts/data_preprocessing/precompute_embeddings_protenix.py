#!/usr/bin/env python
"""
使用官方 Protenix 提取 Pairformer Embedding

从官方 Protenix 模型中提取 s 和 z embedding，
格式与项目现有的 embedding 兼容。

使用方法:
    cd <bundle>/code
    uv run python scripts/data_preprocessing/precompute_embeddings_protenix.py \
        --input_json_dir data/processed/specificity_50w/train/json \
        --output_dir data/processed/specificity_50w/train/pairformer \
        --checkpoint /path/to/protenix_base_20250630_v1.0.0.pt \
        --batch_size 1 \
        --num_workers 4 \
        --compress_z

输出格式:
    每个样本生成一个 .npz 文件:
    {
        's': [N_token, 384],      # single embedding (float16 bits)
        'z': [N_token, N_token, 128],  # pair embedding (float16 bits)
                                        # 开启 --compress_z 后变为 [N_token, 128]
        'is_ligand': [N_token],   # 1 for ligand, 0 for protein
        'single_mask': [N_token],  # all 1s (valid tokens)
    }
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
from contextlib import redirect_stderr

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# Silence RDKit warnings that cause slow I/O during embedding extraction
warnings.filterwarnings('ignore', message='.*not removing hydrogen atom.*')
warnings.filterwarnings('ignore', message='.*UFFTYPER.*')

# Suppress RDKit logging entirely via rdBase
try:
    from rdkit import RDLogger
    rdkit_logger = RDLogger.logger()
    rdkit_logger.setLevel(RDLogger.ERROR)
except ImportError:
    pass

# 添加项目路径 - 使用相对于脚本目录的路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

OFFICIAL_PROTENIX_ROOT = os.environ.get("ALPHAENZYME_OFFICIAL_PROTENIX_ROOT")
if OFFICIAL_PROTENIX_ROOT:
    sys.path.insert(0, OFFICIAL_PROTENIX_ROOT)
    from protenix.model.protenix import Protenix, update_input_feature_dict
    from protenix.config.config import parse_configs
    from protenix.data.inference.json_parser import add_entity_atom_array
    from protenix.data.inference.json_to_feature import SampleDictToFeatures
    from protenix.data.core.featurizer import Featurizer
    from protenix.data.tokenizer import AtomArrayTokenizer
    from protenix.data.constants import STD_RESIDUES_WITH_GAP
    HAS_CUEQUIVARIANCE = False
    try:
        import cuequivariance_torch  # noqa: F401
        HAS_CUEQUIVARIANCE = True
    except ImportError:
        pass
else:
    raise RuntimeError("Set ALPHAENZYME_OFFICIAL_PROTENIX_ROOT to the pinned upstream Protenix checkout")

# MSA residue mapping (protein only, single letter code)
# Standard amino acids in single letter code, matching AlphaFold3 MSA encoding
MSA_PROTEIN_SINGLE_LETTER = list('ARNDCQEGHILKMFPSTWYV-')  # 21 amino acids + gap
MSA_AA_TO_ID = {res: i for i, res in enumerate(MSA_PROTEIN_SINGLE_LETTER)}


def parse_a3m(a3m_content: str) -> list:
    """Parse A3M file content and return list of sequences."""
    sequences = []
    current_seq = []

    for line in a3m_content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if current_seq:
                sequences.append(''.join(current_seq))
                current_seq = []
        else:
            current_seq.append(line)

    if current_seq:
        sequences.append(''.join(current_seq))

    return sequences


def compute_msa_features(sequences: list, max_depth: int = 1024) -> tuple:
    """
    Compute profile and deletion_mean from MSA sequences.

    Args:
        sequences: List of aligned sequences (first is query)
        max_depth: Maximum number of sequences to use

    Returns:
        profile: [seq_len, 32] - amino acid frequency
        deletion_mean: [seq_len, 1] - average deletion count
    """
    if not sequences or len(sequences) == 0:
        return None, None

    # Limit depth
    sequences = sequences[:max_depth]

    # Get query sequence length (remove insertions for reference)
    query_seq = sequences[0]
    ref_len = sum(1 for c in query_seq if c in MSA_AA_TO_ID)

    num_seqs = len(sequences)

    # Initialize arrays
    msa_array = np.zeros((num_seqs, ref_len), dtype=np.int32)
    deletion_array = np.zeros((num_seqs, ref_len), dtype=np.int32)

    # Parse sequences
    for i, seq in enumerate(sequences):
        del_count = 0
        pos = 0
        for char in seq:
            if char in MSA_AA_TO_ID:
                if pos < ref_len:
                    msa_array[i, pos] = MSA_AA_TO_ID[char]
                    deletion_array[i, pos] = del_count
                    del_count = 0
                    pos += 1
            else:
                # Lowercase letters indicate insertions, count as deletions for alignment
                del_count += 1

    # Compute profile (frequency of each amino acid)
    # One-hot encode and average over sequences
    profile = np.zeros((ref_len, 32), dtype=np.float32)
    for j in range(ref_len):
        for k in range(32):
            profile[j, k] = (msa_array[:, j] == k).mean()

    # Compute deletion_mean
    deletion_mean = deletion_array.mean(axis=0, keepdims=True).T.astype(np.float32)

    return profile, deletion_mean


def load_msa_for_sequence(sequence: str, msa_mapping: dict, msa_dir: str, max_depth: int = 1024) -> tuple:
    """
    Load MSA for a given sequence and compute features.

    Args:
        sequence: Protein sequence
        msa_mapping: seq_to_msa mapping dict
        msa_dir: Directory containing MSA files
        max_depth: Maximum MSA depth

    Returns:
        profile, deletion_mean or None, None if not found
    """
    if msa_mapping is None or sequence not in msa_mapping:
        return None, None

    msa_file = msa_mapping[sequence]
    msa_path = os.path.join(msa_dir, msa_file)

    if not os.path.exists(msa_path):
        return None, None

    try:
        with open(msa_path, 'r') as f:
            a3m_content = f.read()

        sequences = parse_a3m(a3m_content)
        if len(sequences) == 0:
            return None, None

        return compute_msa_features(sequences, max_depth)
    except Exception as e:
        print(f"Warning: Failed to load MSA from {msa_path}: {e}")
        return None, None


def parse_args():
    parser = argparse.ArgumentParser(description='Precompute Protenix Embeddings')
    parser.add_argument('--input_json_dir', type=str, required=True,
                        help='Directory containing input JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--model_name', type=str, default='protenix_base_20250630_v1.0.0',
                        help='Model name')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--n_cycle', type=int, default=1,
                        help='Number of recycling cycles')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index (for distributed processing)')
    parser.add_argument('--end_idx', type=int, default=-1,
                        help='End index (for distributed processing, -1 for all)')
    parser.add_argument('--msa_mapping', type=str,
                        default=None,
                        help='Path to seq_to_msa_mapping.pkl (optional, for MSA features)')
    parser.add_argument('--msa_dir', type=str, default=None,
                        help='Directory containing MSA files')
    parser.add_argument('--use_msa', action='store_true', default=True,
                        help='Enable MSA features (default: enabled)')
    parser.add_argument('--no_msa', action='store_true', default=False,
                        help='Disable MSA features')
    parser.add_argument('--max_msa_depth', type=int, default=1024,
                        help='Maximum MSA depth to use')
    parser.add_argument('--require_msa', action='store_true', default=False,
                        help='Fail a sample instead of using zero dummy MSA when true MSA is missing')
    # Compression arguments
    parser.add_argument('--compress_z', action='store_true', default=False,
                        help='Extract then auto-compress z: (n,n,d) -> (n,d) via mean(axis=0)')
    parser.add_argument('--compress_dst_dir', type=str, default=None,
                        help='Compression output directory, default overwrites in-place')
    parser.add_argument('--compress_workers', type=int, default=16,
                        help='Number of parallel workers for compression')
    parser.add_argument('--skip_post_compress', action='store_true', default=False,
                        help='Skip the legacy post-process compression pass after saving NPZ files.')
    return parser.parse_args()


def load_model(checkpoint_path: str, model_name: str, device: str, use_msa: bool = False):
    """加载官方 Protenix 模型"""
    print(f"Loading model from {checkpoint_path}...")
    print(f"  MSA enabled: {use_msa}")

    # 使用官方仓库的配置 - 从基础配置开始
    # 先创建一个包含所有必要配置的字典
    configs = {}

    if OFFICIAL_PROTENIX_ROOT:
        from configs.configs_base import model_configs as base_model_configs

        for key in [
            'c_s', 'c_z', 'c_s_inputs', 'c_atom', 'c_atompair', 'c_token',
            'n_blocks', 'max_atoms_per_token', 'no_bins', 'sigma_data',
            'triangle_multiplicative', 'triangle_attention',
        ]:
            if key in base_model_configs:
                configs[key] = base_model_configs[key]
        if not HAS_CUEQUIVARIANCE:
            configs['triangle_multiplicative'] = 'torch'
            configs['triangle_attention'] = 'torch'

    # 添加基础配置
    configs['enable_tf32'] = True
    configs['enable_diffusion_shared_vars_cache'] = False
    configs['enable_efficient_fusion'] = False
    configs['N_cycle'] = 10
    configs['train_confidence_only'] = False
    configs['c_s'] = 384
    configs['c_z'] = 128
    configs['c_s_inputs'] = 449
    configs['diffusion_batch_size'] = 1
    configs['mc_dropout_rate'] = 0.4

    # MSA 配置：根据 use_msa 参数设置
    msa_n_blocks = 4 if use_msa else 0

    # 构建模型配置
    configs['model'] = {
        'N_model_seed': 0,
        'N_cycle': 10,
        'input_embedder': {
            'c_atom': 128,
            'c_atompair': 16,
            'c_token': 384,
        },
        'relative_position_encoding': {
            'r_max': 32,
            's_max': 2,
            'c_z': 128,
        },
        'template_embedder': {
            'c': 64,
            'c_z': 128,
            'n_blocks': 2,
            'dropout': 0.0,
        },
        'msa_module': {
            'c_m': 64,
            'c_z': 128,
            'c_s_inputs': 449,
            'n_blocks': msa_n_blocks,
            'msa_dropout': 0.0,
            'pair_dropout': 0.0,
        },
        'pairformer': {
            'n_blocks': 48,
            'c_z': 128,
            'c_s': 384,
            'n_heads': 16,
            'dropout': 0.0,
        },
        'constraint_embedder': {
            'pocket_embedder': {
                'enable': False,
            },
            'contact_embedder': {
                'enable': False,
            },
            'contact_atom_embedder': {
                'enable': False,
            },
            'substructure_embedder': {
                'enable': False,
            },
            'c_constraint_z': 128,
        },
        'diffusion_module': {
            'use_fine_grained_checkpoint': True,
            'sigma_data': 16.0,
            'c_token': 768,
            'c_atom': 128,
            'c_atompair': 16,
            'c_z': 128,
            'c_s': 384,
            'c_s_inputs': 449,
            'atom_encoder': {
                'n_blocks': 3,
                'n_heads': 4,
            },
            'transformer': {
                'n_blocks': 24,
                'n_heads': 16,
            },
            'atom_decoder': {
                'n_blocks': 3,
                'n_heads': 4,
            },
        },
        'confidence_head': {
            'c_z': 128,
            'c_s': 384,
            'c_s_inputs': 449,
            'n_blocks': 4,
            'max_atoms_per_token': 24,
            'pairformer_dropout': 0.0,
            'hidden_scale_up': False,
            'distance_bin_start': 3.25,
            'distance_bin_end': 52.0,
            'distance_bin_step': 1.25,
            'stop_gradient': True,
        },
        'distogram_head': {
            'c_z': 128,
            'no_bins': 64,
        },
    }

    # 添加训练和推理配置
    configs['train_noise_sampler'] = {
        'p_mean': -1.2,
        'p_std': 1.5,
        'sigma_data': 16.0,
    }
    configs['inference_noise_scheduler'] = {
        's_max': 160.0,
        's_min': 0.0004,
        'rho': 7,
        'sigma_data': 16.0,
    }
    configs['sample_diffusion'] = {
        'gamma0': 0.8,
        'gamma_min': 1.0,
        'noise_scale_lambda': 1.003,
        'step_scale_eta': 1.5,
        'N_step': 200,
        'N_sample': 5,
    }
    configs['skip_amp'] = {
        'sample_diffusion': False,
        'confidence_head': False,
    }
    configs['infer_setting'] = {
        'chunk_size': 256,
        'sample_diffusion_chunk_size': 5,
    }

    # 添加 data 配置
    configs['data'] = {
        'msa': {
            'enable': False,
        }
    }

    # 设置 checkpoint 路径
    configs['load_checkpoint_dir'] = os.path.dirname(checkpoint_path)

    # 解析配置
    parsed_configs = parse_configs(configs)

    # 创建模型
    print(f"  Creating model...")
    model = Protenix(parsed_configs)

    # 加载权重
    print(f"  Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']

    # 移除 'module.' 前缀（如果存在）
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 加载权重
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")

    model = model.to(device)
    model.eval()

    # 打印关键配置用于对比
    print("\n=== Model Configuration ===")
    print(f"  c_s: {parsed_configs.c_s}")
    print(f"  c_z: {parsed_configs.c_z}")
    print(f"  c_s_inputs: {parsed_configs.c_s_inputs}")
    print(f"  mc_dropout_rate: {parsed_configs.mc_dropout_rate}")
    print(f"  enable_tf32: {parsed_configs.enable_tf32}")
    print(f"  enable_diffusion_shared_vars_cache: {parsed_configs.enable_diffusion_shared_vars_cache}")
    print(f"  pairformer.n_blocks: {parsed_configs.model.pairformer.n_blocks}")
    print(f"  pairformer.dropout: {parsed_configs.model.pairformer.dropout}")
    print(f"  msa_module.n_blocks: {parsed_configs.model.msa_module.n_blocks}")
    print(f"  Model training mode: {model.training}")
    print(f"  Pairformer training mode: {model.pairformer_stack.training}")

    print(f"  Model loaded successfully")
    return model, parsed_configs


def json_to_features(
    json_path: str,
    msa_mapping: dict = None,
    msa_dir: str = None,
    max_msa_depth: int = 1024,
    require_msa: bool = False,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    将 Protenix JSON 文件转换为特征

    Args:
        json_path: JSON 文件路径
        msa_mapping: 序列到 MSA 文件的映射字典
        msa_dir: MSA 文件目录
        max_msa_depth: 最大 MSA 深度

    Returns:
        特征字典，如果失败返回 None
    """
    try:
        with open(json_path, 'r') as f:
            sample_dict = json.load(f)

        # 从 JSON 中提取蛋白质序列
        protein_sequence = None
        for entity in sample_dict.get('sequences', []):
            if 'proteinChain' in entity:
                protein_sequence = entity['proteinChain']['sequence']
                break

        # 添加 atom_array（与验证脚本保持一致）
        sample_dict = add_entity_atom_array(sample_dict)

        # 解析 JSON 并构建特征
        sample_processor = SampleDictToFeatures(sample_dict)
        atom_array = sample_processor.get_atom_array()

        # Tokenize
        aa_tokenizer = AtomArrayTokenizer(atom_array)
        token_array = aa_tokenizer.get_token_array()

        # Featurize
        featurizer = Featurizer(token_array, atom_array)
        feature_dict = featurizer.get_all_input_features()

        # 添加 frame 信息
        token_array_with_frame = featurizer.get_token_frame(
            token_array=token_array,
            atom_array=atom_array,
            ref_pos=feature_dict["ref_pos"],
            ref_mask=feature_dict["ref_mask"],
        )

        feature_dict["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()
        feature_dict["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()

        # 添加 token-level is_ligand（通过 centre_atom_index 获取）
        # 获取每个 token 的中心原子索引
        centre_atom_indices = []
        for token in token_array.tokens:
            centre_atom_indices.append(token.centre_atom_index)

        # 使用中心原子的 is_ligand 作为 token 的 is_ligand
        token_is_ligand = atom_array.is_ligand[centre_atom_indices]
        feature_dict["is_ligand"] = torch.Tensor(token_is_ligand).long()

        # 计算序列长度（仅蛋白质部分，排除配体）
        n_tokens = feature_dict["restype"].shape[0]
        n_protein_tokens = (~feature_dict["is_ligand"].bool()).sum().item()

        # 尝试加载 MSA 特征
        profile, deletion_mean = None, None
        if protein_sequence and msa_mapping is not None:
            profile, deletion_mean = load_msa_for_sequence(
                protein_sequence, msa_mapping, msa_dir, max_msa_depth
            )

        if profile is not None:
            # 将 MSA 特征扩展到总 token 数（蛋白质 + 配体）
            # 蛋白质部分用 MSA，配体部分用零
            full_profile = torch.zeros(n_tokens, 32)
            full_deletion = torch.zeros(n_tokens, 1)

            # 计算要复制的蛋白质 token 数（取 MSA 长度和实际蛋白质 token 数的较小值）
            n_copy = min(profile.shape[0], n_protein_tokens)
            full_profile[:n_copy] = torch.from_numpy(profile[:n_copy])
            full_deletion[:n_copy] = torch.from_numpy(deletion_mean[:n_copy])

            feature_dict["profile"] = full_profile
            feature_dict["deletion_mean"] = full_deletion
            print(f"  Loaded MSA features for {os.path.basename(json_path)} "
                  f"(MSA: {profile.shape[0]}, protein tokens: {n_protein_tokens}, total: {n_tokens})")
        else:
            if require_msa:
                raise ValueError(
                    f"Required true MSA is missing for {os.path.basename(json_path)}"
                )
            # 没有 MSA，使用 dummy 值
            feature_dict["profile"] = torch.zeros(n_tokens, 32)  # [N_token, 32]
            feature_dict["deletion_mean"] = torch.zeros(n_tokens, 1)  # [N_token, 1]

        return feature_dict

    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_embedding(model, input_features: Dict[str, torch.Tensor], device: str, n_cycle: int = 1) -> tuple:
    """
    提取 Protenix 的 pairformer embedding

    Args:
        model: Protenix 模型
        input_features: 输入特征字典
        device: 设备
        n_cycle: 循环次数

    Returns:
        s: [N_token, C_s] - single embedding
        z: [N_token, N_token, C_z] - pair embedding
    """
    # 将输入移动到设备并添加 batch 维度
    input_features = {k: v.to(device).unsqueeze(0) if isinstance(v, torch.Tensor) else v
                     for k, v in input_features.items()}

    with torch.no_grad():
        # 更新输入特征（添加 d_lm, v_lm, pad_info）
        input_features = update_input_feature_dict(input_features)

        # 生成相对位置编码
        input_features = model.relative_position_encoding.generate_relp(input_features)

        # 使用 get_pairformer_output 方法
        # Returns: s_inputs, s, z
        _, s, z = model.get_pairformer_output(
            input_features,
            N_cycle=n_cycle,
            inplace_safe=True,
            chunk_size=None,
        )

    # 提取 embedding（移除 batch 维度）
    s = s.squeeze(0)  # [N_token, C_s]
    z = z.squeeze(0)  # [N_token, N_token, C_z]

    return s, z


def process_json_files(args):
    """处理 JSON 文件并生成 embedding"""
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取所有 JSON 文件
    json_dir = Path(args.input_json_dir)
    json_files = sorted(list(json_dir.glob("*.json")))
    print(f"Found {len(json_files)} JSON files")

    # 处理索引范围
    if args.end_idx == -1:
        args.end_idx = len(json_files)
    json_files = json_files[args.start_idx:args.end_idx]
    print(f"Processing files {args.start_idx} to {args.end_idx}")

    # 加载 MSA 映射（如果需要）
    msa_mapping = None
    use_msa = args.use_msa and not args.no_msa

    if use_msa and args.msa_mapping:
        print(f"Loading MSA mapping from {args.msa_mapping}...")
        with open(args.msa_mapping, 'rb') as f:
            msa_data = pickle.load(f)
            msa_mapping = msa_data.get('seq_to_msa', msa_data)
        print(f"  Loaded {len(msa_mapping)} sequence mappings")

    # 加载模型
    model, configs = load_model(args.checkpoint, args.model_name, args.device, use_msa)

    # 统计
    success_count = 0
    error_count = 0
    msa_found_count = 0

    # 处理每个文件
    for json_path in tqdm(json_files, desc="Extracting embeddings"):
        sample_id = json_path.stem
        output_path = os.path.join(args.output_dir, f"{sample_id}.npz")

        # 跳过已存在的文件
        if os.path.exists(output_path):
            print(f"Skipping {sample_id}, already exists")
            success_count += 1
            continue

        # 转换 JSON 为特征
        features = json_to_features(
            str(json_path),
            msa_mapping=msa_mapping,
            msa_dir=args.msa_dir,
            max_msa_depth=args.max_msa_depth,
            require_msa=args.require_msa,
        )
        if features is None:
            error_count += 1
            continue

        # 提取 embedding
        try:
            s, z = extract_embedding(model, features, args.device, args.n_cycle)

            # 创建 mask
            is_ligand = features['is_ligand']
            single_mask = torch.ones_like(is_ligand).float()

            # 保存 embedding: s 和 z 存为 float16 bits (uint16)，读取时 reinterpret
            # 路径: bf16 -> float16 (保持形状) -> uint16 bits
            z_out = z.mean(axis=0) if args.compress_z else z
            embedding = {
                's': s.cpu().to(torch.float16).view(torch.uint16).numpy(),
                'z': z_out.cpu().to(torch.float16).view(torch.uint16).numpy(),
                'is_ligand': is_ligand.float().cpu().numpy(),
                'single_mask': single_mask.cpu().numpy(),
            }
            # Save as compressed npz
            np.savez_compressed(output_path, **embedding)

            success_count += 1

        except Exception as e:
            print(f"Error extracting embedding for {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print(f"\nExtraction complete!")
    print(f"  Success: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Output directory: {args.output_dir}")

    if error_count:
        raise RuntimeError(
            f"Pairformer extraction failed for {error_count}/{len(json_files)} input JSON files"
        )

    if args.compress_z and not args.skip_post_compress:
        print("\nStarting z compression...")
        compress_embeddings(args.output_dir, args.compress_dst_dir, args.compress_workers)

def _arr_to_fp16_bits(arr: np.ndarray) -> np.ndarray:
    """Convert array to float16 bits (uint16) for storage."""
    arr_t = torch.from_numpy(arr)
    if arr.dtype == np.uint16:
        # uint16 bits -> float16
        arr_t = arr_t.view(torch.float16)
    else:
        arr_t = arr_t.to(torch.float16)
    return arr_t.view(torch.uint16).numpy()


def compress_npz(src_path: str, dst_path: str) -> tuple:
    """Compress a single NPZ file: z.mean(axis=0), s/z saved as float16 bits."""
    if os.path.exists(dst_path):
        return src_path, True, None
    try:
        data = np.load(src_path, allow_pickle=True)
        save_dict = {}
        for key in data.files:
            arr = data[key]
            if key == 'z':
                arr_t = torch.from_numpy(arr)
                if arr.dtype == np.uint16:
                    arr_t = arr_t.view(torch.float16)
                else:
                    arr_t = arr_t.to(torch.float16)
                # (n, n, d) -> (n, d) if 3D, skip if already compressed
                if arr_t.ndim == 3:
                    arr_t = arr_t.mean(axis=0)
                save_dict[key] = arr_t.view(torch.uint16).numpy()
            elif key in ('s',):
                save_dict[key] = _arr_to_fp16_bits(arr)
            else:
                save_dict[key] = arr
        np.savez_compressed(dst_path.replace('.npz', ''), **save_dict)
        return src_path, True, None
    except Exception as e:
        return src_path, False, str(e)


def _compress_worker(task):
    """Module-level worker for multiprocessing (must be picklable)."""
    return compress_npz(task[0], task[1])


def compress_embeddings(src_dir: str, dst_dir: str | None, num_workers: int):
    """Compress all NPZ files in src_dir with parallel workers."""
    if dst_dir is None:
        dst_dir = src_dir
    os.makedirs(dst_dir, exist_ok=True)

    files = [f for f in os.listdir(src_dir) if f.endswith('.npz')]
    print(f'Compression: found {len(files)} NPZ files in {src_dir}')

    success, failed = 0, 0
    tasks = []
    for f in files:
        src_path = os.path.join(src_dir, f)
        dst_path = os.path.join(dst_dir, f)
        tasks.append((src_path, dst_path))

    from multiprocessing import Pool

    with Pool(num_workers) as pool:
        for src_path, ok, err in tqdm(
            pool.imap_unordered(_compress_worker, tasks),
            total=len(tasks),
            desc='Compressing z',
        ):
            if ok:
                success += 1
            else:
                failed += 1
                print(f'  FAILED: {src_path} -> {err}')

    print(f'\nCompression complete! Success: {success}, Failed: {failed}')


def set_deterministic(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✓ Set deterministic mode with seed={seed}")


def main():
    set_deterministic(42)
    args = parse_args()
    process_json_files(args)


if __name__ == '__main__':
    main()
