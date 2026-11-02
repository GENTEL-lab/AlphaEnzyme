#!/usr/bin/env python
"""Export Protenix-predicted structures for kinetics pocket completion.

This script intentionally reuses the feature-building path from
``precompute_embeddings_protenix.py`` so the protein tokenization and optional
MSA handling stay aligned with the existing AlphaEnzyme feature pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_bundle_root() -> Path:
    """Locate the AlphaEnzyme bundle root from env or by walking upward."""
    env_root = os.environ.get("ALPHAENZYME_BUNDLE_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "code/scripts/data_preprocessing/precompute_embeddings_protenix.py").exists():
            return root
        raise RuntimeError(f"Invalid ALPHAENZYME_BUNDLE_ROOT: {root}")

    marker = Path("code/scripts/data_preprocessing/precompute_embeddings_protenix.py")
    for parent in Path(__file__).resolve().parents:
        if (parent / marker).exists():
            return parent
    raise RuntimeError("Cannot locate AlphaEnzyme bundle root for Protenix structure export")


BUNDLE_ROOT = resolve_bundle_root()
PRECOMPUTE_PATH = BUNDLE_ROOT / "code/scripts/data_preprocessing/precompute_embeddings_protenix.py"


def _ensure_protenix_root_env() -> None:
    """Align Protenix runtime paths before importing its modules."""
    if os.environ.get("PROTENIX_ROOT_DIR"):
        return
    data_root = os.environ.get("PROTENIX_DATA_ROOT_DIR")
    if not data_root:
        return
    data_root_path = Path(data_root).resolve()
    if data_root_path.name == "common":
        candidate = data_root_path.parent
    elif (data_root_path / "common/components.cif").exists():
        candidate = data_root_path
    else:
        candidate = data_root_path.parent
    os.environ["PROTENIX_ROOT_DIR"] = str(candidate)


_ensure_protenix_root_env()


def load_precompute_module():
    spec = importlib.util.spec_from_file_location("alphaenzyme_precompute_protenix", PRECOMPUTE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import precompute module from {PRECOMPUTE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pe = load_precompute_module()

from ml_collections import ConfigDict  # noqa: E402
if os.environ.get("ALPHAENZYME_OFFICIAL_PROTENIX_ROOT"):
    from protenix.data.core.featurizer import Featurizer  # noqa: E402
    from protenix.data.inference.json_parser import add_entity_atom_array  # noqa: E402
    from protenix.data.inference.json_to_feature import SampleDictToFeatures  # noqa: E402
    from protenix.data.tokenizer import AtomArrayTokenizer  # noqa: E402
    from protenix.data.utils import save_structure_cif  # noqa: E402
else:
    raise RuntimeError("Set ALPHAENZYME_OFFICIAL_PROTENIX_ROOT")

def json_to_features_and_atom(
    json_path: Path,
    msa_mapping: dict[str, str] | None,
    msa_dir: str | None,
    max_msa_depth: int,
    require_msa: bool,
) -> tuple[dict[str, torch.Tensor], Any, str]:
    with json_path.open("r", encoding="utf-8") as f:
        sample_dict = json.load(f)

    if isinstance(sample_dict, list):
        if len(sample_dict) != 1:
            raise ValueError(f"{json_path} contains {len(sample_dict)} samples; expected one")
        sample_dict = sample_dict[0]

    sample_name = sample_dict.get("name") or sample_dict.get("sample_id") or json_path.stem
    protein_sequence = None
    for entity in sample_dict.get("sequences", []):
        if "proteinChain" in entity:
            protein_sequence = entity["proteinChain"]["sequence"]
            break

    sample_dict = add_entity_atom_array(sample_dict)
    sample_processor = SampleDictToFeatures(sample_dict)
    atom_array = sample_processor.get_atom_array()

    tokenizer = AtomArrayTokenizer(atom_array)
    token_array = tokenizer.get_token_array()
    featurizer = Featurizer(token_array, atom_array)
    feature_dict = featurizer.get_all_input_features()

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

    centre_atom_indices = [token.centre_atom_index for token in token_array.tokens]
    token_is_ligand = atom_array.is_ligand[centre_atom_indices]
    feature_dict["is_ligand"] = torch.Tensor(token_is_ligand).long()

    n_tokens = feature_dict["restype"].shape[0]
    n_protein_tokens = (~feature_dict["is_ligand"].bool()).sum().item()
    profile, deletion_mean = None, None
    if protein_sequence and msa_mapping is not None:
        profile, deletion_mean = pe.load_msa_for_sequence(
            protein_sequence, msa_mapping, msa_dir, max_msa_depth
        )

    if profile is not None:
        full_profile = torch.zeros(n_tokens, 32)
        full_deletion = torch.zeros(n_tokens, 1)
        n_copy = min(profile.shape[0], n_protein_tokens)
        full_profile[:n_copy] = torch.from_numpy(profile[:n_copy])
        full_deletion[:n_copy] = torch.from_numpy(deletion_mean[:n_copy])
        feature_dict["profile"] = full_profile
        feature_dict["deletion_mean"] = full_deletion
    else:
        if require_msa:
            raise ValueError(f"Required true MSA is missing for {json_path.name}")
        feature_dict["profile"] = torch.zeros(n_tokens, 32)
        feature_dict["deletion_mean"] = torch.zeros(n_tokens, 1)

    return feature_dict, atom_array, sample_name


def add_inference_loss_config(model: torch.nn.Module) -> None:
    """Add the small loss config section needed by Protenix confidence outputs."""
    model.configs.loss = ConfigDict(
        {
            "distogram": {
                "min_bin": 2.3125,
                "max_bin": 21.6875,
                "no_bins": 64,
                "eps": 1e-6,
            },
            "plddt": {"min_bin": 0, "max_bin": 1.0, "no_bins": 50, "normalize": True, "eps": 1e-6},
            "pde": {"min_bin": 0, "max_bin": 32, "no_bins": 64, "eps": 1e-6},
            "pae": {"min_bin": 0, "max_bin": 32, "no_bins": 64, "eps": 1e-6},
            "resolved": {"eps": 1e-6},
        }
    )


def to_device_features(feature_dict: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move one Protenix sample to device.

    Unlike pairformer embedding extraction, diffusion sampling expects the
    atom-local indexing tensors to remain unbatched.
    """
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in feature_dict.items()
    }


@torch.no_grad()
def predict_coordinates(model: torch.nn.Module, input_feature_dict: dict[str, Any]) -> torch.Tensor:
    """Run Protenix inference only up to coordinate sampling.

    The full ``model(..., mode='inference')`` path also computes confidence
    summaries. Those summaries expect the official inference dataloader tensor
    conventions, while pocket completion only needs coordinates for P2Rank.
    """
    input_feature_dict = model.relative_position_encoding.generate_relp(input_feature_dict)
    input_feature_dict = pe.update_input_feature_dict(input_feature_dict)
    s_inputs, s, z = model.get_pairformer_output(
        input_feature_dict=input_feature_dict,
        N_cycle=model.N_cycle,
        inplace_safe=True,
        chunk_size=model.configs.infer_setting.chunk_size,
        mc_dropout=False,
    )

    for key in [
        "msa",
        "has_deletion",
        "deletion_value",
        "profile",
        "deletion_mean",
    ]:
        input_feature_dict.pop(key, None)
    for key in [key for key in list(input_feature_dict) if "template_" in key]:
        input_feature_dict.pop(key, None)

    noise_schedule = model.inference_noise_scheduler(
        N_step=model.configs.sample_diffusion["N_step"],
        device=s.device,
        dtype=s.dtype,
    )
    return model.sample_diffusion(
        denoise_net=model.diffusion_module,
        input_feature_dict=input_feature_dict,
        s_inputs=s_inputs,
        s_trunk=s,
        z_trunk=z,
        pair_z=None,
        p_lm=None,
        c_l=None,
        N_sample=model.configs.sample_diffusion["N_sample"],
        noise_schedule=noise_schedule,
        inplace_safe=True,
        enable_efficient_fusion=model.enable_efficient_fusion,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Protenix CIF structures")
    parser.add_argument("--input_json_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model_name", type=str, default="protenix_base_20250630_v1.0.0")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n_cycle", type=int, default=1)
    parser.add_argument("--n_step", type=int, default=20)
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--msa_mapping", type=Path, default=None)
    parser.add_argument("--msa_dir", type=str, default=None)
    parser.add_argument("--no_msa", action="store_true")
    parser.add_argument("--max_msa_depth", type=int, default=1024)
    parser.add_argument("--require_msa", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def main() -> None:
    args = parse_args()
    set_deterministic(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    msa_mapping = None
    use_msa = (not args.no_msa) and args.msa_mapping is not None
    if use_msa:
        with args.msa_mapping.open("rb") as f:
            msa_data = pickle.load(f)
        msa_mapping = msa_data.get("seq_to_msa", msa_data)

    model, _ = pe.load_model(
        checkpoint_path=str(args.checkpoint),
        model_name=args.model_name,
        device=str(device),
        use_msa=use_msa,
    )
    add_inference_loss_config(model)
    model.N_cycle = args.n_cycle
    model.configs.model.N_cycle = args.n_cycle
    model.configs.sample_diffusion["N_step"] = args.n_step
    model.configs.sample_diffusion["N_sample"] = args.n_sample
    model.configs.infer_setting.chunk_size = None
    model.configs.infer_setting.sample_diffusion_chunk_size = 5

    json_files = sorted(args.input_json_dir.glob("*.json"))
    end_idx = len(json_files) if args.end_idx < 0 else min(args.end_idx, len(json_files))
    json_files = json_files[args.start_idx:end_idx]

    amp_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    enable_amp = device.type == "cuda" and args.dtype != "fp32"

    success, failed = 0, []
    for json_path in json_files:
        try:
            feature_dict, atom_array, sample_name = json_to_features_and_atom(
                json_path=json_path,
                msa_mapping=msa_mapping,
                msa_dir=args.msa_dir,
                max_msa_depth=args.max_msa_depth,
                require_msa=args.require_msa,
            )
            sample_dir = args.output_dir / sample_name / "predictions"
            expected_cif = sample_dir / f"{sample_name}_sample_0.cif"
            if expected_cif.exists():
                success += 1
                print(f"[SKIP] {sample_name}: existing {expected_cif}", flush=True)
                continue

            input_features = to_device_features(feature_dict, device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=enable_amp):
                    coordinates = predict_coordinates(model, input_features)

            sample_dir.mkdir(parents=True, exist_ok=True)
            coords = coordinates.detach().cpu()
            for idx in range(coords.shape[0]):
                out_cif = sample_dir / f"{sample_name}_sample_{idx}.cif"
                save_structure_cif(
                    atom_array=atom_array,
                    pred_coordinate=coords[idx],
                    output_fpath=str(out_cif),
                    entity_poly_type={"1": "polypeptide(L)"},
                    pdb_id=sample_name,
                )
            success += 1
            print(f"[OK] {sample_name}: saved {coords.shape[0]} CIF(s)", flush=True)
        except Exception as exc:  # keep batch robust
            tb = traceback.format_exc()
            failed.append({"json": str(json_path), "error": repr(exc), "traceback": tb})
            print(f"[FAIL] {json_path.name}: {exc}\n{tb}", flush=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    report = {"success": success, "failed": failed, "total": len(json_files)}
    (args.output_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
