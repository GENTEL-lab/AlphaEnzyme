#!/usr/bin/env python3
"""Prepare new canonical enzyme/reaction pairs with explicit upstream runtimes."""
from __future__ import annotations
import argparse,hashlib,json,os,pickle,subprocess,sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))
def run_command(cmd,env=None):
    subprocess.run(cmd,env=env,cwd=ROOT,check=True)
def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

def build_runtime_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("MAX_JOBS", "1")
    env["ALPHAENZYME_BUNDLE_ROOT"] = str(ROOT)
    env["ALPHAENZYME_OFFICIAL_PROTENIX_ROOT"] = str(args.protenix_root)
    protenix_data_root = Path(args.protenix_data_root)
    protenix_common_root = (
        protenix_data_root if protenix_data_root.name == "common" else protenix_data_root / "common"
    )
    env["PROTENIX_ROOT_DIR"] = str(
        protenix_data_root.parent if protenix_data_root.name == "common" else protenix_data_root
    )
    env["PROTENIX_DATA_ROOT_DIR"] = str(protenix_common_root)
    return env

def stable_hash(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]

def seq_hash(sequence: str) -> str:
    return hashlib.sha1(sequence.encode("utf-8")).hexdigest()[:16]

def extract_left_side_ligands(reaction_smiles: str) -> list[str]:
    reaction_smiles = str(reaction_smiles).strip()
    if not reaction_smiles:
        return []
    left = reaction_smiles.split(">>", 1)[0].strip() if ">>" in reaction_smiles else reaction_smiles
    ligands = []
    for ligand in left.split("."):
        ligand = ligand.strip()
        if not ligand or ligand in {"*", "[*]"}:
            continue
        ligands.append(ligand.replace("*", "C"))
    return ligands

def _first_present(row: pd.Series, columns: list[str]) -> str | None:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if pd.isna(value):
            continue
        value = str(value).strip()
        if value and value.lower() not in {"nan", "none"}:
            return value
    return None

def write_protenix_json_and_tables(df: pd.DataFrame, work_dir: Path) -> dict[str, Path]:
    json_dir = work_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for _, row in df.iterrows():
        sample_id = str(row["_sample_id"])
        raw_reaction = str(row["_ae_raw_reaction_smiles"])
        cano_reaction = str(row["_ae_cano_rxn_smiles"])
        sequence = str(row["sequence"])
        ligands = extract_left_side_ligands(raw_reaction)
        if not ligands:
            raise ValueError(f"sample_id={sample_id} has no usable substrate ligand in reaction_smiles")

        sequences = [{"proteinChain": {"sequence": sequence, "count": 1}}]
        sequences.extend({"ligand": {"ligand": ligand, "count": 1}} for ligand in ligands)
        payload = {
            "name": sample_id,
            "sample_id": sample_id,
            "sequences": sequences,
            "label": int(row["Label"]) if "Label" in row.index and pd.notna(row.get("Label")) else 0,
            "ec_number": _first_present(row, ["EC number", "ec_number"]),
            "uniprot_id": _first_present(row, ["UniprotID", "enzyme", "uniprot_id"]) or f"infer_{sample_id}",
            "rhea_id": _first_present(row, ["RHEA_ID", "rhea_id"]) or "",
            "cano_rxn_smiles": cano_reaction,
        }
        (json_dir / f"{sample_id}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        rows.append(
            {
                "id": sample_id,
                "UniprotID": payload["uniprot_id"],
                "CANO_RXN_SMILES": cano_reaction,
                "SMILES": raw_reaction,
                "sequence": sequence,
                "Label": payload["label"],
                "pocket_task_id": sample_id,
            }
        )

    table_csv = work_dir / "input_table.csv"
    task_csv = work_dir / "pocket_tasks.csv"
    pd.DataFrame(rows).to_csv(table_csv, index=False)
    pd.DataFrame(rows)[["pocket_task_id", "UniprotID", "sequence"]].to_csv(task_csv, index=False)
    return {"json_dir": json_dir, "table_csv": table_csv, "task_csv": task_csv}

def precompute_pairformer_embeddings(
    json_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    use_msa: bool,
    env: dict[str, str],
) -> None:
    cmd = [
        sys.executable,
        "code/scripts/data_preprocessing/precompute_embeddings_protenix.py",
        "--input_json_dir",
        str(json_dir),
        "--output_dir",
        str(output_dir),
        "--checkpoint",
        str(args.protenix_checkpoint),
        "--compress_z",
        "--compress_workers",
        "1",
        "--device",
        "cuda",
        "--n_cycle",
        str(args.n_cycle),
        "--num_workers",
        str(args.num_workers),
    ]
    if use_msa:
        cmd.extend(["--use_msa", "--require_msa", "--msa_mapping", str(args.msa_mapping), "--msa_dir", str(args.msa_dir)])
    else:
        cmd.append("--no_msa")
    run_command(cmd, env=env)

def _read_float_feature(data: Any, key: str) -> np.ndarray:
    arr = data[key]
    if arr.dtype == np.uint16:
        return arr.view(np.float16).astype(np.float32)
    return arr.astype(np.float32, copy=False)

def pool_pairformer_bank(feature_dir: Path, output_path: Path, include_z: bool = True) -> Path:
    files = sorted(feature_dir.glob("*.npz"), key=lambda path: path.stem)
    if not files:
        raise FileNotFoundError(f"No npz embeddings found under {feature_dir}")

    sample_ids, s_rows, z_rows, enzyme_counts, total_counts = [], [], [], [], []
    for path in files:
        data = np.load(path, allow_pickle=True)
        single_mask = data["single_mask"].astype(np.float32) > 0
        is_ligand = data["is_ligand"].astype(np.float32) > 0
        enzyme_mask = single_mask & (~is_ligand)
        if not np.any(enzyme_mask):
            enzyme_mask = single_mask

        s = _read_float_feature(data, "s")
        s_rows.append((s[enzyme_mask].mean(axis=0) if np.any(enzyme_mask) else s.mean(axis=0)).astype(np.float16))
        if include_z:
            z = _read_float_feature(data, "z")
            z_rows.append((z[enzyme_mask].mean(axis=0) if np.any(enzyme_mask) else z.mean(axis=0)).astype(np.float16))
        sample_ids.append(path.stem)
        enzyme_counts.append(int(np.sum(enzyme_mask)))
        total_counts.append(int(np.sum(single_mask)))

    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "s_mean": np.stack(s_rows).astype(np.float16),
        "enzyme_token_count": np.asarray(enzyme_counts, dtype=np.int32),
        "total_token_count": np.asarray(total_counts, dtype=np.int32),
    }
    if include_z:
        payload["z_mean"] = np.stack(z_rows).astype(np.float16)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "feature_dir": str(feature_dir),
                "num_samples": len(sample_ids),
                "include_z": bool(include_z),
                "s_dim": int(payload["s_mean"].shape[1]),
                "z_dim": int(payload["z_mean"].shape[1]) if include_z else 0,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path

def compute_esmc_nodes(task_csv: Path, node_dir: Path, work_dir: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "code/scripts/data_preprocessing/compute_missing_esmc.py",
        "--task_csv",
        str(task_csv),
        "--node_dir",
        str(node_dir),
        "--partial_seq2feature",
        str(work_dir / "esmc_partial_seq2feature.pkl"),
        "--mapping_csv",
        str(work_dir / "esmc_mapping.csv"),
        "--failed_csv",
        str(work_dir / "esmc_failed.csv"),
        "--device",
        "cuda",
        "--model_name",
        "esmc_600m",
        "--ignore_existing_seq2feature",
    ]
    run_command(cmd, env=env)

def export_structures(json_dir: Path, structure_dir: Path, args: argparse.Namespace, env: dict[str, str]) -> None:
    cmd = [
        sys.executable,
        "code/scripts/data_preprocessing/export_protenix_structures.py",
        "--input_json_dir",
        str(json_dir),
        "--output_dir",
        str(structure_dir),
        "--checkpoint",
        str(args.protenix_checkpoint),
        "--device",
        "cuda",
        "--n_cycle",
        str(args.structure_n_cycle),
        "--n_step",
        str(args.structure_n_step),
        "--n_sample",
        str(args.structure_n_sample),
        "--seed",
        str(args.structure_seed),
        "--dtype",
        "bf16",
        "--no_msa",
    ]
    run_command(cmd, env=env)

def run_p2rank(structure_dir: Path, p2rank_dir: Path, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "code/scripts/data_preprocessing/run_p2rank_for_structures.py",
        "--structure_dir",
        str(structure_dir),
        "--output_dir",
        str(p2rank_dir),
        "--p2rank_bin",
        str(args.p2rank_bin),
        "--java_home",
        str(args.java_home),
        "--workers",
        str(args.p2rank_workers),
        "--threads_per_job",
        "2",
    ]
    run_command(cmd)
    report_path = p2rank_dir / "p2rank_resume_report.json"
    if not report_path.exists():
        raise RuntimeError(f"P2Rank did not write report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report.get("summary", {})
    total = int(summary.get("total", 0))
    ok = int(summary.get("ok", 0))
    skip = int(summary.get("skip", 0))
    fail = int(summary.get("fail", 0))
    if total <= 0 or fail > 0 or ok + skip != total:
        raise RuntimeError(f"P2Rank failed or incomplete: {json.dumps(summary, ensure_ascii=False)}")

def read_top_pocket_indices(residue_csv: Path, sequence_len: int) -> tuple[np.ndarray, str]:
    if not residue_csv.exists():
        return np.asarray([], dtype=np.int64), "missing_p2rank"
    df = pd.read_csv(residue_csv)
    df.columns = [str(c).strip() for c in df.columns]
    if "pocket" not in df.columns or "residue_label" not in df.columns:
        return np.asarray([], dtype=np.int64), "bad_p2rank_columns"
    pocket = pd.to_numeric(df["pocket"], errors="coerce").fillna(0).astype(int)
    top = df[pocket == 1].copy()
    if top.empty:
        return np.asarray([], dtype=np.int64), "no_top_pocket"
    residue_label = pd.to_numeric(top["residue_label"], errors="coerce").dropna().astype(int)
    idx = np.unique(residue_label.to_numpy(dtype=np.int64) - 1)
    idx = idx[(idx >= 0) & (idx < sequence_len)]
    if idx.size == 0:
        return idx, "no_valid_residue_index"
    return idx, "ok"

def build_self_protein_pocket_bank(
    task_csv: Path,
    node_dir: Path,
    p2rank_dir: Path,
    output_path: Path,
    *,
    zero_pocket: bool = False,
) -> Path:
    tasks = pd.read_csv(task_csv)
    sample_ids, protein_rows, pocket_rows, statuses, counts = [], [], [], [], []
    zero = np.zeros((1152,), dtype=np.float32)

    for row in tasks.itertuples(index=False):
        sample_id = str(row.pocket_task_id)
        sequence = str(row.sequence)
        node_path = node_dir / f"{seq_hash(sequence)}.npz"
        if not node_path.exists():
            raise FileNotFoundError(f"ESMC features missing for {sample_id}")
        else:
            node = np.load(node_path)["node_feature"].astype(np.float32)
            if node.shape[0] != len(sequence):
                raise ValueError(f"ESMC sequence length mismatch for {sample_id}")
            else:
                protein = node.mean(axis=0).astype(np.float32)
                if zero_pocket:
                    pocket = zero
                    status = "zero_pocket_debug"
                    idx = np.asarray([], dtype=np.int64)
                else:
                    residue_csv = p2rank_dir / f"{sample_id}_sample_0.cif_residues.csv"
                    idx, status = read_top_pocket_indices(residue_csv, len(sequence))
                    if status in {"missing_p2rank","bad_p2rank_columns","no_valid_residue_index"}:
                        raise ValueError(f"Invalid P2Rank result for {sample_id}: {status}")
                    pocket = node[idx].mean(axis=0).astype(np.float32) if status == "ok" else zero

        sample_ids.append(sample_id)
        protein_rows.append(protein.astype(np.float16))
        pocket_rows.append(pocket.astype(np.float16))
        statuses.append(status)
        counts.append(int(idx.size))

    payload = {
        "sample_ids": np.asarray(sample_ids, dtype=str),
        "s_mean": np.stack(protein_rows).astype(np.float16),
        "z_mean": np.stack(pocket_rows).astype(np.float16),
        "pocket_status": np.asarray(statuses, dtype=str),
        "num_pocket_residues": np.asarray(counts, dtype=np.int32),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)
    status_counts = pd.Series(statuses).value_counts().to_dict()
    output_path.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "task_csv": str(task_csv),
                "node_dir": str(node_dir),
                "p2rank_dir": str(p2rank_dir),
                "num_samples": len(sample_ids),
                "s_dim": int(payload["s_mean"].shape[1]),
                "z_dim": int(payload["z_mean"].shape[1]),
                "status_counts": status_counts,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_path

def build_train_only_context_cache(
    df: pd.DataFrame,
    query_msa_bank: Path,
    drfp_path: Path,
    output_cache: Path,
    args: argparse.Namespace,
) -> Path:
    from scripts.direct.build_direct_context_assets import build_query_cache, load_positive_bank
    from scripts.rerank.retrieval_context_common import get_or_encode_rxn_fp, load_rxn2fp

    rxn2fp = load_rxn2fp(str(drfp_path))
    drfp_rows = [
        get_or_encode_rxn_fp(str(reaction), rxn2fp).astype(np.float32, copy=False)
        for reaction in df["_ae_cano_rxn_smiles"].astype(str)
    ]
    query_table = {
        "sample_ids": df["_sample_id"].astype(str).to_numpy(),
        "reactions": df["_ae_cano_rxn_smiles"].astype(str).to_numpy(),
        "labels": np.zeros((len(df),), dtype=np.int64),
        "drfp_features": np.stack(drfp_rows).astype(np.float32, copy=False),
    }
    positive_bank = load_positive_bank(str(args.train_pos_context_bank))
    cache_payload, reaction_neighbor_cache = build_query_cache(
        query_table=query_table,
        query_external_bank_path=str(query_msa_bank),
        include_query_z=True,
        positive_bank=positive_bank,
        reaction_topk=5,
        reaction_preselect=24,
        top_enzymes_per_reaction=10,
        max_context=50,
        exclude_self_sample=False,
        rxn2fp=rxn2fp,
        num_workers=int(args.retrieval_workers),
        similarity_mode="hybrid_exact",
    )
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_cache, **cache_payload)
    output_cache.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "query_msa_bank": str(query_msa_bank),
                "train_pos_context_bank": str(args.train_pos_context_bank),
                "reference_scope": "train_positive_only",
                "reaction_topk": 5,
                "reaction_preselect": 24,
                "top_enzymes_per_reaction": 10,
                "max_context": 50,
                "similarity_mode": "hybrid_exact",
                "reaction_neighbor_count": len(reaction_neighbor_cache),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return output_cache

def build_mixed_external_bank(
    context_cache: Path,
    self_bank: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> Path:
    cmd = [
        sys.executable,
        "code/scripts/direct/build_self_heavy_specmsa_bank.py",
        "--cache_path",
        str(context_cache),
        "--positive_context_bank_path",
        str(args.train_pos_context_bank),
        "--positive_source_bank_path",
        str(args.train_pos_protein_pocket_bank),
        "--query_self_bank_path",
        str(self_bank),
        "--output_path",
        str(output_path),
        "--self_weight",
        str(args.self_weight),
        "--score_threshold",
        str(args.score_threshold),
        "--context_pool_mode",
        "mean",
        "--fallback_mode",
        "self_only",
    ]
    run_command(cmd)
    return output_path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='CSV: unique id, sequence, CANO_RXN_SMILES; canonical reactions are required')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--task',choices=['specificity','kinetics','mutation'],required=True)
    p.add_argument('--protenix-root',type=Path,default=ROOT/'code/third_party/protenix_runtime')
    for name in ['protenix-data-root','protenix-checkpoint','p2rank-bin','java-home']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--msa-mapping',type=Path)
    p.add_argument('--msa-dir',type=Path)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--p2rank-workers',type=int,default=2)
    p.add_argument('--retrieval-workers',type=int,default=2)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    for name in ['input','protenix_root','protenix_data_root','protenix_checkpoint','p2rank_bin','java_home']:
        value=getattr(a,name).resolve();require_path(value,name);setattr(a,name,value)
    if a.task!='mutation':
        if a.msa_mapping is None or a.msa_dir is None:raise ValueError('The final specificity/kinetics feature recipe requires --msa-mapping and --msa-dir')
        a.msa_mapping=a.msa_mapping.resolve();a.msa_dir=a.msa_dir.resolve()
        require_path(a.msa_mapping,'MSA mapping');require_path(a.msa_dir,'MSA directory')
    df=pd.read_csv(a.input,dtype={'id':str})
    if not {'id','sequence','CANO_RXN_SMILES'}<=set(df):raise ValueError('Expected id, sequence, CANO_RXN_SMILES')
    if df.empty or df[['id','sequence','CANO_RXN_SMILES']].isna().any().any() or not df.id.is_unique:raise ValueError('Invalid input IDs or missing values')
    if any('/' in v or '\\' in v or v in {'.','..'} for v in df.id):raise ValueError('IDs must be safe filenames')
    df['_sample_id']=df.id
    df['_ae_raw_reaction_smiles']=df.CANO_RXN_SMILES
    df['_ae_cano_rxn_smiles']=df.CANO_RXN_SMILES
    a.n_cycle=1;a.structure_n_cycle=1;a.structure_n_step=20;a.structure_n_sample=1;a.structure_seed=42
    a.self_weight=.95;a.score_threshold=.55
    reference=ROOT/'data/inputs/specificity/features/reference'
    a.train_pos_context_bank=reference/'positive_context.npz'
    a.train_pos_protein_pocket_bank=reference/'positive_protein_pocket.npz'
    work=a.output.resolve();work.mkdir(parents=True)
    env=build_runtime_env(a);paths=write_protenix_json_and_tables(df,work)
    from drfp import DrfpEncoder
    # Reference reaction fingerprints are needed for specificity's train-only similarity gate.
    if a.task=='specificity':
        with (ROOT/'data/inputs/specificity/features/drfp/reaction_fingerprints.pkl').open('rb') as f:rxn2fp=pickle.load(f)
    else:rxn2fp={}
    for reaction in df.CANO_RXN_SMILES.unique():
        if reaction not in rxn2fp:rxn2fp[reaction]=np.asarray(DrfpEncoder.encode(reaction,n_folded_length=2048)[0],dtype=np.float32)
    drfp=work/'reaction_fingerprints.pkl'
    with drfp.open('wb') as f:pickle.dump(rxn2fp,f,protocol=4)
    embedding=work/'pairformer'
    precompute_pairformer_embeddings(paths['json_dir'],embedding,a,use_msa=a.task!='mutation',env=env)
    if {p.stem for p in embedding.glob('*.npz')}!=set(df.id):raise RuntimeError('Incomplete Protenix embedding output')
    compute_esmc_nodes(paths['task_csv'],work/'esmc_nodes',work,a,env)
    export_structures(paths['json_dir'],work/'structures',a,env)
    run_p2rank(work/'structures',work/'p2rank',a)
    bank=work/'protein_pocket.npz'
    build_self_protein_pocket_bank(paths['task_csv'],work/'esmc_nodes',work/'p2rank',bank)
    if a.task=='specificity':
        pooled=work/'query_structure_pool.npz';pool_pairformer_bank(embedding,pooled)
        cache=work/'context.npz';build_train_only_context_cache(df,pooled,drfp,cache,a)
        bank=build_mixed_external_bank(cache,bank,work/'protein_pocket_mixed.npz',a)
    name='public_strict_iteration1_fulltrue_msa'
    (work/'sources.json').write_text(json.dumps({name:{'embedding_dir':'pairformer','drfp_path':drfp.name,'bank_path':bank.name}},indent=2)+'\n')
    (work/'preparation.json').write_text(json.dumps({'task':a.task,'rows':len(df),'msa':a.task!='mutation','protein_pocket_bank':bank.name,'uses_target_labels':False,'foundation_model':'protenix_base_20250630_v1.0.0','esmc_model':'esmc_600m'},indent=2)+'\n')
    print(f'Prepared {len(df)} new pairs in {work}')
if __name__=='__main__':main()
