import json,os,pickle
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from drfp import DrfpEncoder
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem,DataStructs
RDLogger.DisableLog("rdApp.*")

def l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-8, None)
    return x / denom

def load_external_similarity_bank(bank_path):
    data = np.load(bank_path, allow_pickle=True)
    bank = {
        "sample_ids": data["sample_ids"].astype(str),
        "s_mean": data["s_mean"].astype(np.float16, copy=False),
    }
    if "z_mean" in data:
        bank["z_mean"] = data["z_mean"].astype(np.float16, copy=False)
    return bank

def align_external_similarity_features(sample_ids, external_bank, include_z=False):
    external_ids = external_bank["sample_ids"]
    order = np.argsort(external_ids)
    sorted_ids = external_ids[order]
    target = np.asarray(sample_ids).astype(str)
    pos = np.searchsorted(sorted_ids, target)
    valid = (pos < len(sorted_ids)) & (sorted_ids[np.clip(pos, 0, len(sorted_ids) - 1)] == target)
    if not np.all(valid):
        missing = target[~valid][:10].tolist()
        raise KeyError(f"Missing {np.sum(~valid)} sample_ids in external similarity bank, examples: {missing}")
    aligned_idx = order[pos]
    features = [external_bank["s_mean"][aligned_idx].astype(np.float16, copy=False)]
    if include_z:
        if "z_mean" not in external_bank:
            raise KeyError("include_z=True but z_mean not found in external similarity bank")
        features.append(external_bank["z_mean"][aligned_idx].astype(np.float16, copy=False))
    return np.concatenate(features, axis=-1).astype(np.float16, copy=False)

def load_rxn2fp(drfp_path):
    with open(drfp_path, "rb") as f:
        return pickle.load(f)

def get_or_encode_rxn_fp(rxn, rxn2fp):
    fp = rxn2fp.get(rxn)
    if fp is not None:
        return np.asarray(fp, dtype=np.float32)
    encoded = DrfpEncoder.encode([rxn])[0]
    rxn2fp[rxn] = encoded
    return np.asarray(encoded, dtype=np.float32)

def get_rdkit_mol(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    return mol

def neutralize_atoms(smi):
    mol = get_rdkit_mol(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()

    smi_uncharged = Chem.MolToSmiles(mol)
    if not get_rdkit_mol(smi_uncharged):
        smi_uncharged = smi
    return smi_uncharged

def get_morgan_fp(mol):
    mol = get_rdkit_mol(mol)
    Chem.SanitizeMol(mol)
    info1 = {}
    fp_m = AllChem.GetMorganFingerprint(
        mol,
        8,
        bitInfo=info1,
        invariants=AllChem.GetConnectivityInvariants(mol, includeRingMembership=False),
    )
    return fp_m

def bulk_tani(target_fp, fps, rxn_ids):
    similarity_list = DataStructs.BulkTanimotoSimilarity(target_fp, list(fps))
    return {rxn_ids[i]: float(similarity_list[i]) for i in range(len(similarity_list))}

def get_rsim(rcts_counter, pros_counter, cand_rcts_counter, cand_pros_counter, sim):
    info_dict = {
        "s1": rcts_counter,
        "p1": pros_counter,
        "s2": cand_rcts_counter,
        "p2": cand_pros_counter,
    }
    ss = {}
    simm = {}
    pairs = [("s1", "s2"), ("s1", "p2"), ("p1", "s2"), ("p1", "p2")]

    for pair_tuple in pairs:
        pairings = set()
        simm[pair_tuple] = {}
        for mol_x in info_dict[pair_tuple[0]].keys():
            simm[pair_tuple][mol_x] = (0.0, mol_x, None)
            if mol_x in sim:
                for mol_y in info_dict[pair_tuple[1]].keys():
                    if mol_y in sim[mol_x]:
                        pairings.add((sim[mol_x][mol_y], mol_x, mol_y))
        found = {"left": set(), "right": set()}
        for v in sorted(pairings, key=lambda h: -h[0]):
            if v[1] not in found["left"] and v[2] not in found["right"]:
                if v[0] > simm[pair_tuple][v[1]][0]:
                    simm[pair_tuple][v[1]] = v
                    found["left"].add(v[1])
                    found["right"].add(v[2])
        scores = [simm[pair_tuple][mol_x][0] for mol_x in simm[pair_tuple]]
        ss[pair_tuple] = sum(scores) / len(scores) if scores else 0.0

    s1 = float(np.sqrt(ss[pairs[0]] ** 2 + ss[pairs[3]] ** 2) / np.sqrt(2.0))
    s2 = float(np.sqrt(ss[pairs[1]] ** 2 + ss[pairs[2]] ** 2) / np.sqrt(2.0))
    return max(s1, s2)

def get_mol_similarity_dict(query_rxn, candidate_rxns):
    query_smiles = list(
        set(query_rxn.split(">>")[0].split(".") + query_rxn.split(">>")[1].split("."))
    )
    cand_smiles = []
    for rxn in candidate_rxns:
        cand_smiles.extend(rxn.split(">>")[0].split("."))
        cand_smiles.extend(rxn.split(">>")[1].split("."))
    cand_smiles = list(set(cand_smiles))

    cand_mol_ids = [f"MOL_{i}" for i in range(len(cand_smiles))]
    cand_mol_to_id = dict(zip(cand_smiles, cand_mol_ids))
    cand_fps = []
    for smi in cand_smiles:
        smi = neutralize_atoms(smi)
        cand_fps.append(get_morgan_fp(smi))

    cpd_simi_dict = {}
    for smi in query_smiles:
        fp_target = get_morgan_fp(neutralize_atoms(smi))
        cpd_simi_dict[smi] = bulk_tani(fp_target, cand_fps, cand_mol_ids)
    return cpd_simi_dict, cand_mol_to_id

def exact_reaction_similarity(query_rxn, candidate_rxns):
    if not candidate_rxns:
        return {}
    cpd_simi_dict, cand_mol_to_id = get_mol_similarity_dict(query_rxn, candidate_rxns)
    rcts_counter = defaultdict(int)
    pros_counter = defaultdict(int)
    for smi in query_rxn.split(">>")[0].split("."):
        rcts_counter[smi] += 1
    for smi in query_rxn.split(">>")[1].split("."):
        pros_counter[smi] += 1

    result = {}
    for cand_rxn in candidate_rxns:
        cand_rcts_counter = defaultdict(int)
        cand_pros_counter = defaultdict(int)
        for smi in cand_rxn.split(">>")[0].split("."):
            cand_rcts_counter[cand_mol_to_id.get(smi)] += 1
        for smi in cand_rxn.split(">>")[1].split("."):
            cand_pros_counter[cand_mol_to_id.get(smi)] += 1
        result[cand_rxn] = get_rsim(
            rcts_counter,
            pros_counter,
            cand_rcts_counter,
            cand_pros_counter,
            cpd_simi_dict,
        )
    return result

def _reaction_neighbor_worker(task):
    query_rxn, candidate_rxns, topk = task
    exact = exact_reaction_similarity(query_rxn, candidate_rxns)
    ranked = sorted(exact.items(), key=lambda x: x[1], reverse=True)[:topk]
    return query_rxn, ranked

def build_reaction_neighbor_cache(
    query_reactions,
    train_reactions,
    rxn2fp,
    topk=5,
    preselect=24,
    exclude_self=False,
    num_workers=0,
    similarity_mode="hybrid_exact",
):
    train_reactions = [r for r in train_reactions]
    query_reactions = [r for r in query_reactions]
    train_reaction_arr = np.asarray(train_reactions, dtype=str)
    train_fp_mat = l2_normalize(
        np.asarray([get_or_encode_rxn_fp(r, rxn2fp) for r in train_reactions], dtype=np.float32)
    )
    train_idx_map = {r: i for i, r in enumerate(train_reactions)}

    cache = {}
    tasks = []
    for query_rxn in query_reactions:
        query_fp = l2_normalize(get_or_encode_rxn_fp(query_rxn, rxn2fp)[None, :])[0]
        coarse_scores = train_fp_mat @ query_fp
        if exclude_self and query_rxn in train_idx_map:
            coarse_scores[train_idx_map[query_rxn]] = -1.0
        topn = min(int(preselect), len(train_reaction_arr))
        coarse_idx = np.argpartition(coarse_scores, -topn)[-topn:]
        coarse_idx = coarse_idx[np.argsort(coarse_scores[coarse_idx])[::-1]]
        if str(similarity_mode).lower() == "drfp":
            cache_topk = min(int(topk), coarse_idx.shape[0])
            cache[query_rxn] = [
                (str(train_reaction_arr[idx]), float(coarse_scores[idx]))
                for idx in coarse_idx[:cache_topk]
            ]
            continue
        candidate_rxns = train_reaction_arr[coarse_idx].tolist()
        tasks.append((query_rxn, candidate_rxns, topk))

    if int(num_workers) > 1:
        with Pool(processes=int(num_workers)) as pool:
            for query_rxn, ranked in pool.imap_unordered(_reaction_neighbor_worker, tasks):
                cache[query_rxn] = ranked
    else:
        for task in tasks:
            query_rxn, ranked = _reaction_neighbor_worker(task)
            cache[query_rxn] = ranked
    return cache
