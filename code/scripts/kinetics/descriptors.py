"""Input-only chemical and sequence descriptors used by the final kinetics model."""
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem,Crippen,Descriptors,Lipinski,MACCSkeys,rdMolDescriptors
RDLogger.DisableLog('rdApp.*')
RDKIT_2D_DESCRIPTORS = list(Descriptors._descList)

SIMPLE_NAMES = [
    "molwt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable",
    "rings",
    "heavy_atoms",
    "fraction_csp3",
    "formal_charge",
    "aromatic_rings",
    "hetero_atoms",
]

def _simple_descriptors(mol: Chem.Mol) -> list[float]:
    return [
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        Lipinski.RingCount(mol),
        mol.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(mol),
        Chem.GetFormalCharge(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcNumHeteroatoms(mol),
    ]

def _morgan_bits(mol: Chem.Mol, radius: int, n_bits: int) -> np.ndarray:
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def _maccs_bits(mol: Chem.Mol) -> np.ndarray:
    fp = MACCSkeys.GenMACCSKeys(mol)
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.float32)

def _features_for_smiles(args: tuple[str, int]) -> np.ndarray:
    smiles, n_bits = args
    simple_sum = np.zeros(len(SIMPLE_NAMES), dtype=np.float64)
    rdkit_sum = np.zeros(len(RDKIT_2D_DESCRIPTORS), dtype=np.float64)
    morgan2 = np.zeros(n_bits, dtype=np.float32)
    morgan3 = np.zeros(n_bits, dtype=np.float32)
    maccs = np.zeros(167, dtype=np.float32)
    if not isinstance(smiles, str) or not smiles or smiles.lower() == "nan":
        return np.concatenate([simple_sum, rdkit_sum, morgan2, morgan3, maccs]).astype(np.float32)
    for part in [part for part in smiles.split(".") if part]:
        mol = Chem.MolFromSmiles(part)
        if mol is None:
            continue
        try:
            simple_sum += np.asarray(_simple_descriptors(mol), dtype=np.float64)
        except Exception:
            pass
        values = []
        for _name, func in RDKIT_2D_DESCRIPTORS:
            try:
                value = float(func(mol))
            except Exception:
                value = 0.0
            if not np.isfinite(value):
                value = 0.0
            values.append(value)
        rdkit_sum += np.asarray(values, dtype=np.float64)
        morgan2 = np.maximum(morgan2, _morgan_bits(mol, 2, n_bits))
        morgan3 = np.maximum(morgan3, _morgan_bits(mol, 3, n_bits))
        maccs = np.maximum(maccs, _maccs_bits(mol))
    out = np.concatenate([simple_sum, rdkit_sum, morgan2, morgan3, maccs])
    out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
    out = np.clip(out, -1e6, 1e6)
    return out.astype(np.float32)

def _names(prefix: str, n_bits: int) -> list[str]:
    names = [f"{prefix}_simple_{name}" for name in SIMPLE_NAMES]
    names.extend([f"{prefix}_rdkit2d_{name}" for name, _func in RDKIT_2D_DESCRIPTORS])
    names.extend([f"{prefix}_morgan2_{idx}" for idx in range(n_bits)])
    names.extend([f"{prefix}_morgan3_{idx}" for idx in range(n_bits)])
    names.extend([f"{prefix}_maccs_{idx}" for idx in range(167)])
    return names
AA20 = "ACDEFGHIKLMNPQRSTVWY"

AA_GROUPS = {
    "hydrophobic": "AILMFWYV",
    "polar": "STNQCY",
    "acidic": "DE",
    "basic": "KRH",
    "small": "AGSTP",
    "aromatic": "FWY",
}

def _sequence_feature_matrix(frame: pd.DataFrame, *, use_dipeptides: bool) -> tuple[np.ndarray, list[str]]:
    sequences = frame["sequence"].fillna("").astype(str) if "sequence" in frame.columns else pd.Series([""] * len(frame))
    dipeptides = [a + b for a in AA20 for b in AA20]
    rows = []
    for sequence in sequences:
        seq = "".join([aa for aa in sequence.upper() if aa.isalpha()])
        length = max(len(seq), 1)
        valid = [aa for aa in seq if aa in AA20]
        valid_len = max(len(valid), 1)
        counts = {aa: 0 for aa in AA20}
        for aa in valid:
            counts[aa] += 1
        features = [
            float(len(seq)),
            float(np.log1p(len(seq))),
            float(len(valid) / length),
            float(1.0 - len(valid) / length),
        ]
        features.extend([counts[aa] / valid_len for aa in AA20])
        valid_text = "".join(valid)
        for group in AA_GROUPS.values():
            features.append(sum(counts[aa] for aa in group) / valid_len)
        if use_dipeptides:
            denom = max(len(valid_text) - 1, 1)
            pair_counts = {pair: 0 for pair in dipeptides}
            for idx in range(max(len(valid_text) - 1, 0)):
                pair = valid_text[idx : idx + 2]
                if pair in pair_counts:
                    pair_counts[pair] += 1
            features.extend([pair_counts[pair] / denom for pair in dipeptides])
        rows.append(features)

    names = ["seq_len", "seq_log_len", "seq_valid_frac", "seq_unknown_frac"]
    names.extend([f"aa_frac_{aa}" for aa in AA20])
    names.extend([f"aa_group_frac_{group}" for group in AA_GROUPS])
    if use_dipeptides:
        names.extend([f"dipeptide_frac_{pair}" for pair in dipeptides])
    return np.asarray(rows, dtype=np.float32), names
