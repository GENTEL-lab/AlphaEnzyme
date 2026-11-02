#!/usr/bin/env python3
"""Prepare kinetics row descriptors and substrate fingerprints without reading labels."""
import argparse,json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from descriptors import _features_for_smiles,_names,_sequence_feature_matrix,_morgan_bits
ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stats',type=Path,default=ROOT/'data/inputs/kinetics/alpha_training_zscore_stats.npz')
    p.add_argument('--fit-training-stats',action='store_true',help='Fit on this training CSV only; reuse the saved stats for validation/test')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    frame=pd.read_csv(a.csv,usecols=['id','smiles','sequence'],dtype=str)
    if frame.empty or frame.isna().any().any() or not frame.id.is_unique:raise ValueError('Nonempty unique id, smiles, sequence required')
    template=np.load(a.stats,allow_pickle=True);names=template['names'].astype(str).tolist();lookup={n:i for i,n in enumerate(names)}
    chem={};fingerprints={}
    for smiles in frame.smiles.unique():
        if any(Chem.MolFromSmiles(s) is None for s in smiles.split('.')):raise ValueError('Invalid SMILES: '+smiles)
        chem[smiles]=_features_for_smiles((smiles,2048))
        fingerprints[smiles]=_morgan_bits(Chem.MolFromSmiles(smiles),2,2048).astype(np.float16)
    chemical_names=_names('substrate_smiles',2048)
    sequence,sequence_names=_sequence_feature_matrix(frame,use_dipeptides=True)
    if any(n not in lookup for n in chemical_names+sequence_names):raise ValueError('RDKit/descriptor vocabulary differs from the released model')
    x=np.zeros((len(frame),len(names)),dtype=np.float32)
    x[:,[lookup[n] for n in chemical_names]]=np.stack([chem[s] for s in frame.smiles])
    x[:,[lookup[n] for n in sequence_names]]=sequence
    if a.fit_training_stats:
        mean=x.mean(0,dtype=np.float64).astype(np.float32);std=x.std(0,dtype=np.float64).astype(np.float32);std=np.where(std<1e-6,1.,std)
    else:mean=template['mean'];std=template['std']
    z=np.clip(np.nan_to_num((x-mean)/std,nan=0,posinf=20,neginf=-20),-20,20).astype(np.float16)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'row_extra.npz',ids=frame.id.to_numpy(str),features=z,mask=np.ones(len(frame),dtype=np.float32),names=np.asarray(names,dtype=str))
    np.savez_compressed(a.output/'training_stats.npz',mean=mean,std=std,names=np.asarray(names,dtype=str))
    with (a.output/'substrate_fingerprints.pkl').open('wb') as f:pickle.dump(fingerprints,f,protocol=4)
    (a.output/'preparation.json').write_text(json.dumps({'n':len(frame),'dimension':len(names),'fit_training_stats':a.fit_training_stats,'uses_labels':False},indent=2)+'\n')
    print(f'Prepared {len(frame)} rows in {a.output}')
if __name__=='__main__':main()
