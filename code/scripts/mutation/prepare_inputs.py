#!/usr/bin/env python3
"""Build the final mutation feature views from unlabelled WT/mutant prepared pairs."""
import argparse,json,re,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/scripts/eval'))
from alphaenzyme.data.external_feature_bank import load_external_feature_bank
from evaluate_ranking_single import load_model

def decoded(x):
    return np.asarray(x.view(np.float16) if x.dtype==np.uint16 else x,dtype=np.float32)

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variants',type=Path,required=True,help='CSV: id,DMS_id,mutant,feature_id')
    p.add_argument('--references',type=Path,required=True,help='CSV: DMS_id,sequence,feature_id; one WT per assay')
    p.add_argument('--prepared-pairs',type=Path,required=True,help='Output of prepare_pairs.py --task mutation, containing WT and mutant pairs')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=pd.read_csv(a.variants,dtype=str);refs=pd.read_csv(a.references,dtype=str)
    if not {'id','DMS_id','mutant','feature_id'}<=set(rows) or not {'DMS_id','sequence','feature_id'}<=set(refs):raise ValueError('Missing mutation mapping columns')
    if not rows.id.is_unique or not refs.DMS_id.is_unique or rows.empty or rows[['id','DMS_id','mutant','feature_id']].isna().any().any():raise ValueError('Invalid or duplicate mapping IDs')
    if not set(rows.DMS_id)<=set(refs.DMS_id):raise ValueError('Missing WT reference')
    prepared_rows=pd.read_csv(a.prepared_pairs/'input_table.csv',dtype={'id':str}).set_index('id')
    bank=load_external_feature_bank(str(a.prepared_pairs/'protein_pocket.npz'),include_z=True)
    cfg=yaml.safe_load((ROOT/'configs/specificity_model.yaml').read_text())
    model=load_model(str(ROOT/'data/checkpoints/specificity/specificity_final.pt'),a.device,use_drfp=True,compressed=True,embedding_dir=str(a.prepared_pairs/'pairformer'),resolved_config=cfg)
    def get(feature_id,length):
        if '/' in feature_id or '\\' in feature_id:raise ValueError('Unsafe feature ID')
        path=a.prepared_pairs/'pairformer'/(feature_id+'.npz')
        with np.load(path) as z:
            s=decoded(z['s']);pair=decoded(z['z']);lig=z['is_ligand'].astype(np.float32);mask=z['single_mask'].astype(np.float32)
        protein=(lig<.5)&(mask>.5)
        if protein.sum()!=length:raise ValueError(f'Protein token count does not match full sequence for {feature_id}')
        token=np.concatenate([s[protein],pair[protein]],axis=1)
        raw=bank.get(feature_id)
        if raw is None:raise ValueError('Missing protein-pocket feature: '+feature_id)
        t=lambda x:torch.from_numpy(x).unsqueeze(0).to(a.device)
        result=model(t(s),t(pair),t(lig),t(mask),drfp=torch.zeros((1,2048),device=a.device),extra_features=t(raw.astype(np.float32)))
        alpha=result['alpha_repr'][0].cpu().numpy().astype(np.float16).astype(np.float32)
        return alpha,raw[:1152].astype(np.float32),token
    reference={}
    for row in refs.itertuples(index=False):
        if prepared_rows.loc[row.feature_id,'sequence']!=row.sequence:raise ValueError('WT mapping sequence differs from prepared pair')
        reference[row.DMS_id]=(row.sequence,*get(row.feature_id,len(row.sequence)))
    original=[];paired=[];metadata=[];site=[];aa='ACDEFGHIKLMNPQRSTVWY'
    for row in rows.itertuples(index=False):
        sequence,wt_alpha,wt_raw,wt_token=reference[row.DMS_id]
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',row.mutant)
        if m is None:raise ValueError('Only single substitutions are supported: '+row.mutant)
        old,pos,new=m.group(1),int(m.group(2)),m.group(3)
        if old not in aa or new not in aa or not 1<=pos<=len(sequence) or sequence[pos-1]!=old:raise ValueError('Mutation does not match WT sequence: '+row.mutant)
        expected=sequence[:pos-1]+new+sequence[pos:]
        if prepared_rows.loc[row.feature_id,'sequence']!=expected:raise ValueError('Mutant sequence differs from its substitution')
        alpha,raw,token=get(row.feature_id,len(sequence));delta=raw-wt_raw
        original.append(np.concatenate([alpha,delta]));paired.append(np.concatenate([alpha-wt_alpha,delta]))
        site.append(np.concatenate([wt_token[pos-1],token[pos-1]-wt_token[pos-1]]))
        meta=np.zeros(62,dtype=np.float32);meta[aa.index(old)]=1;meta[20+aa.index(new)]=1
        local=sequence[max(0,pos-4):min(len(sequence),pos+3)]
        for letter in local:
            if letter in aa:meta[40+aa.index(letter)]+=1/len(local)
        meta[60]=pos/len(sequence);meta[61]=np.log1p(len(sequence));metadata.append(meta)
    (a.output/'features').mkdir(parents=True);(a.output/'inputs').mkdir()
    for name,value in [('original',original),('paired',paired),('metadata',metadata),('site_pair',site)]:
        matrix=np.asarray(value,dtype=np.float32)
        if not np.isfinite(matrix).all():raise ValueError('Nonfinite generated features')
        np.save(a.output/'features'/(name+'.npy'),matrix)
    rows[['id','DMS_id','mutant']].to_csv(a.output/'inputs/rows.csv',index=False)
    print(f'Prepared {len(rows)} unlabelled mutation variants in {a.output}')
if __name__=='__main__':main()
