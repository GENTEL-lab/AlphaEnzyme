#!/usr/bin/env python3
"""Score unlabelled prepared pairs with the final specificity head and DRFP gate."""
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch,yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/scripts/eval'))
from evaluate_ranking_single import EmbeddingDataset,load_model,load_specificity_alpha_head,build_similarity_gated_drfp_dict,set_eval_reproducibility

@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['csv','embedding-dir','json-dir','drfp','protein-pocket-bank','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-drfp',type=Path,default=ROOT/'data/inputs/specificity/features/drfp/reaction_fingerprints.pkl')
    p.add_argument('--train-reactions',type=Path,default=ROOT/'data/inputs/specificity/tables/train.csv')
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    frame=pd.read_csv(a.csv,dtype={'id':str})
    if frame.empty or frame.id.isna().any() or not frame.id.is_unique:raise ValueError('Nonempty unique IDs are required')
    if set(frame.id)!={p.stem for p in a.embedding_dir.glob('*.npz')}:raise ValueError('CSV and Pairformer IDs must match exactly')
    with a.drfp.open('rb') as f:raw=pickle.load(f)
    if set(frame.CANO_RXN_SMILES)-set(raw):raise ValueError('Missing reaction fingerprints')
    cfg=json.loads((ROOT/'configs/specificity_evaluation.json').read_text());gate=cfg['drfp_similarity_gate']
    # The gate needs training reference reactions even when the query file is new.
    with a.reference_drfp.open('rb') as f:reference=pickle.load(f)
    reference.update(raw)
    drfp,_=build_similarity_gated_drfp_dict(reference,[str(a.train_reactions)],str(a.csv),gate['low'],gate['high'],gate['power'])
    dataset=EmbeddingDataset(str(a.embedding_dir),str(a.json_dir),drfp,str(a.csv),external_feature_banks=[{'name':'protein_pocket','path':str(a.protein_pocket_bank),'include_z':True}],external_missing_policy='error')
    set_eval_reproducibility(42,True)
    checkpoint=ROOT/cfg['checkpoint'];resolved=yaml.safe_load((ROOT/'configs/specificity_model.yaml').read_text())
    model=load_model(str(checkpoint),a.device,use_drfp=True,compressed=True,embedding_dir=str(a.embedding_dir),resolved_config=resolved)
    head,scale=load_specificity_alpha_head(str(checkpoint),a.device)
    scores={}
    for i in range(len(dataset)):
        item=dataset[i]
        t=lambda k:item[k].unsqueeze(0).to(a.device)
        result=model(t('s'),t('z'),t('is_ligand'),t('single_mask'),drfp=t('drfp'),extra_features=t('extra_features'))
        value=torch.sigmoid(head(result['alpha_repr'])+scale*result['logits']).item()
        if not np.isfinite(value):raise ValueError('Nonfinite prediction')
        scores[item['sample_id']]=value
    output=pd.DataFrame({'id':frame.id,'score':[scores[i] for i in frame.id]})
    a.output.parent.mkdir(parents=True,exist_ok=True);output.to_csv(a.output,index=False,mode='x')
    print(f'{len(output)} predictions saved to {a.output}')
if __name__=='__main__':main()
