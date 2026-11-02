#!/usr/bin/env python3
"""Apply the final fixed three-seed ensemble to externally supplied prepared inputs."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code/scripts/kinetics'))
from evaluate_kinetics_checkpoint import load_checkpoint, load_source_specs_json, predict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,required=True)
    parser.add_argument('--sources',type=Path,required=True,help='Feature-source JSON; relative feature paths resolve beside the JSON file')
    parser.add_argument('--substrate-fp',type=Path,required=True)
    parser.add_argument('--row-extra',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=128)
    args=parser.parse_args()
    for path in [args.csv,args.sources,args.substrate_fp,args.row_extra]:
        if not path.is_file():raise FileNotFoundError(path)
    if args.output.exists():raise FileExistsError(args.output)
    expected=pd.read_csv(args.csv,dtype={'id':str})
    if expected.id.isna().any() or not expected.id.is_unique:raise ValueError('Input IDs must be nonempty and unique')
    sources=load_source_specs_json(args.sources)
    source_name,feature=next(iter(sources.items()))
    models=json.loads((ROOT/'configs/models.json').read_text())['checkpoints']
    frames=[]
    for row in models:
        if row['task']!='kinetics':continue
        parameters=SimpleNamespace(checkpoint=ROOT/row['file'],csv=args.csv,
            embedding_dir=Path(feature['embedding_dir']),drfp_path=Path(feature['drfp_path']),
            bank=Path(feature['bank_path']),substrate_fp_path=args.substrate_fp,
            row_extra_features_npz=args.row_extra,aux_row_extra_features_npz=None,
            source_specs_json=args.sources,source_column='feature_source',source_name=source_name,
            batch_size=args.batch_size,num_workers=0,device=args.device,dropout=0.05,
            kin_head_hidden_size=1024,kin_head_layers=4,kin_head_mode='task_moe_gate',
            ratio_prediction_mode='independent',zero_drfp_input=True,disable_posthoc_raw=True)
        payload,meta,stats=load_checkpoint(parameters.checkpoint)
        frame,_=predict(parameters,payload,meta,stats)
        if not frame.id.is_unique:raise ValueError('Prediction IDs must be unique')
        if set(frame.id.astype(str))!=set(expected.id):raise ValueError('Prediction IDs do not cover the input table')
        frames.append(frame.set_index('id'))
    first=frames[0]
    columns=['pred_log_kcat','pred_log_km']
    for frame in frames:
        if set(frame.index)!=set(first.index):raise ValueError('Seed prediction IDs differ')
    mean=np.mean([f.loc[first.index,columns].to_numpy(float) for f in frames],axis=0)
    if not np.isfinite(mean).all():raise ValueError('Nonfinite prediction')
    output=pd.DataFrame(mean,index=first.index,columns=columns)
    output['model']='AlphaEnzyme'
    output['n_seeds']=3
    args.output.parent.mkdir(parents=True,exist_ok=True)
    output.reset_index().to_csv(args.output,index=False,mode='x')
    print(f'{len(output)} final three-seed predictions saved to {args.output}')


if __name__=='__main__':main()
