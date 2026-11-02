#!/usr/bin/env python3
"""Retrain the final specificity head on the frozen released representations."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--epochs',type=int)
    p.add_argument('--steps-per-epoch',type=int)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads((ROOT/'configs/specificity_training.json').read_text())
    cfg.pop('base_checkpoint_and_representation_inputs_bundled', None)
    if a.epochs is not None:cfg['epochs']=a.epochs
    if a.steps_per_epoch is not None:cfg['steps_per_epoch']=a.steps_per_epoch
    cmd=[sys.executable,'-B',str(ROOT/'code/scripts/specificity/train_head.py'),
        '--base_checkpoint',str(ROOT/'data/checkpoints/specificity/specificity_final.pt'),
        '--repr_cache_list',str(ROOT/'data/inputs/specificity/features/alpha_repr_cache/train_parts.list'),
        '--output_dir',str(a.output.resolve()),'--gpu',str(a.gpu)]
    for key,value in cfg.items():
        if value is True:cmd.append('--'+key)
        elif value is not False:cmd.extend(['--'+key,str(value)])
    subprocess.run(cmd,check=True)
if __name__=='__main__':main()
