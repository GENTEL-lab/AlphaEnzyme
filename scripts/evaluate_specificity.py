#!/usr/bin/env python3
"""Evaluate prepared specificity inputs with the final checkpoint and paper settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['embedding-dir','json-dir','csv','drfp','protein-pocket-bank','train-reactions','output-dir']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    paths=[args.embedding_dir,args.json_dir,args.csv,args.drfp,args.protein_pocket_bank,args.train_reactions]
    for path in paths:
        if not path.exists():raise FileNotFoundError(path)
    if args.output_dir.exists():raise FileExistsError('Choose a new output directory: '+str(args.output_dir))
    config=json.loads((ROOT/'configs/specificity_evaluation.json').read_text())
    gate=config['drfp_similarity_gate']
    command=[sys.executable,'-B',str(ROOT/'code/scripts/eval/evaluate_ranking_single.py'),
        '--checkpoint',str(ROOT/config['checkpoint']),
        '--config_path',str(ROOT/'configs/specificity_model.yaml'),'--external_include_z', '--embedding_dir',str(args.embedding_dir.resolve()),
        '--json_dir',str(args.json_dir.resolve()),'--csv_path',str(args.csv.resolve()),
        '--drfp_path',str(args.drfp.resolve()),'--external_feature_bank',str(args.protein_pocket_bank.resolve()),
        '--drfp_similarity_gate_train_csv',str(args.train_reactions.resolve()),
        '--drfp_similarity_gate_low',str(gate['low']),'--drfp_similarity_gate_high',str(gate['high']),
        '--drfp_similarity_gate_power',str(gate['power']),'--use_alpha_repr_specificity_head',
        '--compressed','--use_drfp','--deduplicate_reaction_enzymes','--eval_precision','fp32',
        '--seed','42','--gpu',str(args.gpu),'--score_dir',str(args.output_dir.resolve()/'scores'),
        '--output_dir',str(args.output_dir.resolve())]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
