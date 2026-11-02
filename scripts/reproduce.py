#!/usr/bin/env python3
"""Recompute selected paper outputs from released checkpoints and verify them."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['all','specificity','kinetics','mutation'],default='all')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={}
    def run(script,args):subprocess.run([sys.executable,'-B',str(ROOT/script),*map(str,args)],cwd=ROOT,check=True)
    if a.task in ['all','specificity']:
        cfg=json.loads((ROOT/'configs/specificity_evaluation.json').read_text())['release_inputs'];args=[]
        mapping={'embedding_dir':'embedding-dir','json_dir':'json-dir','csv':'csv','drfp':'drfp','protein_pocket_bank':'protein-pocket-bank','train_reactions':'train-reactions'}
        for key,option in mapping.items():args.extend(['--'+option,ROOT/cfg[key]])
        run('scripts/evaluate_specificity.py',args+['--gpu',a.gpu,'--output-dir',out/'specificity'])
        actual=json.loads((out/'specificity/ranking_evaluation_results.json').read_text());expected=json.loads((ROOT/'data/results/specificity/metrics.json').read_text())
        for key in ['Top-10 Success Rate (%)','DCG@10']:
            np.testing.assert_allclose(actual[key],expected[key],rtol=0,atol=1e-12)
        report['specificity']={k:actual[k] for k in ['Raw Samples','Top-10 Success Rate (%)','DCG@10']}
    if a.task in ['all','kinetics']:
        inp=ROOT/'data/inputs/kinetics';path=out/'kinetics.csv'
        run('scripts/predict_kinetics.py',['--csv',inp/'test.csv','--sources',inp/'alpha_sources.json','--substrate-fp',inp/'features/substrate/substrate_fp_ecfp4_2048.pkl','--row-extra',inp/'alpha_test_row_extra.npz','--output',path,'--device',f'cuda:{a.gpu}'])
        pred=pd.read_csv(path,float_precision='round_trip').set_index('id');expected=pd.read_csv(ROOT/'data/results/kinetics/predictions.csv',float_precision='round_trip');report['kinetics']={}
        for task,frame in expected.groupby('task'):
            values=pred.loc[frame.id,'pred_log_'+task].to_numpy();target=frame.prediction.to_numpy()
            np.testing.assert_allclose(values,target,rtol=0,atol=1e-5)
            report['kinetics'][task]={'n':len(frame),'max_abs_error':float(np.max(abs(values-target)))}
    if a.task in ['all','mutation']:
        path=out/'mutation.csv'
        run('code/scripts/mutation/score_library.py',['--input',ROOT/'data/inputs/mutation/component_inputs_unlabelled.csv','--output',path])
        pred=pd.read_csv(path,float_precision='round_trip').set_index('id');expected=pd.read_csv(ROOT/'data/results/mutation/predictions.csv',float_precision='round_trip').set_index('id').loc[pred.index]
        np.testing.assert_allclose(pred.prediction,expected.prediction,rtol=0,atol=1e-12)
        report['mutation']={'n':len(pred),'max_abs_error':float(abs(pred.prediction-expected.prediction).max())}
    (out/'verification.json').write_text(json.dumps({'status':'PASS','tasks':report},indent=2)+'\n')
    print(json.dumps({'status':'PASS','tasks':report},indent=2))
if __name__=='__main__':main()
