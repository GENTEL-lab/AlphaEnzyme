"""Train the released kinetics architecture with validation-based checkpoint selection."""
import argparse
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
import torch
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
SEEDS = [1234, 2026, 3407]

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + chr(10))


class ValidationEarlyStop(Exception):
    pass

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,choices=SEEDS,required=True)
    parser.add_argument('--inputs',type=Path,required=True,help='External directory with train.csv, valid.csv and alpha_{train,valid}_row_extra.npz')
    parser.add_argument('--sources',type=Path,required=True,help='External feature-source JSON with embedding_dir, drfp_path and bank_path')
    parser.add_argument('--substrate-fp',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--architecture-checkpoint',type=Path)
    parser.add_argument('--epochs',type=int,default=None,
                        help='Override the final schedule for a smoke test or controlled rerun.')
    parser.add_argument('--steps-per-epoch',type=int,default=None,
                        help='Override steps per epoch for a smoke test.')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():
        raise FileExistsError('Use a new training output directory: '+str(out))
    for name in ['train.csv','valid.csv','alpha_train_row_extra.npz','alpha_valid_row_extra.npz']:
        if not (args.inputs/name).is_file():
            raise FileNotFoundError(args.inputs/name)
    for path in [args.sources,args.substrate_fp]:
        if not path.is_file():raise FileNotFoundError(path)
    code=ROOT/'code/scripts/kinetics/train_multitask_with_replay.py'
    spec=importlib.util.spec_from_file_location('strict_alpha_release_training',code)
    implementation=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=implementation;spec.loader.exec_module(implementation)
    native_load=implementation.load_init_checkpoint
    def architecture_only(*a,**k):
        template=native_load(*a,**k)
        template.kinetics_head_state=None
        template.specificity_alpha_head_state=None
        template.specificity_alpha_base_logit_residual_scale=None
        template.run_meta={}
        return template
    implementation.load_init_checkpoint=architecture_only
    sources=implementation.load_source_specs_json(args.sources)
    feature=next(iter(sources.values()))
    models=json.loads((ROOT/'configs/models.json').read_text())
    row=next(r for r in models['checkpoints'] if r['task']=='kinetics' and r['seed']==args.seed)
    checkpoint=args.architecture_checkpoint or ROOT/row['file']
    slots=dict(checkpoint=checkpoint,inputs=args.inputs.resolve(),sources=args.sources.resolve(),
               substrate_fp=args.substrate_fp.resolve(),output=out,seed=args.seed,**feature)
    argv=['train_final']+[arg.format(**slots) for arg in json.loads((ROOT/'configs/kinetics_training_arguments.json').read_text())['arguments']]
    if args.epochs is not None:
        argv[argv.index('--epochs') + 1] = str(args.epochs)
    if args.steps_per_epoch is not None:
        if '--steps_per_epoch' in argv:
            argv[argv.index('--steps_per_epoch') + 1] = str(args.steps_per_epoch)
        else:
            argv.extend(['--steps_per_epoch', str(args.steps_per_epoch)])
    started=time.monotonic();history=[];best=float('inf');best_epoch=None;best_file=None;stale=0
    save=torch.save
    def select_checkpoint(payload,path,*a,**k):
        nonlocal best,best_epoch,best_file,stale
        if not Path(path).name.startswith('multitask_epoch'):
            return save(payload,path,*a,**k)
        metrics=payload['metrics']
        stats=payload['run_meta']['label_stats'];stats=stats.get('global',stats)
        score=float(np.mean([metrics[f'log_{task}_rmse']/stats[f'log_{task}']['std'] for task in ('kcat','km')]))
        if not np.isfinite(score):
            raise ValueError('Invalid Alpha validation score')
        epoch=int(metrics['epoch'])
        record={**metrics,'valid_mean_normalized_rmse':score,'elapsed_seconds':time.monotonic()-started}
        history.append(record)
        pd.DataFrame(history).to_csv(out/'validation_history.csv',index=False)
        if score<best-1e-4:
            best=score;best_epoch=epoch;best_file=str(path);stale=0
            # Inference checkpoints intentionally omit optimizer buffers. Each saved
            # improvement has a unique path; no old or new checkpoint is overwritten.
            clean={k:v for k,v in payload.items() if k!='optimizer'}
            save(clean,path,*a,**k)
        else:
            stale+=1
        print(json.dumps({'epoch':epoch,'valid_selection_score':score,'best_epoch':best_epoch,'stale':stale,'elapsed_seconds':time.monotonic()-started}),flush=True)
        if stale>=15 and epoch>=15:
            raise ValidationEarlyStop()
    torch.save=select_checkpoint
    sys.argv=argv
    try:
        implementation.main()
    except ValidationEarlyStop:
        print('Validation patience reached; training completed',flush=True)
    finally:
        torch.save=save
    if best_file is None or not Path(best_file).exists():
        raise ValueError('Alpha training did not produce a validation checkpoint')
    write_json(out/'manifest.json',{'completed_requested_schedule':True,'paper_schedule_requested':args.epochs is None and args.steps_per_epoch is None,'model':'AlphaEnzyme_kinetics','task':'joint','seed':args.seed,
        'best_epoch':best_epoch,'best_checkpoint':best_file,'best_valid_mean_normalized_rmse':best,'epochs_completed':len(history),
        'train_sha256':sha256(args.inputs/'train.csv'),'valid_sha256':sha256(args.inputs/'valid.csv'),
        'initialization':'classifier and kinetics heads trained from scratch for each seed',
        'input_parity':'products and EC disabled; DRFP zero; sequence/reactants and input-derived structure features',
        'elapsed_seconds':time.monotonic()-started,'argv':argv,'training_code_sha256':sha256(code)})

if __name__=='__main__':
    main()
