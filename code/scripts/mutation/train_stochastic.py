import argparse
import math
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from common import *

@torch.no_grad()
def predict_nn(model,x):
    model.eval()
    return np.concatenate([model(x[i:i+4096]).cpu().numpy() for i in range(0,len(x),4096)])

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,choices=[c['id'] for c in GLOBAL_CONFIGS+LOCAL_CONFIGS])
    parser.add_argument('--fold',required=True,type=int)
    parser.add_argument('--seed',required=True,type=int)
    add_common_arguments(parser)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--rounds',type=int,default=300)
    args=parser.parse_args()
    HERE=SOURCE=args.inputs.resolve()
    CONFIG_MAP={c['id']:c for c in GLOBAL_CONFIGS+LOCAL_CONFIGS}
    cfg=CONFIG_MAP[args.config]
    assert args.seed in SEEDS and torch.cuda.is_available()
    torch.set_num_threads(2)
    native=native_module()
    native.set_seed(args.seed)
    fold=read_json(HERE/'folds.json')[args.fold]
    out=output_directory(args.output)
    assert not (out/'complete.json').exists()
    start=time.time()
    frames={s:pd.read_csv(SOURCE/f'inputs/fold_{args.fold}_{s}_rows.csv',float_precision='round_trip') for s in ['train','valid','test']}
    assert 'DMS_score' not in frames['test'] and set(frames['train'].DMS_id)==set(fold['train'])
    assert set(frames['valid'].DMS_id)=={fold['valid']} and set(frames['test'].DMS_id)=={fold['test']}
    indices={s:np.load(SOURCE/f'inputs/fold_{args.fold}_{s}_indices.npy') for s in frames}
    feature=load_view(cfg['view'],HERE)
    assays=frames['train'].DMS_id.to_numpy(str)
    mean,std=normalize_features(feature[indices['train']],assays,cfg['normalization'])
    np.savez(out/'normalization.npz',mean=mean,std=std,fitted_rows=indices['train'])
    inputs={s:np.asarray((feature[idx]-mean)/std,dtype=np.float32) for s,idx in indices.items()}
    assert all(np.isfinite(x).all() for x in inputs.values())
    y=train_targets(frames['train'],cfg['target'])
    valid_y=frames['valid'].DMS_score.to_numpy(float)
    protocol_hash=sha(HERE/'protocol.json')
    if cfg['kind']=='nn':
        model_args=dict(input_dim=inputs['train'].shape[1],alpha_dim=1152 if cfg['view'] in ['paired','paired_meta','global_site_meta'] else 0,
            raw_dim=inputs['train'].shape[1]-(1152 if cfg['view'] in ['paired','paired_meta','global_site_meta'] else 0),
            head_type='residual_mlp',hidden_dim=256,bottleneck_dim=128,dropout=.4)
        model=native.CleanHead(**model_args).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.03)
        xt={s:torch.from_numpy(x).cuda() for s,x in inputs.items()}
        yt=torch.from_numpy(y).cuda()
        rng=torch.Generator(device='cpu').manual_seed(args.seed)
        groups=[np.flatnonzero(assays==a) for a in fold['train']]
        best,best_epoch,best_state=-float('inf'),0,None
        history=[]
        for epoch in range(1,args.epochs+1):
            model.train()
            order=torch.randperm(len(y),generator=rng).numpy()
            losses=[]
            for step in range(math.ceil(len(y)/512)):
                if cfg['balanced']:
                    ii=np.concatenate([g[torch.randint(len(g),(85,),generator=rng).numpy()] for g in groups])
                    ii=ii[torch.randperm(len(ii),generator=rng).numpy()]
                else:
                    ii=order[step*512:(step+1)*512]
                gi=torch.as_tensor(ii,device='cuda',dtype=torch.long)
                pred=model(xt['train'][gi])
                target=yt[gi]
                optimizer.zero_grad(set_to_none=True)
                loss=F.mse_loss(pred,target)+cfg['rank_weight']*native.rx.pairwise_loss(pred,target,assays[ii].tolist())
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                losses.append(float(loss.detach()))
            vpred=predict_nn(model,xt['valid'])
            vm=metrics(vpred,valid_y)
            history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),valid_SCC=vm['SCC'],valid_PCC=vm['PCC'],elapsed_seconds=time.time()-start))
            if vm['SCC']>best:
                best,best_epoch=vm['SCC'],epoch
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            if epoch==1 or epoch%10==0:
                print(history[-1],flush=True)
        torch.save(dict(state_dict=best_state,model_args=model_args,config=cfg,fold=fold,seed=args.seed,
            best_epoch=best_epoch,protocol_sha256=protocol_hash,initialization='random'),out/'best.pt')
        model.load_state_dict(best_state)
        vp=predict_nn(model,xt['valid'])
        check=native.CleanHead(**model_args).cuda()
        check.load_state_dict(torch.load(out/'best.pt',weights_only=False,map_location='cpu')['state_dict'])
        assert np.array_equal(vp,predict_nn(check,xt['valid']))
        tp=predict_nn(check,xt['test'])
        selected_step=best_epoch
        checkpoint=out/'best.pt'
    else:
        import xgboost as xgb
        order=np.argsort(assays,kind='stable')
        groups=[int((assays==a).sum()) for a in sorted(set(assays))]
        ranking=cfg['objective'].startswith('rank:')
        weight=np.ones(len(groups),dtype=np.float32) if ranking else np.asarray([len(y)/(6*(assays==a).sum()) for a in assays[order]],dtype=np.float32)
        train_dm=xgb.DMatrix(inputs['train'][order],label=y[order],weight=weight,nthread=2,
            group=groups if ranking else None)
        valid_dm=xgb.DMatrix(inputs['valid'],label=valid_y,nthread=2,group=[len(valid_y)] if ranking else None)
        test_dm=xgb.DMatrix(inputs['test'],nthread=2)
        params=dict(objective=cfg['objective'],device='cuda',tree_method='hist',max_depth=cfg['depth'],
                    eta=.05,min_child_weight=20,reg_lambda=10.,max_bin=128,subsample=.8,
                    colsample_bytree=.8,nthread=2,seed=args.seed,disable_default_eval_metric=1)
        if ranking:
            params.update(lambdarank_pair_method='mean',lambdarank_num_pair_per_sample=4)
        def metric(pred,dm):
            value=0. if np.ptp(pred)==0 else float(spearmanr(pred,valid_y).statistic)
            return 'SCC',value
        history={}
        model=xgb.train(params,train_dm,num_boost_round=args.rounds,evals=[(valid_dm,'valid')],
            custom_metric=metric,maximize=True,early_stopping_rounds=30,evals_result=history,verbose_eval=50)
        assert 'cuda' in model.save_config(), 'XGBoost did not use GPU.'
        model=model[:model.best_iteration+1]
        checkpoint=out/'best.ubj'
        model.save_model(checkpoint)
        pd.DataFrame(dict(iteration=np.arange(1,len(history['valid']['SCC'])+1),valid_SCC=history['valid']['SCC'])).to_csv(out/'history.csv',index=False)
        vp=model.predict(valid_dm)
        check=xgb.Booster(model_file=checkpoint,params={'device':'cuda'})
        assert np.array_equal(vp,check.predict(valid_dm))
        tp=check.predict(test_dm)
        selected_step=model.num_boosted_rounds()
        write_json(out/'model_params.json',dict(params=params,configuration=cfg,protocol_sha256=protocol_hash))
    assert np.isfinite(vp).all() and np.isfinite(tp).all()
    frames['valid'].assign(prediction=vp).to_csv(out/'validation_predictions.csv',index=False)
    frames['test'].assign(prediction=tp).to_csv(out/'test_predictions_unlabelled.csv',index=False)
    write_json(out/'complete.json',dict(configuration=args.config,fold=args.fold,seed=args.seed,kind=cfg['kind'],
        selected_step=selected_step,valid_SCC=metrics(vp,valid_y)['SCC'],elapsed_seconds=time.time()-start,
        checkpoint_sha256=sha(checkpoint),checkpoint_reload_check=True,test_label_access=False,protocol_sha256=protocol_hash))
    print('COMPLETE',flush=True)

if __name__=='__main__':
    main()
