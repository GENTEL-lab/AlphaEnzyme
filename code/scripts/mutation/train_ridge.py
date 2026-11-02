import argparse
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from common import *

def scc(score,target):
    return 0. if np.ptp(score)==0 else float(spearmanr(score,target).statistic)

def sufficient_statistics(x,assays,targets):
    parts=[]
    for a in sorted(set(assays)):
        mask=assays==a
        group=x[torch.from_numpy(mask).cuda()]
        y=targets[torch.from_numpy(mask).cuda()]
        mu=group.mean(0)
        centered=group-mu
        yc=y-y.mean(0)
        parts.append(dict(assay=a,mean=mu,cov=centered.T@centered/len(group),cross=centered.T@yc/len(group)))
    return parts

def coefficients(parts,ids,mode,lambdas):
    selected=[parts[j] for j in ids]
    mean=torch.stack([p['mean'] for p in selected]).mean(0)
    covariance=torch.stack([p['cov'] for p in selected]).mean(0)
    if mode=='total':
        means=torch.stack([p['mean']-mean for p in selected])
        covariance=covariance+means.T@means/len(selected)
    cross=torch.stack([p['cross'] for p in selected]).mean(0)
    std=torch.sqrt(torch.clamp(covariance.diagonal(),min=0.))
    std=torch.where(std<1e-6,torch.ones_like(std),std)
    gram=covariance/std[:,None]/std[None,:]
    rhs=cross/std[:,None]
    lam=torch.as_tensor(lambdas,dtype=torch.float64,device='cuda')
    identity=torch.eye(len(std),dtype=torch.float64,device='cuda')
    matrix=gram[None,:,:]+lam[:,None,None]*identity[None,:,:]
    factor=torch.linalg.cholesky(matrix)
    coef=torch.cholesky_solve(rhs.expand(len(lambdas),-1,-1),factor)
    assert torch.isfinite(coef).all()
    return coef/std[None,:,None],mean,std

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',required=True,type=int)
    add_common_arguments(parser)
    parser.add_argument('--family',choices=['global','local'],required=True)
    args=parser.parse_args()
    HERE=SOURCE=args.inputs.resolve()
    VIEWS=GLOBAL_VIEWS if args.family=='global' else LOCAL_VIEWS
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    fold=read_json(HERE/'folds.json')[args.fold]
    out=output_directory(args.output)
    assert not (out/'complete.json').exists()
    start=time.time()
    train=pd.read_csv(SOURCE/f'inputs/fold_{args.fold}_train_rows.csv',float_precision='round_trip')
    valid=pd.read_csv(SOURCE/f'inputs/fold_{args.fold}_valid_rows.csv',float_precision='round_trip')
    test=pd.read_csv(SOURCE/f'inputs/fold_{args.fold}_test_rows.csv',float_precision='round_trip')
    assert 'DMS_score' not in test and set(train.DMS_id)==set(fold['train'])
    train_idx=np.load(SOURCE/f'inputs/fold_{args.fold}_train_indices.npy')
    valid_idx=np.load(SOURCE/f'inputs/fold_{args.fold}_valid_indices.npy')
    test_idx=np.load(SOURCE/f'inputs/fold_{args.fold}_test_indices.npy')
    assays=train.DMS_id.to_numpy(str)
    assay_names=sorted(set(assays))
    y=torch.from_numpy(np.stack([train_targets(train,t) for t in TARGETS],axis=1).astype(np.float64)).cuda()
    inner_rows=[]
    # No outer validation/test labels contribute to this hyperparameter search.
    for view in VIEWS:
        raw=load_view(view,HERE)
        x=torch.from_numpy(np.asarray(raw[train_idx],dtype=np.float64)).cuda()
        parts=sufficient_statistics(x,assays,y)
        for normalization in NORMALIZATIONS:
            for held in range(6):
                included=[j for j in range(6) if j!=held]
                beta,mean,std=coefficients(parts,included,normalization,LAMBDAS)
                mask=assays==assay_names[held]
                matrix=(x[torch.from_numpy(mask).cuda()]-mean)@beta.permute(1,0,2).reshape(x.shape[1],-1)
                preds=matrix.cpu().numpy().reshape(int(mask.sum()),len(LAMBDAS),len(TARGETS))
                target=train.loc[mask,'DMS_score'].to_numpy(float)
                for li,lam in enumerate(LAMBDAS):
                    for ti,t in enumerate(TARGETS):
                        inner_rows.append(dict(view=view,normalization=normalization,target=t,regularization=lam,
                            inner_valid=assay_names[held],inner_train=';'.join(assay_names[j] for j in included),
                            inner_SCC=scc(preds[:,li,ti],target)))
        pd.DataFrame(inner_rows).to_csv(out/'inner_cv_details.csv',index=False)
        print(f'fold={args.fold} ridge view={view} completed elapsed={time.time()-start:.1f}s',flush=True)
    details=pd.DataFrame(inner_rows)
    grid=details.groupby(['view','normalization','target','regularization'],as_index=False,sort=False).inner_SCC.mean()
    grid=grid.sort_values('inner_SCC',ascending=False,kind='stable').reset_index(drop=True)
    grid.to_csv(out/'inner_cv_grid.csv',index=False)
    best=grid.iloc[0].to_dict()
    assert np.isfinite(best['inner_SCC'])
    raw=load_view(best['view'],HERE)
    x=torch.from_numpy(np.asarray(raw[train_idx],dtype=np.float64)).cuda()
    parts=sufficient_statistics(x,assays,y)
    beta,mean,std=coefficients(parts,list(range(6)),best['normalization'],[best['regularization']])
    ti=TARGETS.index(best['target'])
    coefficient=beta[0,:,ti]
    def predict(indices):
        xx=torch.from_numpy(np.asarray(raw[indices],dtype=np.float64)).cuda()
        return ((xx-mean)@coefficient).cpu().numpy()
    vp,tp=predict(valid_idx),predict(test_idx)
    checkpoint=out/'best.pt'
    torch.save(dict(coefficient=coefficient.cpu(),input_mean=mean.cpu(),input_scale=std.cpu(),
        selected_configuration=best,fold=fold,feature_view=best['view'],fitted_proteins=assay_names,
        protocol_sha256=sha(HERE/'protocol.json'),deterministic=True),checkpoint)
    loaded=torch.load(checkpoint,weights_only=False,map_location='cuda')
    xx=torch.from_numpy(np.asarray(raw[valid_idx],dtype=np.float64)).cuda()
    assert np.array_equal(((xx-loaded['input_mean'])@loaded['coefficient']).cpu().numpy(),vp)
    valid.assign(prediction=vp).to_csv(out/'validation_predictions.csv',index=False)
    test.assign(prediction=tp).to_csv(out/'test_predictions_unlabelled.csv',index=False)
    write_json(out/'selection_inner_cv.json',dict(selection_basis='six inner protein-held-out SCC values averaged equally',
        configuration=best,outer_valid_labels_used_for_selection=False,outer_test_labels_used=False))
    write_json(out/'complete.json',dict(fold=args.fold,kind='ridge',configuration=best,deterministic=True,
        inner_cv_models=len(inner_rows),valid_SCC=metrics(vp,valid.DMS_score.to_numpy())['SCC'],
        elapsed_seconds=time.time()-start,checkpoint_sha256=sha(checkpoint),checkpoint_reload_check=True,
        test_label_access=False,protocol_sha256=sha(HERE/'protocol.json')))
    print('COMPLETE',best,flush=True)

if __name__=='__main__':
    main()
