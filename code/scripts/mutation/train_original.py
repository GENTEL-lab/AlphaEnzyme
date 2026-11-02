import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import *

@torch.no_grad()
def predict(model, x):
    model.eval()
    return np.concatenate([model(x[start:start + RECIPE['eval_batch_size']]).cpu().numpy()
                           for start in range(0, len(x), RECIPE['eval_batch_size'])])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    add_common_arguments(parser)
    parser.add_argument('--epochs',type=int,default=40)
    args = parser.parse_args()
    HERE=args.inputs.resolve()
    RECIPE['epochs']=args.epochs
    assert args.seed in SEEDS and torch.cuda.is_available(), 'GPU required'
    torch.set_num_threads(2)
    native = native_module()
    native.set_seed(args.seed)
    fold = read_json(HERE / 'folds.json')[args.fold]
    out = output_directory(args.output)
    assert not (out / 'complete.json').exists()
    start = time.time()
    recipe_hash = sha(HERE / 'protocol.json')
    train_rows = pd.read_csv(HERE / f'inputs/fold_{args.fold}_train_rows.csv')
    valid_rows = pd.read_csv(HERE / f'inputs/fold_{args.fold}_valid_rows.csv')
    test_rows = pd.read_csv(HERE / f'inputs/fold_{args.fold}_test_rows.csv')
    assert 'DMS_score' not in test_rows
    assert set(train_rows.DMS_id) == set(fold['train'])
    assert set(valid_rows.DMS_id) == {fold['valid']} and set(test_rows.DMS_id) == {fold['test']}
    features = np.load(HERE / 'features/original.npy', mmap_mode='r')
    stats = np.load(HERE / f'features/fold_{args.fold}_normalization.npz')
    def input_tensor(split):
        idx = np.load(HERE / f'inputs/fold_{args.fold}_{split}_indices.npy')
        if split == 'train':
            assert np.array_equal(idx, stats['fitted_row_indices'])
        x = native.sanitize((features[idx] - stats['mean']) / stats['std'])
        return torch.from_numpy(x).to('cuda')
    xtrain, xvalid = input_tensor('train'), input_tensor('valid')
    assays = train_rows.DMS_id.to_numpy(str)
    y = train_rows.DMS_score.to_numpy(np.float32)
    normalized_y = np.empty_like(y)
    label_stats = {}
    for assay in fold['train']:
        mask = assays == assay
        mean, std = float(y[mask].mean()), float(y[mask].std())
        std = std if std >= 1e-8 else 1.0
        normalized_y[mask] = (y[mask] - mean) / std
        label_stats[assay] = dict(mean=mean, std=std, n=int(mask.sum()))
    write_json(out / 'label_normalization.json', label_stats)
    yt = torch.from_numpy(normalized_y).to('cuda')
    valid_y = valid_rows.DMS_score.to_numpy(float)
    model = new_model(native).to('cuda')
    # Data-order RNG is separate from CUDA dropout / ranking RNG and follows the fixed seed.
    order_rng = torch.Generator(device='cpu').manual_seed(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=RECIPE['lr'], weight_decay=RECIPE['weight_decay'])
    history, best, best_epoch, best_state = [], -float('inf'), None, None
    write_json(out / 'started.json', dict(fold=fold, seed=args.seed, initialization='random',
                  warm_start=False, device=torch.cuda.get_device_name(), protocol_sha256=recipe_hash))
    print(f"fold={args.fold} seed={args.seed} train={len(y)} valid={len(valid_y)} GPU={torch.cuda.get_device_name()}", flush=True)
    for epoch in range(1, RECIPE['epochs'] + 1):
        model.train()
        order = torch.randperm(len(y), generator=order_rng).numpy()
        losses, sizes = [], []
        for offset in range(0, len(order), RECIPE['batch_size']):
            idx = order[offset:offset + RECIPE['batch_size']]
            gpu_idx = torch.as_tensor(idx, device='cuda', dtype=torch.long)
            pred = model(xtrain[gpu_idx])
            target = yt[gpu_idx]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(pred, target) + RECIPE['rank_loss_weight'] * native.rx.pairwise_loss(pred, target, assays[idx].tolist())
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE['gradient_clip'])
            optimizer.step()
            losses.append(float(loss.detach()))
            sizes.append(len(idx))
        vm = metrics(predict(model, xvalid), valid_y)
        row = dict(epoch=epoch, train_loss=float(np.average(losses, weights=sizes)),
                   valid_SCC=vm['SCC'], valid_PCC=vm['PCC'], elapsed_seconds=time.time()-start)
        history.append(row)
        if vm['SCC'] > best:
            best, best_epoch = vm['SCC'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pd.DataFrame(history).to_csv(out / 'history.csv', index=False)
        if epoch == 1 or epoch % 5 == 0:
            print(json.dumps(row), flush=True)
    assert best_state is not None
    model.load_state_dict(best_state)
    saved = dict(state_dict=best_state, recipe=RECIPE, fold=fold, seed=args.seed,
                 best_epoch=best_epoch, valid_SCC=best, label_stats=label_stats,
                 feature_normalization=f'fold_{args.fold}_normalization.npz',
                 protocol_sha256=recipe_hash, initialization='random', warm_start=False)
    np.savez(out / f'fold_{args.fold}_normalization.npz',**{k:stats[k] for k in stats.files})
    torch.save(saved, out / 'best.pt')
    # Verify a fresh reload reproduces validation predictions before scoring any test labels.
    vpred = predict(model, xvalid)
    loaded = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    reloaded = new_model(native).to('cuda')
    reloaded.load_state_dict(loaded['state_dict'])
    assert np.allclose(vpred, predict(reloaded, xvalid), rtol=0, atol=0)
    valid_rows.assign(prediction=vpred).to_csv(out / 'validation_predictions.csv', index=False)
    xtest = input_tensor('test')
    test_pred = predict(reloaded, xtest)
    assert np.isfinite(test_pred).all()
    test_rows.assign(prediction=test_pred).to_csv(out / 'test_predictions_unlabelled.csv', index=False)
    write_json(out / 'complete.json', dict(fold=args.fold, seed=args.seed, best_epoch=best_epoch,
        valid_SCC=best, train_rows=len(train_rows), valid_rows=len(valid_rows), test_rows=len(test_rows),
        epochs=RECIPE['epochs'], elapsed_seconds=time.time()-start, checkpoint_sha256=sha(out/'best.pt'),
        checkpoint_reload_prediction_check=True, test_label_access=False, protocol_sha256=recipe_hash))
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
