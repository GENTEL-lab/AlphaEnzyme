"""Shared definitions for the fixed Fig.3 mutation recipe."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
DEFAULT_INPUTS=ROOT/'data/inputs/mutation'
SEEDS=[1234,2026,3407]
LAMBDAS=[1e-5,1e-4,1e-3,.01,.1,1.,10.]
TARGETS=['zscore','gaussian_rank']
NORMALIZATIONS=['total','within_protein']
GLOBAL_VIEWS=['original','alpha','raw','paired','paired_meta','metadata']
LOCAL_VIEWS=['site_wt','site_delta','site_pair','site_meta','global_site_meta']
RECIPE = dict(input_dim=2304, alpha_dim=1152, raw_dim=1152, head_type='residual_mlp',
              hidden_dim=768, bottleneck_dim=384, dropout=0.2, epochs=40,
              batch_size=512, eval_batch_size=4096, lr=0.0002, weight_decay=0.008,
              rank_loss_weight=0.05, gradient_clip=1.0)

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)

def read_json(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def new_model(native):
    return native.CleanHead(**{k: RECIPE[k] for k in ['input_dim', 'alpha_dim', 'raw_dim',
                            'head_type', 'hidden_dim', 'bottleneck_dim', 'dropout']})

def metrics(pred, target):
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    assert len(pred) == len(target) and np.isfinite(pred).all() and np.isfinite(target).all()
    assert np.ptp(target) > 0
    if np.ptp(pred) == 0:
        scc, pcc = 0.0, 0.0
    else:
        scc = float(spearmanr(pred, target).statistic)
        pcc = float(pearsonr(pred, target).statistic)
    # Linear relevance, as in the existing benchmark; stable ties preserve input row order.
    k = min(20, len(target))
    rel = target - target.min()
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.argsort(-target, kind='stable')[:k]
    order = np.argsort(-pred, kind='stable')[:k]
    ndcg = float(np.sum(rel[order] * discount) / np.sum(rel[ideal] * discount))
    return dict(SCC=scc, PCC=pcc, NDCG20=ndcg, n=len(target))
GLOBAL_CONFIGS = [
    dict(id='nn_paired',kind='nn',view='paired',normalization='total',target='zscore',rank_weight=.05,balanced=False),
    dict(id='nn_paired_within',kind='nn',view='paired',normalization='within_protein',target='zscore',rank_weight=.05,balanced=False),
    dict(id='nn_paired_meta',kind='nn',view='paired_meta',normalization='within_protein',target='gaussian_rank',rank_weight=.2,balanced=True),
    dict(id='nn_raw',kind='nn',view='raw',normalization='within_protein',target='gaussian_rank',rank_weight=.2,balanced=True),
    dict(id='xgb_rank_paired_meta',kind='xgb',view='paired_meta',normalization='total',target='gaussian_rank',objective='rank:pairwise',depth=3),
    dict(id='xgb_rank_paired',kind='xgb',view='paired',normalization='total',target='gaussian_rank',objective='rank:pairwise',depth=5),
    dict(id='xgb_reg_raw',kind='xgb',view='raw',normalization='total',target='gaussian_rank',objective='reg:squarederror',depth=3),
    dict(id='xgb_reg_paired_meta',kind='xgb',view='paired_meta',normalization='total',target='gaussian_rank',objective='reg:squarederror',depth=3),
]
LOCAL_CONFIGS = [
 dict(id='nn_site_pair',kind='nn',view='site_pair',normalization='within_protein',target='gaussian_rank',rank_weight=.2,balanced=True),
 dict(id='nn_global_site',kind='nn',view='global_site_meta',normalization='total',target='gaussian_rank',rank_weight=.2,balanced=True),
 dict(id='xgb_site_rank',kind='xgb',view='site_meta',normalization='total',target='gaussian_rank',objective='rank:pairwise',depth=3),
 dict(id='xgb_global_site_reg',kind='xgb',view='global_site_meta',normalization='total',target='gaussian_rank',objective='reg:squarederror',depth=3),
]
def normalize_features(x, train_assays, mode):
    import numpy as np
    x = np.asarray(x,dtype=np.float64)
    mean=x.mean(0,keepdims=True)
    if mode=='total':
        variance=x.var(0,keepdims=True)
    else:
        variance=np.zeros((1,x.shape[1]))
        for assay in sorted(set(train_assays)):
            group=x[train_assays==assay]
            variance+=len(group)*group.var(0,keepdims=True)/len(x)
    std=np.sqrt(variance)
    std=np.where(std<1e-6,1.,std)
    return mean.astype('float32'),std.astype('float32')

def train_targets(rows, mode):
    import numpy as np
    from scipy.stats import norm,rankdata
    y=np.empty(len(rows),dtype=np.float32)
    assay=rows.DMS_id.to_numpy(str)
    for a in sorted(set(assay)):
        mask=assay==a
        raw=rows.loc[mask,'DMS_score'].to_numpy(float)
        value=raw if mode=='zscore' else norm.ppf((rankdata(raw,method='average')-.5)/len(raw))
        std=float(value.std())
        y[mask]=(value-value.mean())/(std if std>=1e-8 else 1.)
    assert np.isfinite(y).all()
    return y

def native_module():
    import models
    return SimpleNamespace(CleanHead=models.CleanHead,set_seed=models.set_seed,
        sanitize=models.sanitize,rx=SimpleNamespace(pairwise_loss=models.pairwise_loss))

def load_view(view, inputs=DEFAULT_INPUTS):
    folder=Path(inputs)/'features'
    def arr(name):return np.load(folder/(name+'.npy'),mmap_mode='r')
    if view in ('original','paired','metadata','site_pair'):return arr(view)
    if view=='alpha':return arr('original')[:,:1152]
    if view=='raw':return arr('original')[:,1152:]
    if view=='paired_meta':return np.concatenate([arr('paired'),arr('metadata')],axis=1)
    if view=='site_wt':return arr('site_pair')[:,:512]
    if view=='site_delta':return arr('site_pair')[:,512:]
    if view=='site_meta':return np.concatenate([arr('site_pair'),arr('metadata')],axis=1)
    if view=='global_site_meta':return np.concatenate([arr('paired'),arr('metadata'),arr('site_pair')],axis=1)
    raise ValueError('Unknown feature view: '+str(view))

def add_common_arguments(parser):
    parser.add_argument('--inputs',type=Path,default=DEFAULT_INPUTS)
    parser.add_argument('--output',type=Path,required=True)

def output_directory(path):
    path=Path(path).resolve()
    path.mkdir(parents=True,exist_ok=False)
    return path
