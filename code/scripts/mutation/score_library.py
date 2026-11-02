"""Reproduce the released fusion-Q0.5 score from the component predictions.

Inputs are three-seed-averaged component scores for each original protein fold.
The whole unlabelled variant library is required. No experimental labels are used.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

GLOBAL_PARENTS = ['original', 'g_nn_paired', 'g_nn_paired_within', 'g_nn_paired_meta',
                  'g_nn_raw', 'g_xgb_rank_paired_meta', 'g_xgb_rank_paired',
                  'g_xgb_reg_raw', 'g_xgb_reg_paired_meta', 'g_ridge']
SITE_PARENTS = ['original', 's_nn_site_pair', 's_nn_global_site', 's_xgb_site_rank',
                's_xgb_global_site_reg', 's_ridge']
COMPONENTS = list(dict.fromkeys(GLOBAL_PARENTS + SITE_PARENTS))


def score_library(frame):
    assert len(frame) and frame.id.is_unique
    output = frame[['id', 'DMS_id']].copy()
    scores = np.empty(len(frame), dtype=float)
    blend_scores = np.empty(len(frame), dtype=float)
    for assay, positions in frame.groupby('DMS_id', sort=False).indices.items():
        subset = frame.iloc[positions]
        n = len(subset)
        assert np.isfinite(subset[COMPONENTS].to_numpy(float)).all()
        global_scores = np.mean([rankdata(subset[m].to_numpy(float), method='average') / n
                                 for m in GLOBAL_PARENTS], axis=0)
        site_scores = np.mean([rankdata(subset[m].to_numpy(float), method='average') / n
                               for m in SITE_PARENTS], axis=0)
        blend = np.mean([rankdata(global_scores, method='average') / n,
                         rankdata(site_scores, method='average') / n], axis=0)
        scores[positions] = ((rankdata(blend, method='average') - .5) / n) ** .5
        blend_scores[positions] = blend
    output['fusion_score_before_calibration'] = blend_scores
    output['prediction'] = scores
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    frame = pd.read_csv(args.input, float_precision='round_trip')
    assert 'DMS_score' not in frame, 'Use an unlabelled component input table.'
    result = score_library(frame)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output, index=False,mode='x')
    print(f'{len(result)} predictions saved to {args.output}')


if __name__ == '__main__':
    main()
