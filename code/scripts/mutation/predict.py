#!/usr/bin/env python3
"""Run the released mutation ensemble, then the fixed whole-library rank fusion."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import ROOT, DEFAULT_INPUTS, load_view, read_json
from models import CleanHead, sanitize
from score_library import score_library, COMPONENTS

BUNDLE_FORMAT = 'alphaenzyme.mutation.ensemble.v1'
DEFAULT_BUNDLE = ROOT / 'data/checkpoints/mutation_final.pt'


def load_bundle(path: Path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} not found; download AlphaEnzyme_checkpoints.zip and unpack it into data/'
        )
    bundle = torch.load(path, map_location='cpu', weights_only=False)
    if bundle.get('format') != BUNDLE_FORMAT:
        raise ValueError(f'Unsupported mutation bundle format in {path}')
    entries = bundle.get('components') or []
    if len(entries) != 328:
        raise ValueError(f'Expected 328 component fits in {path}, found {len(entries)}')
    if {e.get('component') for e in entries} != set(COMPONENTS):
        raise ValueError('Mutation bundle does not contain the expected 15 components')
    return bundle


@torch.no_grad()
def predict_component(record, indices, inputs, device):
    kind = record['kind']
    if kind == 'xgb':
        import xgboost as xgb
        mean = record['normalization_mean'].numpy()
        std = record['normalization_std'].numpy()
        raw = load_view(record['view'], inputs)[indices]
        x = np.asarray((raw - mean) / std, dtype=np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite normalized features')
        model = xgb.Booster()
        model.load_model(bytearray(record['model_bytes']))
        return model.predict(xgb.DMatrix(x, nthread=2)).astype(str).astype(float)
    if kind == 'ridge':
        x = torch.from_numpy(np.asarray(
            load_view(record['feature_view'], inputs)[indices], dtype=np.float64
        )).to(device)
        return ((x - record['input_mean'].to(device)) @ record['coefficient'].to(device)).cpu().numpy()
    if kind == 'nn':
        mean = record['normalization_mean'].numpy()
        std = record['normalization_std'].numpy()
        raw = load_view(record['view'], inputs)[indices]
        x = sanitize((raw - mean) / std)
        model = CleanHead(**record['model_args']).to(device)
        model.load_state_dict(record['state_dict'], strict=True)
        model.eval()
        scores = np.concatenate([
            model(torch.from_numpy(x[i:i + 4096]).to(device)).cpu().numpy()
            for i in range(0, len(x), 4096)
        ])
        # Retain the original float32 per-seed decimal boundary for identical rank ties.
        return scores.astype(str).astype(float)
    raise ValueError('Unknown mutation component kind: ' + str(kind))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, default=DEFAULT_INPUTS)
    p.add_argument('--rows', type=Path,
                   help='Rows in the same order as the feature matrices; only id and DMS_id are read')
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_BUNDLE,
                   help='Single-file mutation ensemble bundle')
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument('--paper-folds', action='store_true',
                        help='Use the original held-out fold for each known protein')
    choice.add_argument('--fold', type=int, choices=range(8),
                        help='Explicit model fold for a new unlabelled candidate library')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--components-output', type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.set_num_threads(2)
    for path in (a.output, a.components_output):
        if path is not None and path.exists():
            raise FileExistsError(path)

    bundle = load_bundle(a.checkpoint)
    entries = bundle['components']
    frame = pd.read_csv(a.rows or a.inputs / 'inputs/rows.csv', usecols=['id', 'DMS_id'], dtype=str)
    if frame.isna().any().any() or not frame.id.is_unique or frame.empty:
        raise ValueError('Nonempty unique IDs and DMS_id are required')
    for name, dim in [('original', 2304), ('paired', 2304), ('metadata', 62), ('site_pair', 1024)]:
        value = load_view(name, a.inputs)
        if value.shape != (len(frame), dim) or not np.isfinite(value).all():
            raise ValueError(f'Invalid {name} shape or values: {value.shape}')
    fold_map = {f['test']: f['fold'] for f in read_json(a.inputs / 'folds.json')} if a.paper_folds else {}
    if a.paper_folds and not set(frame.DMS_id) <= set(fold_map):
        raise ValueError('Unknown assay: explicitly choose --fold for new data')
    assignment = frame.DMS_id.map(fold_map) if a.paper_folds else pd.Series(a.fold, index=frame.index)
    for component in COMPONENTS:
        frame[component] = np.nan
    for fold in sorted(assignment.unique()):
        indices = np.flatnonzero(assignment.to_numpy() == fold)
        selected = [e for e in entries if e['fold'] == int(fold)]
        for component in COMPONENTS:
            records = [e for e in selected if e['component'] == component]
            expected = 1 if component.endswith('_ridge') else 3
            if len(records) != expected:
                raise ValueError(f'Incomplete component ensemble: {component}/fold {fold}')
            predictions = [predict_component(row, indices, a.inputs, a.device) for row in records]
            mean = np.mean(predictions, axis=0)
            if not np.isfinite(mean).all():
                raise ValueError('Nonfinite component output')
            frame.loc[indices, component] = mean
        print(f'Completed fold {fold}: {len(indices)} variants', flush=True)
    result = score_library(frame)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(a.output, index=False, mode='x')
    if a.components_output is not None:
        a.components_output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(a.components_output, index=False, mode='x')
    print(f'{len(result)} predictions saved to {a.output}')


if __name__ == '__main__':
    main()
