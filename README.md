# AlphaEnzyme

Manuscript (citation coming soon) · [🤗 Model checkpoints](https://huggingface.co/GENTEL-Lab/AlphaEnzyme) · [📦 Data](https://huggingface.co/GENTEL-Lab/AlphaEnzyme)

AlphaEnzyme is a unified enzyme representation and prediction framework. This
release provides the released model weights, training and inference code, the released
task tables used by the paper, and the result tables for three tasks:

- **Specificity**: enzyme–reaction pair ranking.
- **Kinetics**: joint prediction of log10 kcat and log10 Km.
- **Mutation**: within-library ranking of variant activity.

## Release contents

```text
code/alphaenzyme/              shared encoder, task heads, datasets and losses
code/scripts/                  training, evaluation and feature-preparation entry points
code/third_party/              compatible Protenix source runtime (Apache-2.0)
scripts/                       user-facing entry points and reproduction
configs/                       final recipes, model identities and input contracts
data/checkpoints/              released weights (downloaded separately)
data/inputs/                   released task tables (downloaded separately)
data/results/                  released prediction and metric tables
docs/                          third-party notices and licenses
demo/                          small precomputed examples
```

The public repository contains code, configuration, documentation, a small
demo and the released metric tables. Model weights and raw data are distributed
separately because of their size.

## Environment

The tested environment uses Python 3.11 and PyTorch 2.5.1 with CUDA 12.4.

### uv (recommended)

Install uv if needed (tested with 0.9.28), and an NVIDIA driver compatible with
CUDA 12.4 (>= 525):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

```bash
uv venv --python 3.11
source .venv/bin/activate
uv sync --frozen
```

For raw feature generation, install the preprocessing extra:

```bash
uv sync --frozen --extra preprocessing
```

After installation, verify:

```bash
python -c "import torch, accelerate, transformers; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), accelerate.__version__, transformers.__version__)"
python -B scripts/predict_specificity.py --help
python -B scripts/predict_kinetics.py --help
python -B code/scripts/mutation/score_library.py --help
python -B scripts/reproduce.py --help
```

### conda / pip (alternative)

```bash
conda create -n alphaenzyme python=3.11
conda activate alphaenzyme
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -r requirements-preprocessing.txt   # optional
```

Or run `./setup_env.sh` from the repository root (uv).

## Model checkpoints

Download the released weights from Hugging Face and unpack them into `data/`:

```bash
hf download GENTEL-Lab/AlphaEnzyme AlphaEnzyme_checkpoints.zip --local-dir .
unzip AlphaEnzyme_checkpoints.zip -d data/
```

The archive creates:

```text
data/checkpoints/specificity/specificity_final.pt
data/checkpoints/kinetics/seed_1234/multitask_epoch18.pt
data/checkpoints/kinetics/seed_2026/multitask_epoch2.pt
data/checkpoints/kinetics/seed_3407/multitask_epoch16.pt
data/checkpoints/mutation_final.pt            # one-file mutation ensemble (328 component fits)
```

The kinetics result is the mean prediction of the three released seeds. The
mutation result uses the fixed Fig. 3 fusion of the 15 components listed in
`configs/mutation.json`; `mutation_final.pt` is a single-file bundle of the 328
component fits and their fold normalization, used by the mutation predictor. It
is one download file, not a single neural network: the final mutation score is
an ensemble followed by whole-library rank fusion.

## Data

Download `AlphaEnzyme_data.zip` from the
[AlphaEnzyme Hugging Face repository](https://huggingface.co/GENTEL-Lab/AlphaEnzyme)
and unpack it into `data/`:

```bash
hf download GENTEL-Lab/AlphaEnzyme AlphaEnzyme_data.zip --local-dir .
unzip AlphaEnzyme_data.zip -d data/
```

This creates the released task tables and the small input assets used by the three tasks:

```text
data/inputs/specificity/tables/{train,valid}.csv
data/inputs/specificity/eval/eval.csv
data/inputs/kinetics/{train,valid,test}.csv
data/inputs/kinetics/{alpha_sources.json,alpha_training_zscore_stats.npz}
data/inputs/mutation/{variants,references,proteingym_rows}.csv
data/inputs/mutation/inputs/rows.csv
data/inputs/mutation/inputs/fold_*_{train,valid,test}_{rows,indices}.*
data/inputs/mutation/{folds.json,protocol.json,component_inputs_unlabelled.csv}
data/inputs/mutation/features/fold_*_normalization.npz
data/inputs/kinetics/features/{substrate,drfp,protein_pocket}/...
data/inputs/kinetics/alpha_*_row_extra.npz
```

The archive includes the small specificity reference banks, reaction
fingerprints, kinetics descriptor banks, mutation fold row tables and fold
normalization files. It does **not** include the large per-pair Pairformer/ESMC feature
caches. To reproduce the specificity and kinetics results from raw sequences,
generate the features first with
`code/scripts/data_preprocessing/prepare_pairs.py` (see `--help`).
Feature generation requires the upstream Protenix and ESMC-600M weights, CCD
resources, P2Rank/Java and a CUDA GPU.

The download steps use `unzip`; if it is not installed, use
`python -m zipfile -e <archive> <target>`.

## Quick start

A small precomputed specificity example is included in `demo/specificity`; it
requires the specificity checkpoint from [Model checkpoints](#model-checkpoints).

```bash
python -B scripts/predict_specificity.py \
  --csv demo/specificity/input.csv \
  --embedding-dir demo/specificity/pairformer \
  --json-dir demo/specificity/json \
  --drfp demo/specificity/reaction_fingerprints.pkl \
  --protein-pocket-bank demo/specificity/protein_pocket.npz \
  --reference-drfp demo/specificity/reference_fingerprints.pkl \
  --train-reactions demo/specificity/train_reactions.csv \
  --output runs/demo_specificity.csv --device cpu
```

The demo uses five enzyme candidates for one reaction. It ships with a
200-reaction reference subset for the DRFP gate, so its scores are illustrative
rather than the full paper metric; the expected output is in
`demo/specificity/expected.csv`.

## Reproduce the paper results

### Mutation

The mutation fusion is directly reproducible from the released component
scores:

```bash
python -B code/scripts/mutation/score_library.py \
  --input data/inputs/mutation/component_inputs_unlabelled.csv \
  --output runs/mutation_fusion.csv
```

This writes the final within-library scores and reproduces
`data/results/mutation/predictions.csv` exactly.

### Specificity and kinetics

After generating the Pairformer/ESMC features, the final checkpoints can be
evaluated with the released inputs:

```bash
python -B scripts/reproduce.py --task all --output runs/paper --gpu 0
```

The script also accepts `--task specificity`, `--task kinetics` or
`--task mutation`; use a new output directory for each run. The script checks the resulting predictions and metrics against
`data/results/`.

## Train

### Specificity base classifier

The base classifier is trained on the 50w enzyme–reaction table with the
compressed Pairformer representation:

```bash
python -B code/scripts/specificity/train_classifier.py \
  --config configs/specificity_pretrain.yaml
```

The config points to the raw table and to the generated feature directories.
Feature generation uses `code/scripts/data_preprocessing/prepare_pairs.py`;
see `--help` for the arguments.

### Specificity final head

The published final stage trains a new alpha-representation head on top of the
frozen classifier:

```bash
python -B scripts/train_specificity.py --output runs/specificity_train --gpu 0
```

### Kinetics

Kinetics trains a supervised classifier and a `task_moe_gate` head for each of
the three seeds:

```bash
python -B code/scripts/kinetics/train_final.py \
  --seed 1234 --inputs data/inputs/kinetics \
  --sources data/inputs/kinetics/alpha_sources.json \
  --substrate-fp data/inputs/kinetics/features/substrate/substrate_fp_ecfp4_2048.pkl \
  --output runs/kinetics_seed1234
```

Repeat for seeds 2026 and 3407. The final schedule is at most 60 epochs with
validation patience 15.

### Mutation components

Mutation uses the fixed six-training/one-validation/one-test protein folds in
`data/inputs/mutation/folds.json`:

```bash
python -B code/scripts/mutation/train.py \
  --component original --fold 0 --seed 1234 --output runs/mutation_original_f0_s1234
python -B code/scripts/mutation/train.py \
  --component g_nn_paired_meta --fold 0 --seed 1234 --output runs/mutation_nn_f0_s1234
python -B code/scripts/mutation/train.py \
  --component s_xgb_site_rank --fold 0 --seed 1234 --output runs/mutation_tree_f0_s1234
python -B code/scripts/mutation/train.py \
  --component g_ridge --fold 0 --output runs/mutation_ridge_f0
```

The 15 components are listed in `configs/mutation.json`: 13 stochastic
components use seeds 1234, 2026 and 3407 and two ridge components are
deterministic, giving 328 fits across the eight folds. The released
`data/checkpoints/mutation_final.pt` bundles these fits for inference.

## Inference on new data

Raw sequence/reaction preparation uses
`code/scripts/data_preprocessing/prepare_pairs.py`; see `--help` for the
arguments and required upstream resources. The supplied
entry points accept new IDs and do not read target labels:

```bash
python -B scripts/predict_specificity.py --csv new/input_table.csv \
  --embedding-dir new/pairformer --json-dir new/json \
  --drfp new/reaction_fingerprints.pkl --protein-pocket-bank new/protein_pocket_mixed.npz \
  --output runs/new_specificity.csv

python -B scripts/predict_kinetics.py --csv new/kinetics.csv --sources new/sources.json \
  --substrate-fp new/descriptors/substrate_fingerprints.pkl \
  --row-extra new/descriptors/row_extra.npz --output runs/new_kinetics.csv

python -B code/scripts/mutation/predict.py --inputs new/mutation_features \
  --fold 0 --output runs/new_mutation.csv
```

The predictor loads `data/checkpoints/mutation_final.pt` by default; pass
`--checkpoint` to use another bundle.

A new mutation library must choose a released fold explicitly. Submit the
complete candidate library for each `DMS_id`: fusion scores are within-library
ranks, not calibrated absolute activity values.

## Released results

| Task | Released result |
|---|---|
| Specificity (Enzyme-405) | Top-10 success **61.69%**; DCG@10 **0.5587** |
| Kinetics (kcat + Km test) | 3-seed mean prediction; metrics in `data/results/kinetics/` |
| Mutation (8 ProteinGym assays) | macro SCC **0.523** over 41,165 variants |

The full per-task tables, including per-seed and per-protein metrics, are in
`data/results/`.

## Released scope

This is the author-prepared release. The released checkpoints correspond to the
reported results; re-running stochastic training
may produce numerically different weights and metrics. The released task tables
and feature-generation code are provided so that the features can be regenerated
from sequences and reactions. The release does not bundle the upstream
Protenix/ESMC foundation weights.

## License

AlphaEnzyme code, model weights and released data are provided for
non-commercial research use under a custom research license based on the
Creative Commons Attribution-NonCommercial 4.0 International license, with
additional no-drug-development and output-use restrictions. **No commercial
use of the model or data.** See [`LICENSE`](LICENSE) for the full terms.

Third-party components retain their original licenses: Protenix (Apache-2.0),
OpenFold (Apache-2.0), TorchDrug (Apache-2.0), P2Rank (MIT), and ESM/ESMC (upstream
model terms). See [`THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) and
the license texts in [`docs/licenses/`](docs/licenses/).

## Citation

Citation information will be added upon publication.
