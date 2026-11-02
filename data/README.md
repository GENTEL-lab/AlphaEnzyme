# Data

`results/` contains the final prediction and metric tables for the three
tasks. It is tracked in the repository.

The released task tables and model weights are distributed separately:

- Download `AlphaEnzyme_data.zip` and unpack it into this directory
  (`unzip AlphaEnzyme_data.zip -d data/`), which creates `inputs/`.
- Download the released checkpoints from Hugging Face and unpack them into this
  directory (`unzip AlphaEnzyme_checkpoints.zip -d data/`), which creates
  `checkpoints/`.

The data archive contains the released task tables and the small input
assets (specificity reference banks and DRFP tables, kinetics descriptor
banks, mutation fold row tables and fold normalization). It does not contain
the large per-pair Pairformer/ESMC feature caches; feature generation uses
`code/scripts/data_preprocessing/prepare_pairs.py` (see `--help`).
