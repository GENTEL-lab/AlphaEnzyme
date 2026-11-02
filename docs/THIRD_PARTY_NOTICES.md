# Third-party notices

AlphaEnzyme code, model weights and released data are provided for
non-commercial research use under a custom research license based on the
Creative Commons Attribution-NonCommercial 4.0 International license, with
additional no-drug-development and output-use restrictions (`LICENSE`).
Third-party components retain their own licenses; the root license does not
relicense them.

## Code and runtimes

- **Protenix** (`code/third_party/protenix_runtime`): Copyright 2024 ByteDance
  and/or its affiliates, Apache License 2.0. The license text is in
  `docs/licenses/Protenix_APACHE_LICENSE.txt`. Protenix foundation weights are
  external prerequisites and are not bundled.
- **OpenFold**: Apache License 2.0. The dropout compatibility module in
  `code/alphaenzyme/utils/dropout.py` is adapted from OpenFold; the license is
  included in `docs/licenses/OpenFold_APACHE_LICENSE.txt`.
- **TorchDrug**: Apache License 2.0. The `code/alphaenzyme/core/core.py`
  registry/configuration utilities are derived from TorchDrug; the license is
  included in `docs/licenses/TorchDrug_APACHE_LICENSE.txt`.
- **P2Rank 2.5.1**: MIT license, included in
  `docs/licenses/P2Rank_LICENSE.txt`. P2Rank and Java are external
  prerequisites for structure-based pocket extraction.
- PyTorch, RDKit, DRFP, XGBoost, scikit-learn, pandas and other installed
  packages retain their own licenses.

## Foundation models

The tested runtime uses `esm==3.2.1.post1` and ESMC-600M representations. The
license notice shipped with that package identifies ESMC-600M under the
Cambrian Non-Commercial License Agreement; the notice and agreement are
preserved in `docs/licenses/`. Current upstream ESM/ESMC pages may use
different terms. The ESMC foundation weights are **not** bundled; users obtain
them through the upstream loader and must comply with the applicable terms.
Models and representations derived from ESMC are provided for non-commercial
research use.

## Data sources

- **RHEA / Enzyme-405**: the specificity tables are derived from RHEA reaction
  and enzyme annotations. RHEA is distributed under CC BY 4.0; attribution is
  retained in the table identifiers (RHEA_ID, MASTER_ID, DIRECTION).
- **CatPred-DB**: the kinetics tables are derived from the public CatPred-DB
  collection. Source identifiers are retained in the `dataset` and
  `feature_source` columns.
- **ProteinGym**: the mutation tables cover eight ProteinGym DMS assays. Assay
  identifiers are retained in `DMS_id`; users should also follow the terms of
  the original assay publications.
- **EnzymeCAGE**: the Enzyme-405 evaluation protocol and the enzyme-pocket
  representation follow the EnzymeCAGE release from the same laboratory.

The release contains AlphaEnzyme predictions and result tables. Source
identifiers are retained for traceability; no new rights over third-party
measurements are claimed.
