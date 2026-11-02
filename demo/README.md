# Demo

The specificity demo contains one reaction with five enzyme candidates and the
corresponding precomputed Pairformer, JSON, protein-pocket and DRFP inputs. It
runs on CPU or a single GPU after the specificity checkpoint has been downloaded
to `data/checkpoints/specificity/specificity_final.pt`.

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

The demo's DRFP gate uses a 200-reaction reference subset, so the scores are
illustrative. The expected output is in `demo/specificity/expected.csv`.
