# CrediCap

CrediCap is a credibility-aware framework for image-caption evaluation.  This
directory contains only the final method and the code needed for training and
evaluation; exploratory variants, logs, cached features, checkpoints, model
weights, and datasets are intentionally excluded.

## Source layout

- `initial_scoring.py`: reference-aware continuous initial score.
- `reference_credibility.py`: adaptive reference weighting.
- `evidence_decomposition.py`: consensus--dissent evidence decomposition.
- `semantic_verification.py`: seven-dimensional E/A/T/R/C/U/Q verification.
- `score_correction.py`: direction--magnitude score correction.
- `prepare_features.py`: feature construction for the correction stage.
- `pipeline.py`: self-check, training, and evaluation entry point.
- `evaluate.py` and `summarize_results.py`: reported metrics and summaries.

## Entry points

Run commands from `/home/xgd/FLEUR_reproduction/01_source_code/code`:

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m credicap selfcheck --seed 2026
python -m credicap.semantic_verification --help
python -m credicap.prepare_features --help
python -m credicap --help
```

Existing datasets, model weights, feature caches, and checkpoints remain under
the original project paths and are not duplicated here.
