# Modelling continuation — 4 November 2026

User objective: try to improve the score by at least 0.002. All results below are local validation AUC, not Kaggle leaderboard scores.

The original year splits, seven-day embargo, official micro-AUC metric and raw input files were preserved. No 2023–2024 labels, later reservoir volumes or solver runtime features were used.

| Experiment | A: 2020 | B: 2021 | C: 2022 | D: 2019 | Mean | Gain |
|---|---:|---:|---:|---:|---:|---:|
| exp_029_main_river | 0.9867599 | 0.9940679 | 0.9922355 | 0.9918980 | 0.9912403 | +0.0000000 |
| exp_037_terminal_efficiency | 0.9875554 | 0.9944325 | 0.9923592 | 0.9932196 | 0.9918917 | +0.0006513 |
| exp_038_dispatch_context | 0.9889807 | 0.9950722 | 0.9932314 | 0.9941535 | 0.9928594 | +0.0016191 |
| exp_039_dispatch_capacity | 0.9892539 | 0.9952845 | 0.9934676 | 0.9941286 | 0.9930336 | +0.0017933 |

Best completed candidate: `exp_039_dispatch_capacity`. Mean gain: **+0.0017933**. The requested 0.002 gain has not been reached.

## Changes

- Added optional terminal-water valuation at the most efficient production segment. The previous full-load conversion remains the default, preserving old experiments.
- Added weekly and rolling summaries of LP commitment and margin, Tokke–Vinje coupling, and start-cost regret using LP effective margins. These derive only from allowed case inputs.
- Tested a larger LightGBM fit with 60% of training rows and 2,200 trees at learning rate 0.03.
- Reduced memory copies when assembling features and subsampling rows. Regression checks verify identical matrix values, selected rows, labels, weights and ordering.
- Cheap derived feature groups can opt out of redundant disk caching with `CACHE = False`. The dispatch-context group uses this because local disk space is limited.

## Additional checks and rejected ideas

- Efficient terminal valuation improved the LP diagnostic on every sixth training case: mean unit-on AUC 0.9447 → 0.9597; mean margin AUC 0.9739 → 0.9770. These are diagnostics, not final model scores.
- Sixteen production segments gave no clear diagnostic gain over four; not retained.
- A separate weekly-fraction model added only about 0.00005 at the best tested correction strength; not retained.
- Median filtering, probability smoothing and logit smoothing added no meaningful mean gain; not retained.
- The first broad 78-feature context trial was stopped after disk/memory pressure; no validation result was attributed to it.
- The old LP defaults were compared with the original implementation on three cases and were exactly equal. New LP solutions were finite and production remained within capacity.
- Changing labels and reservoir volumes after the selected case start left all optimiser inputs unchanged.
- Both existing short notebooks were regenerated from source, validated with nbformat, and all code cells parsed. They continue to reproduce the original selected models; the report and previous final picks were not rewritten.

## Reproduce

```bash
.venv/bin/python -m src.run configs/exp_039_dispatch_capacity.yaml
.venv/bin/python -m src.submit configs/exp_039_dispatch_capacity.yaml
.venv/bin/python -m unittest discover -s tests -v
```

The submission must be scored on Kaggle to establish a leaderboard gain. No Kaggle CLI, package or credential file was configured in this environment.

## Reserved 2018 check

The selected candidate was scored once on 2018 after selection: **0.9895972**, versus **0.9876971** for exp 029 (gain **+0.0019001**). This check was not used for feature or parameter tuning. Per-fold artifacts are in `experiments/exp_039_dispatch_capacity/scores_all.csv`; unrounded official scores are in `experiments/results.csv`.

## Submission artifact

`submissions/exp_039_dispatch_capacity.csv` was trained on all 2015–2022 cases. It passed the existing submission checker: 725 rows, exact sample columns and Run No order, no NaN or infinite values, probabilities within [0, 1], and a dry run through the official metric. No Kaggle upload was made.

SHA-256: `b2c5da00c8ce09195e9a3943c7f895450a132a91714cd6cc59a95418ae8e2dad`.
