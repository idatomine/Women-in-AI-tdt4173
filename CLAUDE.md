# CLAUDE.md

TDT4173 (NTNU) Kaggle project with SINTEF Energy Research. The task is to predict the hourly
on/off unit commitment of 14 hydropower generating units (Tokke–Vinje system) over 168-hour
cases. **Read `docs/BRIEFING.md` before doing modelling work.** It holds the verified data
facts, the validation protocol, the feature backlog and the pitfalls.

## Environment
- Use the project venv: `.venv/bin/python` (Python 3.12; pandas, numpy, scikit-learn,
  lightgbm, pyyaml, plotly). The base conda env is broken (NumPy 1/2 mix), and system
  `python3` lacks pandas.
- The machine is an Apple M3 with 16 GB RAM. The long-format matrix (2922 × 14 × 168 ≈ 6.9M
  rows) must be kept as float32, and rows subsampled while iterating.

## Layout
- `data/kernel/`: price (hourly UTC; 15-min from 2025-09-30), inflow (hourly), volume and
  water value (daily), labels `Unit_commitment_decisions.csv` (2922 train cases).
- `data/extended/`: min-flow and min-volume constraints, SHOP topology YAML and PDF diagram.
- `prediction_mapping.csv`: the 725 test cases (Run No 2923–3647, 2023-01-01 → 2024-12-25).
- `sample_submission.csv`: the required column order (`Run No` + 2352 label columns).
- `kaggle_metric.py`: the exact metric (micro ROC-AUC). Always evaluate with `score()`.
- `analysis/`: throwaway EDA and baseline scripts, not the pipeline.
- `research/01–04`: literature, Kaggle/peer, domain/SHOP, and autonomous-loop research,
  with sources.
- `docs/BRIEFING.md`: the synthesis and handoff.

## Hard rules
1. **Validation is by year with a ≥ 7-day embargo** (validation years 2020, 2021, 2022).
   Never use a random K-fold over cases, because neighbouring cases overlap 6/7 days.
2. Use the water value at **case start + 7 days** as the end-of-horizon value (the SHOP
   input), and compare it directly to price in EUR/MWh.
3. **Never use these as features:**
   - `total_calculation_time`
   - reservoir volumes after the case start day
   - any external data that could contain 2023–24 labels
4. **Use relative price features.** 2022 averaged 211 EUR/MWh, while test (2023–24) has
   negative prices.
5. Micro-AUC ranks all units together. Blend raw probabilities with global weights. Never
   rank-normalise per unit, and never use per-unit class weights.
6. Label column order is unit-major (`<Plant>_<Unit>_t0..t167` for each unit in
   `sample_submission.csv` order). The plant name `Vesle Kjela` contains a space.
7. Before any Kaggle upload, check the submission against `sample_submission.csv`: 725 rows,
   identical columns, values in [0, 1], no NaN.
