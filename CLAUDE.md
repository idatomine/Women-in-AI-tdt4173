# CLAUDE.md

TDT4173 (NTNU) Kaggle project with SINTEF Energy Research. The task is to predict the hourly
on/off unit commitment of 14 hydropower generating units (Tokke–Vinje system) over 168-hour
cases. **Read `docs/BRIEFING.md` before doing modelling work.** **Current task: final picks, reproducibility and report, see BRIEFING §7.** It holds the verified data
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
- `README.md`: how the harness works, written for the team. Keep it in sync when the
  harness changes.
- `src/`: the experiment harness.
  - `data.py`: all 3647 cases (train first, then test) and the input lookups.
  - `cv.py`: folds, embargo, official scoring and the acceptance rule. **Frozen.**
  - `features/`: one module per feature group, cached to `features/` (not in git).
  - `models.py`: model wrappers; add XGBoost etc. here.
  - `run.py`: runs a config on folds A–D and logs it.
  - `submit.py`: trains on 2015–2022 and writes a checked submission to `submissions/`.
- `Short_notebook_1.ipynb` / `Short_notebook_2.ipynb`: the course deliverables (exp 029 /
  exp 030), **generated** from `src/` by `tools/build_notebooks.py`. Never edit them by
  hand; rebuild them after changing any code they use.
- `Report.ipynb`: the report deliverable (graders don't run it, so save it with outputs).
  Currently a disposition (📝 cells) plus complete figure/table code, run from the repo root
  on the cached features and `experiments/` predictions. Edit it directly. Running it also
  exports every numbered figure/table to `report/figures/` and `report/tables/`
  (`show_figure` / `show_table`).
- `report/`: the LaTeX report (`main.tex`, one file per section in `sections/`, with the
  disposition in yellow boxes). The text is now written there, not in Report.ipynb.
- `configs/`: one YAML per experiment (feature groups on/off, model settings).
- `experiments/journal.md`: one row per run, auto-appended; Decision and Public LB filled
  by hand. `experiments/results.csv` is the machine-readable log.
- Run: `.venv/bin/python -m src.run configs/<exp>.yaml [--folds C] [--holdout]`, then
  `.venv/bin/python -m src.submit configs/<exp>.yaml`; or `experiments/run_experiments.ipynb`.
- `analysis/`: EDA notebook and throwaway baseline scripts (`common.py` holds their shared
  loading/split/training code), not the pipeline.
- `research/01–04`: literature, Kaggle/peer, domain/SHOP, and autonomous-loop research,
  with sources.
- `docs/BRIEFING.md`: the synthesis and handoff.

## Hard rules
1. **Validation is by year with a ≥ 7-day embargo**: folds A–D validate on 2020, 2021,
   2022 and 2019, and 2018 is a locked holdout (BRIEFING §5). Never use a random K-fold
   over cases, because neighbouring cases overlap 6/7 days.
2. Use the **upstream** water value at **case start + 7 days** as the end-of-horizon value
   (the SHOP input), and compare it directly to price in EUR/MWh. Subtracting the
   downstream value instead makes the margin much weaker (BRIEFING §3).
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
8. **Never overwrite the raw data** (`data/`, `sample_submission.csv`,
   `prediction_mapping.csv`). To change a data file, make a copy and edit the copy.
   Write derived or cleaned data to new files outside `data/`.
