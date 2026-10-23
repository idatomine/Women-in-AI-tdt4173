# Women in AI: TDT4173 unit-commitment prediction

Kaggle project with SINTEF Energy Research: predict the hourly on/off decisions of 14
hydropower units (Tokke–Vinje) for 168-hour weeks. The plan, data facts and rules are in
[docs/BRIEFING.md](docs/BRIEFING.md); the EDA is in
[analysis/01_eda.ipynb](analysis/01_eda.ipynb).

## How experiments work

```
configs/exp_001_baseline_v2.yaml      ← you edit this: which features, which model settings
        │
        ▼
src/run.py                            ← you start this (terminal or notebook)
        │
        ├─ src/data.py                 loads the CSVs (train + test cases)
        ├─ src/features/…              computes the features (cached, so only once)
        ├─ src/cv.py                   splits into the 4 validation years, scores with the Kaggle metric
        ├─ src/models.py               trains LightGBM
        │
        ▼
experiments/journal.md                ← a result row is written here; you add the decision
```

### Run an experiment

From the repo root:

```bash
.venv/bin/python -m src.run configs/exp_001_baseline_v2.yaml            # all 4 folds, ~4 min
.venv/bin/python -m src.run configs/exp_001_baseline_v2.yaml --folds C  # one fold, quick check
```

Or open [experiments/run_experiments.ipynb](experiments/run_experiments.ipynb) and run the
cells; it calls the same code.

It prints the AUC for each validation year: **A = 2020, B = 2021, C = 2022, D = 2019**
(each with a 7-day buffer so training and validation weeks never overlap) and the mean.
The row is added to [experiments/journal.md](experiments/journal.md) automatically. Fill in
**Decision** (keep/drop and why) and, if you submitted it, **Public LB**.

### Make a new experiment

1. Copy the config: `configs/exp_002_<short_name>.yaml`.
2. Change `name` and `change` (one line describing what you changed).
3. Set `compare_to:` to the experiment you're trying to improve on.
4. Change **one** thing, usually one line under `features:` or a model setting.
5. Run it. The output says whether the change passes the acceptance rule (BRIEFING §5b):
   mean gain ≥ 0.001, at least 3 of 4 years not worse, no year dropping more than 0.002.

### Submit to Kaggle

```bash
.venv/bin/python -m src.submit configs/exp_001_baseline_v2.yaml
```

Trains on all of 2015–2022, checks the file (725 rows, correct columns, values in [0, 1])
and writes `submissions/<name>.csv`. Upload it to Kaggle by hand and add the score to the
journal. Submit sparingly (≤ 2 per day); the private leaderboard decides the grade.

## What each file does

**Files you work with**

| File | What it is |
|---|---|
| `configs/*.yaml` | One file per experiment: the feature groups and LightGBM settings. |
| `experiments/journal.md` | The results log: one row per run, auto-appended. |
| `experiments/run_experiments.ipynb` | Optional: run experiments from a notebook instead of the terminal. Make a personal copy to avoid merge conflicts. |

**The machinery in `src/`** (normally no need to open these)

| File | What it does |
|---|---|
| `run.py` | Starts an experiment: reads the config, trains on each validation year, prints scores, writes the journal row. |
| `data.py` | Loads all CSV files, for the train cases (2015–2022) and the test cases (2023–24). |
| `cv.py` | The validation rules: which years validate, the 7-day buffer, scoring with `kaggle_metric.py`, and the acceptance rule. **Don't change it during experiments**, or scores stop being comparable. |
| `models.py` | Trains LightGBM and predicts. Other model types (XGBoost) go here. Training is deterministic by default (seed from the config, 4 threads), so two runs give identical predictions. |
| `submit.py` | Trains on all years and writes a checked Kaggle file. |
| `__init__.py` | Empty marker so Python treats `src/` as a package. |

**Feature groups in `src/features/`**: one small file per theme, so a group can be
switched on or off with one line in a config. The first eight hold baseline_v2's 31
features.

| Group | Features |
|---|---|
| `price_shape` | The week's prices: raw, z-score, rank, relative to the week's level, 24 h average |
| `calendar` | Hour within the week, hour of day, day of year, weekday |
| `water_value` | Upstream/downstream water values at start and +7 days, net value, trend |
| `hydro` | Reservoir fill on the start day (upstream, downstream), the week's inflow |
| `unit_static` | Unit ID, p_max, start cost |
| `wv_margin` | Price minus water value: the core signal |
| `scarcity` | How scarce the water is (water value ÷ price level) |
| `case_economics` | Share of the week where running pays off, total positive margin |

Added during the experiments (see the journal for which ones were kept):

| Group | Features |
|---|---|
| `margin_context` | The margin in neighbouring hours, rolling means, within-day price rank |
| `profitable_blocks` | Length of the profitable run around each hour |
| `hydro_pressure` | The week's inflow ÷ free space, upstream and downstream (spill pressure) |
| `relaxed_dp` | Per-plant DP with start costs: optimal on/off path, regret, units on (BRIEFING §4.5 Level 1) |
| `tokke_vinje` | Each of Tokke/Vinje sees the other plant's DP schedule and margin |
| `cascade` | Each plant sees the DP schedule of the plants feeding its reservoir |
| `relaxed_lp` | LP of the whole watercourse per case: production, effective water value, margin, fill (BRIEFING §4.5 Level 2). Takes ~2 min to build on all cores. |
| `relaxed_lp_minflow` | The same LP plus the six min-flow constraints (features `lpm_*`); used instead of `relaxed_lp` from exp 017 |
| `relaxed_lp_uc` | The min-flow LP plus a relaxed unit commitment: units on, p_min, start/stop costs (features `lpu_*`); used from exp 021 |
| `relaxed_lp_shop` | The commitment LP plus YAML efficiency curves, head losses and SHOP's spill/bypass costs (features `lps_*`); used from exp 024 |
| `storage_upstream` | Fill of the real storage lake behind tiny pass-through reservoirs (Haukeli → Langeidvatn) |
| `relaxed_lp_shop2` | `relaxed_lp_shop` with the Hyljelihyl routing fix (`main_river`; features `lps2_*`); used from exp 029, **final model** |
| `lp_extra` | Extra outputs of the same LP: unclipped units-on gap, plant share on, joint fill of tunnel-joined lakes (`lpx_*`); tested in exp 031, not used |

Config options besides `features:`:
- `drop_features:` removes single features by name from the selected groups.
- `model: type: xgboost` uses XGBoost instead of LightGBM (see `configs/exp_012_xgboost.yaml`).
- `model: per_unit: true` trains one model per unit instead of one pooled model.
- `model: type: blend` averages several member models (`members:`, each a model config
  with a `weight`).
- `model: slow_solver: {seconds: 900, weight: 0.3}` down-weights cases from slow SHOP runs
  in training (exp 032; not used).

To add a group, create `src/features/<name>.py` with a `build(data)` function (see the
existing files and the docstring in `src/features/__init__.py`), then list it in a config.
Features are cached in `features/` and rebuilt automatically when the group's code changes.

**Generated and not in git** (rebuilt from the raw data): `features/` (≈ 1 GB cache),
`experiments/<name>/` (per-run predictions, per-unit scores, feature importance) and
`submissions/`.

## Other folders

- `analysis/`: the EDA notebook and the early baseline scripts (kept for reference).
- `research/`: literature and domain research behind the briefing.
- `docs/`: the briefing and the course/task PDFs.

## Final delivery

The course requires two short notebooks with all code inside (no imports from `src/`).
At the end, the relevant parts of `src/` are copied into `Short_notebook_1.ipynb` and
`Short_notebook_2.ipynb`. See `docs/TDT4173_Project_Submission_and_Grading.pdf`.

## Setup

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On macOS with conda, LightGBM also needs OpenMP: `conda install -c conda-forge llvm-openmp`.

## Final submissions and short notebooks

The two selected Kaggle predictions are exp 029 (LightGBM) and exp 030 (XGBoost), both on
the same features. `Short_notebook_1.ipynb` and `Short_notebook_2.ipynb` reproduce them from
the raw data with all code inline, as the course requires. **Don't edit the notebooks by
hand:** they are generated from `src/` and the two configs by

```bash
.venv/bin/python tools/build_notebooks.py
```

so rerun it after any change to the code they use.
