"""Build Short_notebook_1.ipynb and Short_notebook_2.ipynb from the harness code.

The course requires all code inside the notebooks (no separate files). Instead of keeping a
second copy by hand, this script pastes the modules from src/ that the final configs need
into notebook cells, so the notebooks always match the code that made the submissions:
- `from src...` imports are removed (everything lives in one namespace),
- each feature group's `build` is renamed `build_<group>`,
- paths point to the raw data found next to the notebook, and the LP runs in-process,
- the feature cache and the experiment logging are left out,
- the module docstrings are replaced by the neutral texts in MARKDOWN (for the graders),
- progress lines with the elapsed time are printed (prints only, no computation changes).

From the repo root:
    .venv/bin/python tools/build_notebooks.py           # final 046 / 047 pair
    .venv/bin/python tools/build_notebooks.py --legacy  # fallback 038 / 041 pair
"""

import argparse
import ast
import csv
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parent.parent

LEGACY_NOTEBOOKS = {
    "Short_notebook_1.ipynb": "configs/exp_038_dispatch_context.yaml",
    "Short_notebook_2.ipynb": "configs/exp_041_xgb_dispatch_context.yaml",
}
NOTEBOOKS = {
    "Short_notebook_1.ipynb": "configs/exp_046_no_next_week_price.yaml",
    "Short_notebook_2.ipynb": "configs/exp_047_xgb_no_next_week_price.yaml",
}
TEAM_HEADER = """**Kaggle team:** [36] Women in AI

| Name | Student ID |
|---|---|
| Ida Tomine Berg | 592890 |
| Ida Nikoline Rogne | 115536 |
| Ingrid Kapstad Østgulen | 104288 |
"""
PUBLIC_LB = {
    "exp_029_main_river": "0.99414",
    "exp_030_xgb_main_river": "0.99429",
    "exp_038_dispatch_context": "0.99554",
    "exp_041_xgb_dispatch_context": "0.99552",
    "exp_046_no_next_week_price": "0.99558",
    "exp_047_xgb_no_next_week_price": "0.99555",
}

# Markdown shown above each pasted module, instead of its docstring (which refers to the
# team's internal docs). Keyed by the path under src/.
MARKDOWN = {
    "data.py":
        "Loads the raw competition files. Cases are indexed 0 … 3646 in Run No order: the 2922 "
        "training cases (2015–2022) first, then the 725 test cases (2023-01-01 → 2024-12-25). "
        "Labels exist for the training cases only.",
    "features/price_shape.py":
        "Price shape within the case week, shared by all units. The raw EUR/MWh levels `price` "
        "and `price_wkmean` are built here but dropped from the model (`DROP_FEATURES`), because "
        "price levels differ strongly between 2022 and the test years 2023–24.",
    "features/calendar.py":
        "Calendar: hour within the case, hour of day (UTC), and the start day's position.",
    "features/water_value.py":
        "Water values per unit: upstream on the start day and at start + 7 days, downstream at "
        "start + 7 days, and the net value. These raw EUR levels are dropped from the model "
        "(`DROP_FEATURES`); the relative margins below carry the signal.",
    "features/hydro.py":
        "Reservoirs per unit: fill on the start day (upstream and downstream) and the week's "
        "inflow. Only the start-day volume is used, since later volumes are outcomes of the "
        "plan that is being predicted.",
    "features/unit_static.py":
        "Unit statics: unit ID (used as a categorical feature), p_max and start cost.",
    "features/wv_margin.py":
        "Price minus water value per unit and hour, the core decision signal. `margin_up` "
        "(price − upstream water value at start + 7 days) is the strong one; the net versions "
        "also subtract the downstream value and are much weaker on their own.",
    "features/scarcity.py":
        "Scarcity coefficient c in water value ≈ c × mean price, per unit. Two variants that "
        "use the price of the week after the case are built but dropped (`DROP_FEATURES`), "
        "since SHOP did not have that price.",
    "features/case_economics.py":
        "Week-level economics per unit: how much of the week is worth running.",
    "features/relaxed_dp.py":
        "Per-plant dynamic programme over the case week: the most profitable path of \"number "
        "of units on\" over the 168 hours, where k units earn k × p_max × (price − upstream "
        "water value at +7 days) per hour and each start or stop costs the YAML start cost; "
        "reservoir limits are ignored (the LP below adds them). Units within a plant are "
        "switched on in a fixed priority order, and the DP is solved with the start cost ×1 "
        "and ×10, since the effective start cost may be on another scale than in the YAML.",
    "features/tokke_vinje.py":
        "Tokke–Vinje coupling. Vinje releases into the tiny Vinjevatn (9.4 Mm³), which Tokke "
        "empties in about 20 h, so the two plants run together (label correlation 0.6–0.8). "
        "Each Tokke unit gets Vinje's DP schedule and margin and vice versa; other units get "
        "NaN (missing). Reuses the `relaxed_dp` group.",
    "features/relaxed_lp.py":
        "One linear programme per case over the whole watercourse (all plants, reservoirs, "
        "tunnels and rivers from the topology YAML), hourly over the 168 hours, with the "
        "on/off status relaxed to a continuous level. It maximises the income from selling "
        "power plus the value of the water left after 168 h, subject to the water balance of "
        "every reservoir in every hour. Its solution (production, reservoir fill) and duals "
        "(the effective hourly value of water) become features.",
    "features/relaxed_lp_shop2.py":
        "The LP variant used by the model (features prefixed `lps2_`): soft min-flow "
        "constraints, relaxed unit commitment (p_min, start/stop costs), piecewise-linear "
        "production curves, the YAML flow costs, and `main_river`: water left in Hyljelihyl "
        "(and everything upstream) is valued along the main river r_Hyljelihyl → Venemo → "
        "Totak → Vinje, not the bypass that skips Vinje. On every 6th training case this "
        "raised the mean AUC over units of the LP's own on/off output from 0.8885 to 0.9447.",
    "features/relaxed_lp_best.py":
        "The LP variant used by the model (features prefixed `lpb_`): soft min-flow "
        "constraints, relaxed unit commitment (p_min, start/stop costs), piecewise-linear "
        "production curves, the YAML flow costs, and `main_river`: water left in Hyljelihyl "
        "(and everything upstream) is valued along the main river r_Hyljelihyl → Venemo → "
        "Totak → Vinje, not the bypass that skips Vinje. The water left after 168 h is "
        "converted to energy with each unit's most efficient production segment instead of "
        "its full-load average efficiency. This changes only the end-of-week value of the "
        "water, not the production inside the week.",
    "features/lp_dispatch_context.py":
        "Context from the LP above over the whole week, per unit and hour: rolling 6 h and "
        "24 h means and the weekly mean of the LP's unit-on level and margin, the weekly "
        "minimum and maximum of both, the share of hours with a positive margin, the weekly "
        "mean LP production, Tokke's and Vinje's LP production and margin (for every unit), "
        "and a start-cost regret: the best weekly profit (from the LP margin, with each "
        "switch costing the unit's start cost) of a path that is on in that hour, minus the "
        "best one that is off, scaled by p_max and the week's price level. Reuses the "
        "`relaxed_lp_best` group.",
    "models.py":
        "Trains one pooled model on the long format (one row per case × unit × hour, with the "
        "unit ID as a categorical feature), on a random `row_share` of the training rows to "
        "fit in memory. The wrapper also holds the other model types we compared; this "
        "notebook uses only the one set in `MODEL` below.",
}

# Comments in the pasted code that refer to the team's internal docs, reworded (text only,
# no code changes). Keyed by the path under src/; applied to every build of that module.
NEUTRAL_CODE = {
    "features/relaxed_lp.py": {
        "      what the labels follow (BRIEFING §3); the default.":
        "      what the labels follow; the default.",
    },
    "models.py": {
        "    weight w: slow runs cluster near a time limit and may have suboptimal labels (BRIEFING\n"
        "    §2.3). total_calculation_time":
        "    weight w: slow runs cluster near a time limit and may have suboptimal labels.\n"
        "    total_calculation_time",
    },
}

# Progress line after each LP solve (print only).
LP_DONE = {
    "relaxed_lp_shop2": {"    return lp_features(data, lp, prefix=\"lps2\")":
                         "    stage(f\"LP solved for {data.n_cases} cases\")\n"
                         "    return lp_features(data, lp, prefix=\"lps2\")"},
    "relaxed_lp_best": {"    return lp_features(data, lp, prefix='lpb')":
                        "    stage(f\"LP solved for {data.n_cases} cases\")\n"
                        "    return lp_features(data, lp, prefix='lpb')"},
}


def cv_text(name: str) -> str:
    """Use the latest complete measured result, including after a holdout run."""
    value = "not yet measured"
    with (ROOT / "experiments/results.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["name"] == name and all(row.get(f) for f in ("A", "B", "C", "D")):
                value = f"{sum(float(row[f]) for f in ('A', 'B', 'C', 'D')) / 4:.7f}"
    return value


def module_cells(path: Path, title: str, rename_build: str | None = None,
                 replace: dict[str, str] | None = None, drop_lines: tuple[str, ...] = ()) -> list:
    """A markdown cell with the neutral text from MARKDOWN and a code cell with the module's code."""
    source = path.read_text()
    tree = ast.parse(source)
    doc = ast.get_docstring(tree) or ""
    key = path.relative_to(ROOT / "src").as_posix()
    text_md = MARKDOWN[key]
    replace = {**NEUTRAL_CODE.get(key, {}), **(replace or {})}
    body = source.splitlines()[tree.body[0].end_lineno:] if doc else source.splitlines()
    code = []
    for line in body:
        if line.startswith("from src") or line.startswith(drop_lines):
            continue
        if rename_build and line.startswith("def build("):
            line = line.replace("def build(", f"def {rename_build}(")
        code.append(line)
    text = "\n".join(code).strip() + "\n"
    for old, new in replace.items():
        if old not in text:
            raise ValueError(f"{path.name}: expected text not found: {old!r}")
        text = text.replace(old, new)
    if "BRIEFING" in text:
        raise ValueError(f"{path.name}: pasted code still mentions BRIEFING; add it to NEUTRAL_CODE")
    return [nbformat.v4.new_markdown_cell(f"### {title}\n\n{text_md}"), nbformat.v4.new_code_cell(text)]


def build_notebook(config_path: str) -> nbformat.NotebookNode:
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    name, model_cfg = cfg["name"], cfg["model"]
    model_name = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}[model_cfg["type"]]
    cells = [
        nbformat.v4.new_markdown_cell(
            f"# TDT4173 — SINTEF unit commitment: `{name}` ({model_name})\n\n{TEAM_HEADER}\n"
            f"Reproduces our selected Kaggle prediction **{name}** (public LB {PUBLIC_LB.get(name, 'not yet submitted')}, "
            f"local CV {cv_text(name)} as the mean AUC over the leave-one-year-out folds 2019–2022).\n\n"
            "Pipeline: load the raw data → physics features (water-value margins, scarcity, a "
            "per-plant dynamic program, Tokke–Vinje coupling) → **a relaxed LP of the whole "
            "watercourse per case** (our approximation of SINTEF's SHOP optimiser) → one pooled "
            f"{model_name} model over all units and hours, trained on all 2015–2022 cases → "
            f"`submission_{name}.csv`.\n\n"
            "**How to run:** install `requirements.txt` (Python 3.12) and run all cells. The "
            "notebook looks for the competition files (`data/kernel/`, `data/extended/`, "
            "`prediction_mapping.csv`, `sample_submission.csv`) in the notebook's folder, its "
            "parent folder, or any direct subfolder, and uses the first one that holds "
            "`data/kernel/Unit_commitment_decisions.csv`. The submission is written next to the "
            "notebook. CPU only, at most 4 threads, with a progress line per stage. Training is "
            "deterministic: two runs give identical files. How the hyperparameters were chosen is "
            "described in the report."),
        nbformat.v4.new_code_cell(
            "import time\n"
            "from pathlib import Path\n\n"
            "_T0 = time.time()\n\n\n"
            "def stage(message):\n"
            "    \"\"\"One progress line with the elapsed time.\"\"\"\n"
            "    print(f\"[{time.time() - _T0:7.1f} s] {message}\", flush=True)\n\n\n"
            "def find_data_dir():\n"
            "    \"\"\"First of ., .. and the direct subfolders of . that holds the competition files.\"\"\"\n"
            "    marker = \"data/kernel/Unit_commitment_decisions.csv\"\n"
            "    required = [\"data/kernel/\", \"data/extended/\", \"prediction_mapping.csv\", \"sample_submission.csv\"]\n"
            "    candidates = [Path(\".\"), Path(\"..\")] + sorted(p for p in Path(\".\").iterdir() if p.is_dir())\n"
            "    for folder in candidates:\n"
            "        if (folder / marker).is_file():\n"
            "            missing = [r for r in required\n"
            "                       if not ((folder / r).is_dir() if r.endswith(\"/\") else (folder / r).is_file())]\n"
            "            if missing:\n"
            "                raise FileNotFoundError(f\"{folder.resolve()} has {marker} but is missing: {missing}\")\n"
            "            return folder.resolve()\n"
            "    raise FileNotFoundError(\n"
            "        f\"no folder with {marker} (and {required}) found; looked in: \"\n"
            "        + \", \".join(str(p.resolve()) for p in candidates))\n\n\n"
            "ROOT = find_data_dir()  # folder that holds data/, prediction_mapping.csv and sample_submission.csv\n"
            "KERNEL_DIR = ROOT / \"data\" / \"kernel\"\n"
            "EXTENDED_DIR = ROOT / \"data\" / \"extended\"\n"
            f"OUTPUT = Path(\"submission_{name}.csv\")\n"
            "print(f\"data folder: {ROOT}\")"),
        nbformat.v4.new_markdown_cell("## 1. Data"),
        *module_cells(ROOT / "src/data.py", "Loading and shared lookups",
                      drop_lines=("ROOT =", "KERNEL_DIR =", "EXTENDED_DIR =")),
        nbformat.v4.new_markdown_cell(
            "## 2. Features\n\nEach group builds arrays that broadcast to (case, unit, hour) from "
            "the inputs SHOP had for that case only: prices, inflows and constraints inside the "
            "week, the start-day volume, and the water values on the start day and start + 7 days. "
            "`load_group` builds each group once and keeps it in memory as float32 (some groups "
            "reuse another group's output)."),
        nbformat.v4.new_code_cell(
            "_groups = {}\n\n\n"
            "def load_group(group, data):\n"
            "    \"\"\"Features of one group, shape (case, unit, hour, feature), float32.\"\"\"\n"
            "    if group not in _groups:\n"
            "        print(f\"building feature group '{group}'\", flush=True)\n"
            "        features = globals()[f\"build_{group}\"](data)\n"
            "        shape = (data.n_cases, len(data.units), HOURS)\n"
            "        array = np.empty(shape + (len(features),), dtype=np.float32)\n"
            "        for j, values in enumerate(features.values()):\n"
            "            array[..., j] = np.broadcast_to(values, shape)\n"
            "        _groups[group] = array, list(features)\n"
            "        stage(f\"feature group '{group}' built ({len(features)} features)\")\n"
            "    return _groups[group]"),
    ]
    lp_included = False
    for group in cfg["features"]:
        if group in ("relaxed_lp_shop2", "relaxed_lp_best") and not lp_included:
            lp_included = True
            cells += [nbformat.v4.new_markdown_cell("## 3. Relaxed LP of the watercourse")]
            cells += module_cells(ROOT / "src/features/relaxed_lp.py", "LP model (relaxed_lp)",
                                  rename_build="build_relaxed_lp",
                                  replace={"WORKERS = None  # LP worker processes; None = all cores but one, 1 = in-process (notebooks)":
                                           "WORKERS = 1  # in-process: worker processes can't import functions defined in a notebook"})
        replace = dict(LP_DONE.get(group, {}))
        if group == "relaxed_lp_best":
            replace["workers=6"] = "workers=1"
        cells += module_cells(ROOT / f"src/features/{group}.py", f"Feature group `{group}`",
                              rename_build=f"build_{group}", replace=replace or None)

    cells += [
        nbformat.v4.new_markdown_cell("## 4. Feature matrix"),
        nbformat.v4.new_code_cell(
            f"FEATURE_GROUPS = {cfg['features']!r}\n"
            f"DROP_FEATURES = {cfg.get('drop_features', [])!r}  # raw EUR levels (regime-dependent, 2022 vs 2023–24) and the two next-week price features (unknown to SHOP)\n\n\n"
            "def build_matrix(data):\n"
            "    \"\"\"All groups side by side, shape (case, unit, hour, feature), float32.\"\"\"\n"
            "    metadata, names = [], []\n"
            "    for group in FEATURE_GROUPS:\n"
            "        array, group_names = load_group(group, data)\n"
            "        keep = [j for j, name in enumerate(group_names) if name not in DROP_FEATURES]\n"
            "        metadata.append((array, keep))\n"
            "        names += [group_names[j] for j in keep]\n"
            "    X = np.empty((data.n_cases, len(data.units), HOURS, len(names)), dtype=np.float32)\n"
            "    column = 0\n"
            "    for array, keep in metadata:\n"
            "        for j in keep:\n"
            "            X[..., column] = array[..., j]\n"
            "            column += 1\n"
            "    return X, names\n\n\n"
            "data = load_data()\n"
            "stage(f\"data loaded: {data.n_train} train + {data.n_cases - data.n_train} test cases\")\n"
            "X, names = build_matrix(data)\n"
            "_groups.clear()  # release cached groups before model training\n"
            "stage(f\"feature matrix built: {X.shape}, {len(names)} features\")"),
        nbformat.v4.new_markdown_cell("## 5. Model"),
        *module_cells(ROOT / "src/models.py", "Model wrapper"),
        nbformat.v4.new_markdown_cell(
            "## 6. Train on all 2015–2022 cases and write the submission\n\n"
            "Hyperparameters were chosen on the leave-one-year-out folds (see the report)."),
        nbformat.v4.new_code_cell(
            f"MODEL = {model_cfg!r}\n\n"
            "train_idx = np.arange(data.n_train)\n"
            "test_idx = np.arange(data.n_train, data.n_cases)\n"
            "model = fit(MODEL, X, data.Y, train_idx, names)\n"
            "stage(\"model trained\")\n"
            "pred = predict(model, X, test_idx)\n"
            "stage(\"test cases predicted\")\n\n"
            "submission = pd.DataFrame(pred.reshape(len(test_idx), -1), columns=data.label_columns)\n"
            "submission.insert(0, \"Run No\", data.run_no[test_idx])\n\n"
            "# Format checks against sample_submission.csv\n"
            "sample = pd.read_csv(ROOT / \"sample_submission.csv\")\n"
            "assert list(submission.columns) == list(sample.columns)\n"
            "assert (submission[\"Run No\"].to_numpy() == sample[\"Run No\"].to_numpy()).all()\n"
            "values = submission.iloc[:, 1:].to_numpy()\n"
            "assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1\n\n"
            "submission.to_csv(OUTPUT, index=False, float_format=\"%.6f\")\n"
            "stage(f\"written: {OUTPUT.resolve()} ({len(submission)} rows)\")"),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return notebook


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", action="store_true", help="rebuild the original exp 029/030 notebooks")
    args = parser.parse_args()
    for filename, config in (LEGACY_NOTEBOOKS if args.legacy else NOTEBOOKS).items():
        nbformat.write(build_notebook(config), ROOT / filename)
        print(f"written: {filename} (from {config})")


if __name__ == "__main__":
    main()
