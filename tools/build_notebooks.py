"""Build Short_notebook_1.ipynb and Short_notebook_2.ipynb from the harness code.

The course requires all code inside the notebooks (no separate files). Instead of keeping a
second copy by hand, this script pastes the modules from src/ that the final configs need
into notebook cells, so the notebooks always match the code that made the submissions:
- `from src...` imports are removed (everything lives in one namespace),
- each feature group's `build` is renamed `build_<group>`,
- paths point to the raw data next to the notebook, and the LP runs in-process,
- the feature cache and the experiment logging are left out.

From the repo root:
    .venv/bin/python tools/build_notebooks.py           # improved 039 / 040 pair
    .venv/bin/python tools/build_notebooks.py --legacy  # previous 029 / 030 pair
"""

import argparse
import ast
import csv
from pathlib import Path

import nbformat
import yaml

ROOT = Path(__file__).resolve().parent.parent

LEGACY_NOTEBOOKS = {
    "Short_notebook_1.ipynb": "configs/exp_029_main_river.yaml",
    "Short_notebook_2.ipynb": "configs/exp_030_xgb_main_river.yaml",
}
NOTEBOOKS = {
    "Short_notebook_1.ipynb": "configs/exp_039_dispatch_capacity.yaml",
    "Short_notebook_2.ipynb": "configs/exp_040_xgb_dispatch_capacity.yaml",
}
TEAM_HEADER = """**Kaggle team:** TODO "[<group number>] <team name>"

| Name | Student ID |
|---|---|
| TODO | TODO |
"""
PUBLIC_LB = {"exp_029_main_river": "0.99414", "exp_030_xgb_main_river": "0.99429"}
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
    """A markdown cell with the module docstring and a code cell with the module's code."""
    source = path.read_text()
    tree = ast.parse(source)
    doc = ast.get_docstring(tree) or ""
    body = source.splitlines()[tree.body[0].end_lineno:] if doc else source.splitlines()
    code = []
    for line in body:
        if line.startswith("from src") or line.startswith(drop_lines):
            continue
        if rename_build and line.startswith("def build("):
            line = line.replace("def build(", f"def {rename_build}(")
        code.append(line)
    text = "\n".join(code).strip() + "\n"
    for old, new in (replace or {}).items():
        if old not in text:
            raise ValueError(f"{path.name}: expected text not found: {old!r}")
        text = text.replace(old, new)
    return [nbformat.v4.new_markdown_cell(f"### {title}\n\n{doc}"), nbformat.v4.new_code_cell(text)]


def build_notebook(config_path: str) -> nbformat.NotebookNode:
    cfg = yaml.safe_load((ROOT / config_path).read_text())
    name, model_cfg = cfg["name"], cfg["model"]
    model_name = {"lightgbm": "LightGBM", "xgboost": "XGBoost"}[model_cfg["type"]]
    cells = [
        nbformat.v4.new_markdown_cell(
            f"# TDT4173 — SINTEF unit commitment: `{name}` ({model_name})\n\n{TEAM_HEADER}\n"
            f"Reproduces prediction **{name}** (public LB {PUBLIC_LB.get(name, 'not yet submitted')}, "
            f"local CV {cv_text(name)} as the mean AUC over the leave-one-year-out folds 2019–2022).\n\n"
            "Pipeline: load the raw data → physics features (water-value margins, scarcity, a "
            "per-plant dynamic program, Tokke–Vinje coupling) → **a relaxed LP of the whole "
            "watercourse per case** (our approximation of SINTEF's SHOP optimiser) → one pooled "
            f"{model_name} model over all units and hours, trained on all 2015–2022 cases → "
            f"`submission_{name}.csv`.\n\n"
            "**How to run:** place the competition files next to this notebook (`data/kernel/`, "
            "`data/extended/`, `prediction_mapping.csv`, `sample_submission.csv`; or change "
            "`DATA_DIR` below), install `requirements.txt` (Python 3.12), then run all cells. "
            "CPU only, at most 4 threads; runtime and memory depend on the hardware. Training is "
            "deterministic: two runs give identical files. How the hyperparameters were chosen is "
            "described in the report."),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n\n"
            "DATA_DIR = Path(\".\")  # folder that holds data/, prediction_mapping.csv and sample_submission.csv\n"
            "ROOT = DATA_DIR\n"
            "KERNEL_DIR = DATA_DIR / \"data\" / \"kernel\"\n"
            "EXTENDED_DIR = DATA_DIR / \"data\" / \"extended\"\n"
            f"OUTPUT = Path(\"submission_{name}.csv\")"),
        nbformat.v4.new_markdown_cell("## 1. Data"),
        *module_cells(ROOT / "src/data.py", "Loading and shared lookups",
                      drop_lines=("ROOT =", "KERNEL_DIR =", "EXTENDED_DIR =")),
        nbformat.v4.new_markdown_cell(
            "## 2. Features\n\nEach group builds arrays that broadcast to (case, unit, hour) from "
            "the inputs SHOP had for that case only: prices, inflows and constraints inside the "
            "week, the start-day volume, and the water values on the start day and start + 7 days. "
            "`load_group` builds a group once and keeps it in memory as float32, like the "
            "harness's feature cache (some groups reuse another group's output)."),
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
        cells += module_cells(ROOT / f"src/features/{group}.py", f"Feature group `{group}`",
                              rename_build=f"build_{group}",
                              replace={"workers=6": "workers=1"} if group == "relaxed_lp_best" else None)

    cells += [
        nbformat.v4.new_markdown_cell("## 4. Feature matrix"),
        nbformat.v4.new_code_cell(
            f"FEATURE_GROUPS = {cfg['features']!r}\n"
            f"DROP_FEATURES = {cfg.get('drop_features', [])!r}  # raw EUR levels: regime-dependent (2022 vs 2023–24)\n\n\n"
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
            "X, names = build_matrix(data)\n"
            "_groups.clear()  # release cached groups before model training\n"
            "print(X.shape, len(names), \"features\")"),
        nbformat.v4.new_markdown_cell("## 5. Model"),
        *module_cells(ROOT / "src/models.py", "Model wrapper"),
        nbformat.v4.new_markdown_cell(
            "## 6. Train on all 2015–2022 cases and write the submission\n\n"
            "Settings below reproduce the selected experiment; its validation mean is recorded in the introduction."),
        nbformat.v4.new_code_cell(
            f"MODEL = {model_cfg!r}\n\n"
            "train_idx = np.arange(data.n_train)\n"
            "test_idx = np.arange(data.n_train, data.n_cases)\n"
            "model = fit(MODEL, X, data.Y, train_idx, names)\n"
            "pred = predict(model, X, test_idx)\n\n"
            "submission = pd.DataFrame(pred.reshape(len(test_idx), -1), columns=data.label_columns)\n"
            "submission.insert(0, \"Run No\", data.run_no[test_idx])\n\n"
            "# Format checks against sample_submission.csv\n"
            "sample = pd.read_csv(ROOT / \"sample_submission.csv\")\n"
            "assert list(submission.columns) == list(sample.columns)\n"
            "assert (submission[\"Run No\"].to_numpy() == sample[\"Run No\"].to_numpy()).all()\n"
            "values = submission.iloc[:, 1:].to_numpy()\n"
            "assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1\n\n"
            "submission.to_csv(OUTPUT, index=False, float_format=\"%.6f\")\n"
            "print(f\"written: {OUTPUT} ({len(submission)} rows)\")"),
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
