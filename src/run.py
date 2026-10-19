"""Run one experiment on the validation folds and log the result.

From the repo root:
    .venv/bin/python -m src.run configs/exp_001_baseline_v2.yaml
    .venv/bin/python -m src.run configs/exp_002_example.yaml --folds C   # quick check, one fold
    .venv/bin/python -m src.run configs/exp_00X_final.yaml --holdout     # locked 2018, final picks only

From a notebook: experiments/run_experiments.ipynb.

Every run is appended to experiments/journal.md (for people) and experiments/results.csv
(used for comparisons). Predictions, per-unit scores and feature importance are saved in
experiments/<name>/ (not in git).
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src import cv, models
from src.data import ROOT, load_data
from src.features import build_matrix

EXPERIMENTS_DIR = ROOT / "experiments"
JOURNAL = EXPERIMENTS_DIR / "journal.md"
RESULTS = EXPERIMENTS_DIR / "results.csv"
SCORE_COLUMNS = ["A", "B", "C", "D", "holdout_2018"]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    return yaml.safe_load(path.read_text())


def run_experiment(config_path: str | Path, folds: list[str] | None = None,
                   holdout: bool = False, log: bool = True) -> pd.DataFrame:
    """Train and score one config on folds A–D (or a subset, or the locked 2018 holdout).

    Returns a table with one row per fold: the year, the official AUC and the AUC per unit.
    """
    cfg = load_config(config_path)
    print(f"Experiment {cfg['name']}: {cfg['change']}")
    data = load_data()
    X, names = build_matrix(cfg["features"], data, cfg.get("drop_features", []))
    print(f"  {len(names)} features from {len(cfg['features'])} groups")

    years = {"holdout_2018": cv.HOLDOUT_YEAR} if holdout else {f: cv.FOLDS[f] for f in folds or cv.FOLDS}
    out_dir = EXPERIMENTS_DIR / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, importances = [], []
    for fold, year in years.items():
        t0 = time.time()
        train_idx, val_idx = cv.split(data, year)
        model = models.fit(cfg["model"], X, data.Y, train_idx, names)
        pred = models.predict(model, X, val_idx)
        auc = cv.score(data, val_idx, pred)
        print(f"  fold {fold} ({year}): AUC {auc:.4f}  ({time.time() - t0:.0f} s)")

        rows.append({"fold": fold, "year": year, "AUC": auc, **cv.unit_scores(data, val_idx, pred)})
        importances.append(models.importance(model, names).rename(fold))
        np.save(out_dir / f"pred_{fold}.npy", pred.astype(np.float32))

    table = pd.DataFrame(rows).set_index("fold")
    scores = table["AUC"].to_dict()
    if set(cv.FOLDS) <= scores.keys():
        print(f"  mean A–D: {np.mean([scores[f] for f in cv.FOLDS]):.4f}")

    verdict = _compare(cfg, scores)
    if verdict:
        print(f"  vs {cfg['compare_to']}: {verdict}")

    table.round(4).to_csv(out_dir / "scores.csv")
    pd.concat(importances, axis=1).assign(mean=lambda d: d.mean(axis=1)).sort_values(
        "mean", ascending=False).round(4).to_csv(out_dir / "importance.csv")
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    if log:
        _log(cfg, scores, verdict)
    return table


def _latest_scores(name: str) -> dict[str, float] | None:
    """The most recent complete A–D result of an experiment, from results.csv."""
    if not RESULTS.exists():
        return None
    results = pd.read_csv(RESULTS)
    complete = results[(results["name"] == name) & results[list(cv.FOLDS)].notna().all(axis=1)]
    return None if complete.empty else complete.iloc[-1][list(cv.FOLDS)].to_dict()


def _compare(cfg: dict, scores: dict[str, float]) -> str:
    """Acceptance-rule verdict against `compare_to`, if both have all four folds."""
    reference_name = cfg.get("compare_to")
    if not reference_name or not set(cv.FOLDS) <= scores.keys():
        return ""
    reference = _latest_scores(reference_name)
    if reference is None:
        return f"no complete A–D result for {reference_name} in results.csv"
    return cv.acceptance(scores, reference)[1]


def _log(cfg: dict, scores: dict[str, float], verdict: str) -> None:
    date = datetime.now().strftime("%Y-%m-%d %H:%M")
    complete = set(cv.FOLDS) <= scores.keys()
    mean = np.mean([scores[f] for f in cv.FOLDS]) if complete else None

    row = {"date": date, "name": cfg["name"], **{c: scores.get(c) for c in SCORE_COLUMNS}, "mean": mean}
    pd.DataFrame([row]).to_csv(RESULTS, mode="a", header=not RESULTS.exists(), index=False)

    def cell(value):
        return f"{value:.4f}" if value is not None else "—"

    rule = verdict.split(" (")[0] if verdict else "—"
    journal_row = [date, cfg["name"], cfg["change"], *(cell(scores.get(c)) for c in SCORE_COLUMNS),
                   cell(mean), cfg.get("compare_to") or "—", rule, "", ""]
    with JOURNAL.open("a") as f:
        f.write("| " + " | ".join(journal_row) + " |\n")
    print(f"  logged to {JOURNAL.name}: fill in Decision (and Public LB if submitted)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment on the validation folds.")
    parser.add_argument("config", help="path to the experiment's YAML config")
    parser.add_argument("--folds", nargs="+", choices=list(cv.FOLDS), help="run only these folds")
    parser.add_argument("--holdout", action="store_true", help="score the locked 2018 holdout instead")
    parser.add_argument("--no-log", action="store_true", help="don't write to journal.md / results.csv")
    args = parser.parse_args()
    run_experiment(args.config, folds=args.folds, holdout=args.holdout, log=not args.no_log)


if __name__ == "__main__":
    main()
