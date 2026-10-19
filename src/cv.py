"""Validation folds, scoring and the acceptance rule (BRIEFING §5, §5b).

FROZEN: don't change this file while running experiments. Otherwise the scores in
experiments/journal.md stop being comparable.
"""

import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import ROOT, Data

FOLDS = {"A": 2020, "B": 2021, "C": 2022, "D": 2019}
HOLDOUT_YEAR = 2018  # locked: score only the 2–3 final candidates, once
EMBARGO_DAYS = 7

# Acceptance rule for a change, compared fold by fold with the reference experiment
MIN_MEAN_GAIN = 0.001
NOISE = 0.001  # a fold counts as "not worse" if it drops by at most this
MAX_FOLD_DROP = 0.002

_spec = importlib.util.spec_from_file_location("kaggle_metric", ROOT / "kaggle_metric.py")
kaggle_metric = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kaggle_metric)


def split(data: Data, val_year: int) -> tuple[np.ndarray, np.ndarray]:
    """Train and validation case indices for one leave-one-year-out fold.

    Drops every training case that starts within EMBARGO_DAYS of the validation year, on
    both sides, so no training window overlaps a validation window.
    """
    starts = data.starts[: data.n_train]
    in_val = (starts.dt.year == val_year).to_numpy()
    embargo = pd.Timedelta(days=EMBARGO_DAYS)
    first_val, last_val = starts[in_val].min(), starts[in_val].max()
    clear_of_val = ((starts + embargo <= first_val) | (starts - embargo >= last_val)).to_numpy()
    return np.flatnonzero(~in_val & clear_of_val), np.flatnonzero(in_val)


def to_frame(data: Data, case_idx: np.ndarray, values: np.ndarray) -> pd.DataFrame:
    """(case, unit, hour) values -> DataFrame with `Run No` + the 2352 label columns."""
    frame = pd.DataFrame(values.reshape(len(case_idx), -1), columns=data.label_columns)
    frame.insert(0, "Run No", data.run_no[case_idx])
    return frame


def score(data: Data, case_idx: np.ndarray, pred: np.ndarray) -> float:
    """Micro ROC-AUC with the official kaggle_metric.score."""
    solution = to_frame(data, case_idx, data.Y[case_idx])
    return kaggle_metric.score(solution, to_frame(data, case_idx, pred))


def unit_scores(data: Data, case_idx: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """ROC-AUC per unit (diagnostic only; Kaggle scores all units together)."""
    y = data.Y[case_idx]
    return {unit: roc_auc_score(y[:, u].ravel(), pred[:, u].ravel()) for u, unit in enumerate(data.units)}


def acceptance(new: dict[str, float], reference: dict[str, float]) -> tuple[bool, str]:
    """Apply the BRIEFING §5b rule to two complete sets of fold scores A–D."""
    deltas = {fold: new[fold] - reference[fold] for fold in FOLDS}
    mean_gain = np.mean(list(deltas.values()))
    not_worse = sum(d >= -NOISE for d in deltas.values())
    worst = min(deltas.values())

    problems = []
    if mean_gain < MIN_MEAN_GAIN:
        problems.append(f"mean gain {mean_gain:+.4f} < {MIN_MEAN_GAIN}")
    if not_worse < 3:
        problems.append(f"only {not_worse}/4 folds not worse")
    if worst < -MAX_FOLD_DROP:
        problems.append(f"a fold drops {worst:+.4f}")
    delta_text = " ".join(f"{fold} {d:+.4f}" for fold, d in deltas.items())
    verdict = "passes" if not problems else "fails: " + "; ".join(problems)
    return not problems, f"{verdict} (mean {mean_gain:+.4f}; {delta_text})"
