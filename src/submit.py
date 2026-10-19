"""Train on all labelled cases (2015–2022) and write a checked Kaggle submission.

From the repo root:
    .venv/bin/python -m src.submit configs/exp_001_baseline_v2.yaml

Writes submissions/<name>.csv. Upload it to Kaggle by hand, then add the public score to
the experiment's row in experiments/journal.md.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import cv, models
from src.data import ROOT, load_data
from src.features import build_matrix
from src.run import load_config

SUBMISSIONS_DIR = ROOT / "submissions"


def check_submission(submission: pd.DataFrame) -> None:
    """Format checks from CLAUDE.md rule 7, plus a dry run of the official metric."""
    sample = pd.read_csv(ROOT / "sample_submission.csv")
    if list(submission.columns) != list(sample.columns):
        raise ValueError("columns differ from sample_submission.csv")
    if not np.array_equal(submission["Run No"].to_numpy(), sample["Run No"].to_numpy()):
        raise ValueError("Run No values or order differ from sample_submission.csv")
    values = submission.iloc[:, 1:].to_numpy()
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("predictions must be finite probabilities in [0, 1]")

    # The metric must accept the file; random labels, so the score itself means nothing
    fake_solution = sample.copy()
    fake_solution.iloc[:, 1:] = np.random.default_rng(0).integers(0, 2, size=values.shape)
    cv.kaggle_metric.score(fake_solution, submission)


def make_submission(config_path: str | Path, out_dir: Path = SUBMISSIONS_DIR) -> Path:
    cfg = load_config(config_path)
    print(f"Submission for {cfg['name']}: training on all 2015–2022 cases")
    data = load_data()
    X, names = build_matrix(cfg["features"], data, cfg.get("drop_features", []))

    train_idx = np.arange(data.n_train)
    test_idx = np.arange(data.n_train, data.n_cases)
    model = models.fit(cfg["model"], X, data.Y, train_idx, names)
    submission = cv.to_frame(data, test_idx, models.predict(model, X, test_idx))
    check_submission(submission)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg['name']}.csv"
    submission.to_csv(path, index=False, float_format="%.6f")
    print(f"  checked and written: {path} ({len(submission)} rows)")
    print("  next: upload to Kaggle, then add the public score to experiments/journal.md")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a checked Kaggle submission.")
    parser.add_argument("config", help="path to the experiment's YAML config")
    make_submission(parser.parse_args().config)


if __name__ == "__main__":
    main()
