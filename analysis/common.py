"""Shared code for the EDA baselines: data loading, year holdout and LightGBM training.

Throwaway analysis code, not the final pipeline (see docs/BRIEFING.md §7).
Arrays are indexed (case, unit, hour) or (case, hour), with cases in train-file order.
"""

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = ROOT / "data" / "kernel"
EXTENDED_DIR = ROOT / "data" / "extended"
HOURS = 168

# Water routing per plant, from the `connections` in the topology YAML.
# Upstream: the reservoir(s) the plant takes its water from.
UPSTREAM = {
    "Hogga": ["Bandak"],
    "Tokke": ["Vinjevatn"],
    "Lio": ["Byrtevatn"],
    "Byrte": ["Botnedalsvatn"],
    "Vinje": ["Vaamarvatn"],
    "Haukeli": ["Vatjern"],
    "Songa": ["Songavatn", "Bitdalsvatn"],
    "Kjela": ["Foersvatn"],
    "Vesle Kjela": ["Kjelavatn"],
}
# Downstream: the reservoir the plant releases into (None = leaves the modelled system).
DOWNSTREAM = {
    "Hogga": None,
    "Tokke": "Bandak",
    "Lio": None,
    "Byrte": "Byrtevatn",
    "Vinje": "Vinjevatn",
    "Haukeli": "Vinjevatn",
    "Songa": "Totak",
    "Kjela": "Hyljelihyl",
    "Vesle Kjela": "Foersvatn",
}

LGBM_PARAMS = dict(
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=200,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)


@dataclass
class Data:
    units: list[str]  # e.g. "Tokke_G1", in label-column order
    starts: pd.Series  # case start times (UTC)
    Y: np.ndarray  # labels (case, unit, hour)
    price: pd.Series  # day-ahead price, EUR/MWh (hourly up to 2025-09-30)
    inflow: pd.DataFrame  # hourly inflow per reservoir, m³/s
    volume: pd.DataFrame  # daily volume per reservoir, Mm³
    water_value: pd.DataFrame  # daily water value per reservoir, EUR/MWh
    max_vol: dict[str, float]  # reservoir -> max volume, Mm³
    generators: dict[str, dict]  # unit -> YAML generator entry (p_max, startcost_const, ...)


def plant_of(unit: str) -> str:
    """'Vesle Kjela_G1' -> 'Vesle Kjela'."""
    return unit.rsplit("_", 1)[0]


def _read_timeseries(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["date"]).set_index("date")


def load_data() -> Data:
    """Load the train labels and all input tables (train cases only, no test)."""
    labels = pd.read_csv(KERNEL_DIR / "Unit_commitment_decisions.csv", parse_dates=["starttime"])
    label_cols = [c for c in labels.columns if c.startswith("result_committed_")]
    units = list(dict.fromkeys(
        c.removeprefix("result_committed_").rsplit("_t", 1)[0] for c in label_cols
    ))
    topology = yaml.safe_load((EXTENDED_DIR / "Tokke_Vinje_topology.yaml").read_text())["model"]

    return Data(
        units=units,
        starts=labels["starttime"].dt.tz_localize("UTC"),
        Y=labels[label_cols].to_numpy().reshape(len(labels), len(units), HOURS),
        price=_read_timeseries(KERNEL_DIR / "Historical_day_ahead_price_2015_2025.csv").iloc[:, 0],
        inflow=_read_timeseries(KERNEL_DIR / "Historical_inflow_1958_2025.csv"),
        volume=_read_timeseries(KERNEL_DIR / "Historical_volume_2015_2024.csv"),
        water_value=_read_timeseries(KERNEL_DIR / "Synthetic_water_value_2015_2024.csv"),
        max_vol={name: r["max_vol"] for name, r in topology["reservoir"].items()},
        generators=topology["generator"],
    )


def hourly_window(series: pd.Series, starts: pd.Series, offset_hours: int = 0) -> np.ndarray:
    """The 168 hourly values from each start (+ offset), shape (case, hour).

    Assumes an hourly index without gaps, which holds for 2015-2024 (EDA §5.4).
    """
    first = series.index.get_indexer(starts + pd.Timedelta(hours=offset_hours))
    if (first < 0).any():
        raise KeyError("a case start is missing from the hourly index")
    return series.to_numpy()[first[:, None] + np.arange(HOURS)]


def daily_value(table: pd.DataFrame, days: pd.Series, columns: list[str]) -> np.ndarray:
    """Mean of `columns` on each day in `days`, shape (case,)."""
    return table.loc[days, columns].mean(axis=1).to_numpy()


def train_mask(starts: pd.Series, val_year: int, embargo_days: int = 7) -> np.ndarray:
    """Training cases for a leave-one-year-out fold.

    Drops the validation year and every case that starts within `embargo_days` of it on
    either side, so no training window overlaps a validation window (BRIEFING §5).
    """
    in_val = (starts.dt.year == val_year).to_numpy()
    embargo = pd.Timedelta(days=embargo_days)
    first_val, last_val = starts[in_val].min(), starts[in_val].max()
    clear_of_val = ((starts + embargo <= first_val) | (starts - embargo >= last_val)).to_numpy()
    return ~in_val & clear_of_val


def fit_and_predict(X: np.ndarray, data: Data, val_year: int, unit_col: int,
                    n_estimators: int, row_share: float = 0.35, seed: int = 0):
    """Train pooled LightGBM on one year holdout.

    X has shape (case, unit, hour, feature). Training rows are subsampled to `row_share` to
    fit in memory. Returns the validation case mask, the model and the validation
    predictions (case, unit, hour).
    """
    val = (data.starts.dt.year == val_year).to_numpy()
    train = train_mask(data.starts, val_year)
    n_features = X.shape[-1]

    X_train = X[train].reshape(-1, n_features)
    y_train = data.Y[train].ravel()
    keep = np.random.default_rng(seed).random(len(y_train)) < row_share

    model = lgb.LGBMClassifier(n_estimators=n_estimators, **LGBM_PARAMS)
    model.fit(X_train[keep], y_train[keep], categorical_feature=[unit_col])
    pred = model.predict_proba(X[val].reshape(-1, n_features))[:, 1]
    return val, model, pred.reshape(data.Y[val].shape)


def micro_auc(y: np.ndarray, pred: np.ndarray) -> float:
    """Micro ROC-AUC over all units and hours (same as kaggle_metric.score)."""
    return roc_auc_score(y.ravel(), pred.ravel())


def per_unit_auc(y: np.ndarray, pred: np.ndarray, units: list[str]) -> dict[str, float]:
    return {unit: round(micro_auc(y[:, i], pred[:, i]), 3) for i, unit in enumerate(units)}
