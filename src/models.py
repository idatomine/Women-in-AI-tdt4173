"""Model training on the long format (one row per case × unit × hour).

Model types (`model: type:` in a config): `lightgbm`, `xgboost`, `random_forest`,
`logistic`, `mlp` (the last three to compare predictor types; report §5), or `blend` (a weighted
average of the raw probabilities of several member models, each with its own model
config under `members:`; global weights, never per unit, BRIEFING §4.2). Add new types to
`_train`, `_predict_one` and `_gain`; the rest of the harness doesn't need to change.
With `per_unit: true`, one model is trained per unit (a list of 14 models) instead of one
pooled model with the unit as a feature.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import KERNEL_DIR


N_JOBS = 4  # the grading PC has at most 4 cores; a fixed thread count keeps results reproducible


def case_weights(model_cfg: dict) -> np.ndarray | None:
    """Training weight per labelled case, or None for equal weights.

    `slow_solver: {seconds: s, weight: w}` gives cases whose SHOP run took ≥ s seconds the
    weight w: slow runs cluster near a time limit and may have suboptimal labels (BRIEFING
    §2.3). total_calculation_time is used only as a training weight, never as a feature.
    """
    slow = model_cfg.get("slow_solver")
    if slow is None:
        return None
    seconds = pd.read_csv(KERNEL_DIR / "Unit_commitment_decisions.csv",
                          usecols=["total_calculation_time"])["total_calculation_time"].to_numpy()
    return np.where(seconds >= slow["seconds"], slow["weight"], 1.0)


def _train(model_cfg: dict, X: np.ndarray, y: np.ndarray, names: list[str], categorical: list[str],
           weight: np.ndarray | None = None):
    seed = model_cfg.get("seed", 0)
    if model_cfg["type"] == "lightgbm":
        # Fixed seed and thread count plus deterministic=True / force_row_wise=True make two
        # runs give byte-identical predictions (needed for the short notebooks)
        params = {"random_state": seed, "n_jobs": N_JOBS, "deterministic": True,
                  "force_row_wise": True, **model_cfg["params"]}
        model = lgb.LGBMClassifier(verbose=-1, **params)
        model.fit(X, y, sample_weight=weight, feature_name=names, categorical_feature=categorical)
        return model
    if model_cfg["type"] == "xgboost":
        # The unit ID is used as an ordinary numeric feature here (no categorical support
        # for plain numpy input); trees can still split on it.
        params = {"random_state": seed, "n_jobs": N_JOBS, **model_cfg["params"]}
        model = xgb.XGBClassifier(tree_method="hist", importance_type="gain", **params)
        model.fit(X, y, sample_weight=weight)
        return model
    if model_cfg["type"] == "random_forest":
        # Bagged trees (no boosting); sklearn handles the coupling features' NaN natively.
        # The unit ID is numeric, as for XGBoost.
        params = {"random_state": seed, "n_jobs": N_JOBS, **model_cfg["params"]}
        model = RandomForestClassifier(**params)
        model.fit(X, y, sample_weight=weight)
        return model
    if model_cfg["type"] in ("logistic", "mlp"):
        # Linear model and neural network need scaled inputs without NaN: the unit ID is
        # one-hot encoded, missing values (partner features of non-Tokke/Vinje units) become
        # 0 plus a missing indicator, and every column is standardised.
        unit = [names.index(c) for c in categorical]
        rest = [j for j in range(X.shape[1]) if j not in unit]
        prep = ColumnTransformer([
            ("unit", OneHotEncoder(handle_unknown="ignore"), unit),
            ("num", make_pipeline(SimpleImputer(strategy="constant", fill_value=0, add_indicator=True),
                                  StandardScaler()), rest)])
        if model_cfg["type"] == "logistic":
            estimator = LogisticRegression(**model_cfg["params"])
        else:
            estimator = MLPClassifier(random_state=seed, **model_cfg["params"])
        model = Pipeline([("prep", prep), ("model", estimator)])
        model.fit(X, y)
        return model
    raise ValueError(f"unknown model type: {model_cfg['type']}")


def _predict_one(model, X: np.ndarray) -> np.ndarray:
    if isinstance(model, lgb.LGBMClassifier):
        return model.booster_.predict(X)
    return model.predict_proba(X)[:, 1]


def _gain(model) -> np.ndarray:
    if isinstance(model, lgb.LGBMClassifier):
        gain = model.booster_.feature_importance("gain")
    elif isinstance(model, Pipeline):  # logistic / MLP: no split gain
        return np.full(model.n_features_in_, np.nan)
    else:
        gain = model.feature_importances_
    return gain / max(gain.sum(), 1e-12)


class Blend:
    """Weighted average of member models' probabilities (weights normalised to sum 1)."""

    def __init__(self, members: list[tuple[float, object]]):
        total = sum(w for w, _ in members)
        self.members = [(w / total, m) for w, m in members]


def fit(model_cfg: dict, X: np.ndarray, Y: np.ndarray, case_idx: np.ndarray, names: list[str]):
    """Train on the given cases. X is (case, unit, hour, feature), Y is (case, unit, hour).

    Rows are subsampled to `row_share` of the training rows to fit in memory.
    """
    if model_cfg["type"] == "blend":
        return Blend([(m["weight"], fit(m, X, Y, case_idx, names)) for m in model_cfg["members"]])
    rng = np.random.default_rng(model_cfg.get("seed", 0))
    n_features = X.shape[-1]

    if model_cfg.get("per_unit"):
        models = []
        for u in range(X.shape[1]):
            X_unit = X[case_idx, u].reshape(-1, n_features)
            y_unit = Y[case_idx, u].ravel()
            keep = rng.random(len(y_unit)) < model_cfg["row_share"]
            models.append(_train(model_cfg, X_unit[keep], y_unit[keep], names, categorical=[]))
        return models

    X_train = X[case_idx].reshape(-1, n_features)
    y_train = Y[case_idx].ravel()
    keep = rng.random(len(y_train)) < model_cfg["row_share"]
    categorical = ["unit"] if "unit" in names else []
    weights = case_weights(model_cfg)
    if weights is not None:
        weights = np.broadcast_to(weights[case_idx, None, None], Y[case_idx].shape).ravel()[keep]
    return _train(model_cfg, X_train[keep], y_train[keep], names, categorical, weights)


def predict(model, X: np.ndarray, case_idx: np.ndarray) -> np.ndarray:
    """P(ON) for the given cases, shape (case, unit, hour)."""
    if isinstance(model, Blend):
        return sum(w * predict(m, X, case_idx) for w, m in model.members)
    X_cases = X[case_idx]
    if isinstance(model, list):  # one model per unit
        return np.stack([_predict_one(m, X_cases[:, u].reshape(-1, X.shape[-1])).reshape(len(case_idx), -1)
                         for u, m in enumerate(model)], axis=1)
    return _predict_one(model, X_cases.reshape(-1, X.shape[-1])).reshape(X_cases.shape[:-1])


def importance(model, names: list[str]) -> pd.Series:
    """Share of the total split gain per feature (averaged over units for per-unit models)."""
    if isinstance(model, Blend):
        return sum(w * importance(m, names) for w, m in model.members)
    models = model if isinstance(model, list) else [model]
    return pd.Series(np.mean([_gain(m) for m in models], axis=0), index=names)

