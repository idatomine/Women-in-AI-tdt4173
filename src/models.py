"""Model training on the long format (one row per case × unit × hour).

Model types (`model: type:` in a config): `lightgbm` or `xgboost`. Add new types to
`_train`, `_predict_one` and `_gain`; the rest of the harness doesn't need to change.
With `per_unit: true`, one model is trained per unit (a list of 14 models) instead of one
pooled model with the unit as a feature.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb


def _train(model_cfg: dict, X: np.ndarray, y: np.ndarray, names: list[str], categorical: list[str]):
    if model_cfg["type"] == "lightgbm":
        model = lgb.LGBMClassifier(verbose=-1, **model_cfg["params"])
        model.fit(X, y, feature_name=names, categorical_feature=categorical)
        return model
    if model_cfg["type"] == "xgboost":
        # The unit ID is used as an ordinary numeric feature here (no categorical support
        # for plain numpy input); trees can still split on it.
        model = xgb.XGBClassifier(tree_method="hist", importance_type="gain", n_jobs=-1, **model_cfg["params"])
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
    else:
        gain = model.feature_importances_
    return gain / max(gain.sum(), 1e-12)


def fit(model_cfg: dict, X: np.ndarray, Y: np.ndarray, case_idx: np.ndarray, names: list[str]):
    """Train on the given cases. X is (case, unit, hour, feature), Y is (case, unit, hour).

    Rows are subsampled to `row_share` of the training rows to fit in memory.
    """
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
    return _train(model_cfg, X_train[keep], y_train[keep], names, categorical)


def predict(model, X: np.ndarray, case_idx: np.ndarray) -> np.ndarray:
    """P(ON) for the given cases, shape (case, unit, hour)."""
    X_cases = X[case_idx]
    if isinstance(model, list):  # one model per unit
        return np.stack([_predict_one(m, X_cases[:, u].reshape(-1, X.shape[-1])).reshape(len(case_idx), -1)
                         for u, m in enumerate(model)], axis=1)
    return _predict_one(model, X_cases.reshape(-1, X.shape[-1])).reshape(X_cases.shape[:-1])


def importance(model, names: list[str]) -> pd.Series:
    """Share of the total split gain per feature (averaged over units for per-unit models)."""
    models = model if isinstance(model, list) else [model]
    return pd.Series(np.mean([_gain(m) for m in models], axis=0), index=names)
