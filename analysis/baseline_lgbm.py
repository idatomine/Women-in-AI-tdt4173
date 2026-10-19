"""EDA baseline v1: pooled LightGBM with generic features.

Long format: one row per case x unit x hour. Throwaway analysis, not the final pipeline;
the scores are in docs/BRIEFING.md §3.

Usage, from the repo root:
    .venv/bin/python analysis/baseline_lgbm.py 2022 2021 2019
"""

import sys
import time

import numpy as np
import pandas as pd

from common import (DOWNSTREAM, HOURS, UPSTREAM, Data, daily_value, fit_and_predict,
                    hourly_window, load_data, micro_auc, per_unit_auc, plant_of)


def case_features(data: Data, prices: np.ndarray) -> dict[str, np.ndarray]:
    """Features shared by all units in a case, each of shape (case, hour)."""
    n_cases = len(prices)
    week_mean = prices.mean(axis=1, keepdims=True)
    week_std = prices.std(axis=1, keepdims=True) + 1e-3
    level = np.abs(week_mean) + 1
    rolling_24h = pd.DataFrame(prices.T).rolling(24, center=True, min_periods=1).mean().to_numpy().T
    hour = np.broadcast_to(np.arange(HOURS), (n_cases, HOURS))

    def per_case(values):
        return np.repeat(values[:, None], HOURS, axis=1)

    return {
        "price": prices,
        "price_z": (prices - week_mean) / week_std,
        "price_rank": prices.argsort(axis=1).argsort(axis=1) / (HOURS - 1),
        "price_rel": prices / level,
        "price_wkmean": np.repeat(week_mean, HOURS, axis=1),
        "price_d24": rolling_24h / level,
        "h": hour,
        "hod": hour % 24,
        "doy": per_case(data.starts.dt.dayofyear.to_numpy()),
        "dow": per_case(data.starts.dt.dayofweek.to_numpy()),
    }


def unit_features(data: Data, unit: str, prices: np.ndarray) -> dict[str, np.ndarray]:
    """Features for one unit, one value per case, each of shape (case,)."""
    n_cases = len(prices)
    week_mean = prices.mean(axis=1)
    starts = data.starts
    ends = starts + pd.Timedelta(days=7)
    up = UPSTREAM[plant_of(unit)]
    down = DOWNSTREAM[plant_of(unit)]
    up_capacity = sum(data.max_vol[r] for r in up)

    wv_up_end = daily_value(data.water_value, ends, up)
    wv_up_start = daily_value(data.water_value, starts, up)
    wv_down_end = daily_value(data.water_value, ends, [down]) if down else np.zeros(n_cases)
    net_wv = wv_up_end - wv_down_end

    up_fill = sum(data.volume.loc[starts, r].to_numpy() for r in up) / up_capacity
    down_fill = data.volume.loc[starts, down].to_numpy() / data.max_vol[down] if down else np.zeros(n_cases)
    # Inflow over the week in Mm³ (m³/s × 3600 s / 1e6), relative to the upstream capacity
    week_inflow = sum(hourly_window(data.inflow[r], starts).sum(axis=1) * 3600 / 1e6 for r in up)

    return {
        "wv_up_end": wv_up_end,
        "wv_up_start": wv_up_start,
        "wv_dn_end": wv_down_end,
        "net_wv": net_wv,
        "net_wv_rel": net_wv / (np.abs(week_mean) + 1),
        "fill": up_fill,
        "fill_dn": down_fill,
        "inflow_wk": week_inflow / up_capacity,
        "wv_trend": (wv_up_end - wv_up_start) / (np.abs(wv_up_start) + 1),
    }


def build_features(data: Data) -> tuple[np.ndarray, list[str]]:
    """Feature array (case, unit, hour, feature) in float32, plus the feature names."""
    prices = hourly_window(data.price, data.starts)
    shared = case_features(data, prices)
    per_unit = {unit: unit_features(data, unit, prices) for unit in data.units}

    names = list(shared) + list(per_unit[data.units[0]]) + ["unit", "p_max", "startcost", "margin"]
    X = np.empty((len(prices), len(data.units), HOURS, len(names)), dtype=np.float32)
    for u, unit in enumerate(data.units):
        generator = data.generators[unit]
        values = {
            **shared,
            **{name: v[:, None] for name, v in per_unit[unit].items()},
            "unit": u,
            "p_max": generator["p_max"],
            "startcost": generator["startcost_const"],
        }
        for j, name in enumerate(names[:-1]):
            X[:, u, :, j] = values[name]

    # Price minus net water value, scaled by the week's price level
    col = {name: j for j, name in enumerate(names)}
    X[..., col["margin"]] = ((X[..., col["price"]] - X[..., col["net_wv"]])
                             / (np.abs(X[..., col["price_wkmean"]]) + 1))
    return X, names


if __name__ == "__main__":
    data = load_data()
    X, names = build_features(data)
    for val_year in [int(a) for a in sys.argv[1:]] or [2022]:
        t0 = time.time()
        val, model, pred = fit_and_predict(X, data, val_year, names.index("unit"), n_estimators=600)
        y_val = data.Y[val]
        print(f"val {val_year}: micro AUC {micro_auc(y_val, pred):.4f}  ({time.time() - t0:.0f}s)")
        print("  per-unit", per_unit_auc(y_val, pred, data.units))
        gain = pd.Series(model.booster_.feature_importance("gain"), index=names)
        print("  top gain", (gain / gain.sum()).sort_values(ascending=False).round(3).head(12).to_dict())
