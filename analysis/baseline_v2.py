"""EDA baseline v2: v1 plus physics features (price vs the +7 d water value, scarcity).

Throwaway analysis, not the final pipeline; the scores are in docs/BRIEFING.md §3.
Also prints a level-only oracle: the AUC if we knew each unit-week's true ON share, but
nothing about which hours within the week.

Usage, from the repo root:
    .venv/bin/python analysis/baseline_v2.py 2022 2021
"""

import sys
import time

import numpy as np
import pandas as pd

from baseline_lgbm import build_features
from common import (DOWNSTREAM, HOURS, UPSTREAM, Data, daily_value, fit_and_predict,
                    hourly_window, load_data, micro_auc, per_unit_auc, plant_of)


def physics_features(data: Data) -> tuple[np.ndarray, list[str]]:
    """Physics features (case, unit, hour, feature) in float32, plus the feature names."""
    starts = data.starts
    ends = starts + pd.Timedelta(days=7)
    prices = hourly_window(data.price, starts)
    next_week = hourly_window(data.price, starts, offset_hours=7 * 24)
    week_mean = prices.mean(axis=1)
    next_week_mean = next_week.mean(axis=1)
    scale = np.abs(prices).mean(axis=1, keepdims=True) + 1

    def per_case(values):
        return np.repeat(values[:, None], HOURS, axis=1)

    names = ["margin_up", "price_over_wv", "margin_net", "scarcity_start", "scarcity_end",
             "profitable_share", "positive_margin", "next_week_price"]
    per_unit = []
    for unit in data.units:
        up = UPSTREAM[plant_of(unit)]
        down = DOWNSTREAM[plant_of(unit)]
        wv_up_end = daily_value(data.water_value, ends, up)
        wv_up_start = daily_value(data.water_value, starts, up)
        wv_down_end = daily_value(data.water_value, ends, [down]) if down else np.zeros(len(starts))
        margin = prices - wv_up_end[:, None]

        features = {
            "margin_up": margin / scale,
            "price_over_wv": prices / (np.abs(wv_up_end)[:, None] + 1),
            "margin_net": (prices - (wv_up_end - wv_down_end)[:, None]) / scale,
            # Scarcity coefficient c in water value ≈ c × mean price (BRIEFING §2.1)
            "scarcity_start": per_case(wv_up_start / (np.abs(week_mean) + 1e-3)),
            "scarcity_end": per_case(wv_up_end / (np.abs(next_week_mean) + 1e-3)),
            "profitable_share": per_case((margin > 0).mean(axis=1)),
            "positive_margin": per_case(np.clip(margin, 0, None).sum(axis=1) / scale[:, 0] / HOURS),
            "next_week_price": per_case(next_week_mean / (np.abs(week_mean) + 1)),
        }
        per_unit.append(np.stack([features[n] for n in names], axis=-1))
    return np.stack(per_unit, axis=1).astype(np.float32), names


if __name__ == "__main__":
    data = load_data()
    X_v1, names_v1 = build_features(data)
    X_physics, _ = physics_features(data)
    X = np.concatenate([X_v1, X_physics], axis=-1)
    del X_v1, X_physics

    for val_year in [int(a) for a in sys.argv[1:]] or [2022]:
        t0 = time.time()
        val, _, pred = fit_and_predict(X, data, val_year, names_v1.index("unit"), n_estimators=800)
        y_val = data.Y[val]
        weekly_share = np.broadcast_to(y_val.mean(axis=2, keepdims=True), y_val.shape)
        print(f"val {val_year}: v2 AUC {micro_auc(y_val, pred):.4f}  "
              f"| oracle-level-only {micro_auc(y_val, weekly_share):.4f}  ({time.time() - t0:.0f}s)")
        print("  per-unit", per_unit_auc(y_val, pred, data.units))
