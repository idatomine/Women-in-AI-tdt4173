"""The margin in neighbouring hours (BRIEFING §4.1 #3, EDA §6.2 "context for stickiness").

Decisions are sticky (start costs), so whether a unit runs at hour t depends on the
margin around t, not only at t. Uses the scaled upstream margin from `wv_margin`.
"""

import numpy as np
import pandas as pd

from src.data import HOURS, upstream_wv, week_prices

SHIFTS = [1, 3, 6, 12, 24]
WINDOWS = [3, 6, 12, 24]


def _shift(values: np.ndarray, hours: int) -> np.ndarray:
    """Value `hours` later (negative: earlier) within the case, repeating the edge value."""
    index = np.clip(np.arange(HOURS) + hours, 0, HOURS - 1)
    return values[..., index]


def build(data):
    prices = week_prices(data)
    scale = np.abs(prices).mean(axis=1, keepdims=True)[:, None, :] + 1
    margin = (prices[:, None, :] - upstream_wv(data, 7)[:, :, None]) / scale  # (case, unit, hour)

    features = {}
    for h in SHIFTS:
        features[f"margin_lag{h}"] = _shift(margin, -h)
        features[f"margin_lead{h}"] = _shift(margin, h)

    flat = margin.reshape(-1, HOURS).T  # (hour, case × unit) for pandas rolling
    for w in WINDOWS:
        rolled = pd.DataFrame(flat).rolling(w, center=True, min_periods=1).mean().to_numpy().T
        features[f"margin_roll{w}"] = rolled.reshape(margin.shape)

    # Rank of the hour's price within its own day of the case (0 = cheapest, 1 = dearest)
    days = prices.reshape(len(prices), HOURS // 24, 24)
    features["price_rank_day"] = (days.argsort(axis=2).argsort(axis=2) / 23).reshape(len(prices), 1, HOURS)
    return features
