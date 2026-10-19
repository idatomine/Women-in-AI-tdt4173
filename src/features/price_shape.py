"""Price shape within the case week, shared by all units.

`price` and `price_wkmean` are raw EUR/MWh. The briefing (§2.4) prefers relative features;
they are kept here because baseline_v2 has them. Test dropping them as an experiment.
"""

import numpy as np
import pandas as pd

from src.data import HOURS, week_prices


def build(data):
    prices = week_prices(data)
    week_mean = prices.mean(axis=1, keepdims=True)
    week_std = prices.std(axis=1, keepdims=True) + 1e-3
    level = np.abs(week_mean) + 1
    rolling_24h = pd.DataFrame(prices.T).rolling(24, center=True, min_periods=1).mean().to_numpy().T

    hourly = {
        "price": prices,
        "price_z": (prices - week_mean) / week_std,
        "price_rank": prices.argsort(axis=1).argsort(axis=1) / (HOURS - 1),
        "price_rel": prices / level,
        "price_wkmean": np.repeat(week_mean, HOURS, axis=1),
        "price_d24": rolling_24h / level,
    }
    return {name: values[:, None, :] for name, values in hourly.items()}
