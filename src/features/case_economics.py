"""Week-level economics per unit: how much of the week is worth running."""

import numpy as np

from src.data import HOURS, upstream_wv, week_prices


def build(data):
    prices = week_prices(data)[:, None, :]
    scale = np.abs(prices).mean(axis=2, keepdims=True) + 1
    margin = prices - upstream_wv(data, 7)[:, :, None]

    return {
        "profitable_share": (margin > 0).mean(axis=2, keepdims=True),
        "positive_margin": np.clip(margin, 0, None).sum(axis=2, keepdims=True) / scale / HOURS,
    }
