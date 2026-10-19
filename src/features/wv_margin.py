"""Price vs water value per unit and hour: the core decision signal (BRIEFING §2.1, §3).

`margin_up` (price − upstream water value at +7 d) is the strong one. The net versions
subtract the downstream value too, which is much weaker on its own.
"""

import numpy as np

from src.data import downstream_wv, upstream_wv, week_prices


def build(data):
    prices = week_prices(data)[:, None, :]
    week_level = np.abs(prices.mean(axis=2, keepdims=True)) + 1
    scale = np.abs(prices).mean(axis=2, keepdims=True) + 1
    up_end = upstream_wv(data, 7)[:, :, None]
    net = up_end - downstream_wv(data, 7)[:, :, None]

    return {
        "margin": (prices - net) / week_level,
        "margin_up": (prices - up_end) / scale,
        "price_over_wv": prices / (np.abs(up_end) + 1),
        "margin_net": (prices - net) / scale,
    }
