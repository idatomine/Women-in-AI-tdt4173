"""Water values per unit: upstream at start and +7 d, downstream at +7 d, and the net value."""

import numpy as np

from src.data import downstream_wv, upstream_wv, week_prices


def build(data):
    week_mean = week_prices(data).mean(axis=1)[:, None]
    up_end = upstream_wv(data, 7)
    up_start = upstream_wv(data, 0)
    down_end = downstream_wv(data, 7)
    net = up_end - down_end

    per_unit_week = {
        "wv_up_end": up_end,
        "wv_up_start": up_start,
        "wv_dn_end": down_end,
        "net_wv": net,
        "net_wv_rel": net / (np.abs(week_mean) + 1),
        "wv_trend": (up_end - up_start) / (np.abs(up_start) + 1),
    }
    return {name: values[:, :, None] for name, values in per_unit_week.items()}
