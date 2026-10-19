"""Scarcity coefficient c in water value ≈ c × mean price (BRIEFING §2.1), per unit.

Note: `scarcity_end` and `next_week_price` use the prices of the week *after* the case,
which is outside the §2.0 case inputs. It's not label leakage (the +7 d water value
already contains that week's mean price), but keep it in mind for the leakage audit.
"""

import numpy as np

from src.data import HOURS, upstream_wv, week_prices


def build(data):
    week_mean = week_prices(data).mean(axis=1)[:, None]
    next_week_mean = week_prices(data, offset_hours=HOURS).mean(axis=1)[:, None]

    return {
        "scarcity_start": (upstream_wv(data, 0) / (np.abs(week_mean) + 1e-3))[:, :, None],
        "scarcity_end": (upstream_wv(data, 7) / (np.abs(next_week_mean) + 1e-3))[:, :, None],
        "next_week_price": (next_week_mean / (np.abs(week_mean) + 1))[:, :, None],
    }
