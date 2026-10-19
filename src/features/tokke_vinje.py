"""Tokke–Vinje coupling (BRIEFING §2.2, EDA §4.4).

Vinje releases into Vinjevatn (9.4 Mm³, tiny), which Tokke empties in about 20 h, so the
two plants run together (label correlation 0.6–0.8). Each Tokke unit gets Vinje's DP
schedule and margin, and each Vinje unit gets Tokke's. Other units get NaN (LightGBM
treats it as missing).

Reads the relaxed_dp cache, so relaxed_dp must be buildable (it doesn't have to be in the
same config).
"""

import numpy as np

from src.data import plant_of, upstream_wv, week_prices
from src.features import load_group

PARTNER = {"Tokke": "Vinje", "Vinje": "Tokke"}


def build(data):
    dp, dp_names = load_group("relaxed_dp", data)
    prices = week_prices(data)
    scale = np.abs(prices).mean(axis=1, keepdims=True) + 1
    margin = (prices[:, None, :] - upstream_wv(data, 7)[:, :, None]) / scale[:, None, :]

    first_unit = {}
    for u, unit in enumerate(data.units):
        first_unit.setdefault(plant_of(unit), u)

    shape = (data.n_cases, len(data.units), dp.shape[2])
    partner_units = np.full(shape, np.nan)
    partner_gap = np.full(shape, np.nan)
    partner_margin = np.full(shape, np.nan)
    for u, unit in enumerate(data.units):
        partner = PARTNER.get(plant_of(unit))
        if partner is None:
            continue
        p = first_unit[partner]  # plant-level values are the same for all its units
        partner_units[:, u] = dp[:, p, :, dp_names.index("dp_units_x1")]
        partner_gap[:, u] = dp[:, p, :, dp_names.index("dp_gap_x1")]
        partner_margin[:, u] = margin[:, p]

    return {
        "partner_dp_units": partner_units,
        "partner_dp_gap": partner_gap,
        "partner_margin": partner_margin,
    }
