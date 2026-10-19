"""Cascade coupling: each plant sees the DP schedule of the plants that release into its
upstream reservoir (generalises tokke_vinje, exp 009).

When Vinje runs, Tokke runs 99.5% of the time: water arriving in a small reservoir has to
be passed on. The same may hold for Hogga (fed by Tokke via Bandak), Lio (Byrte via
Byrtevatn) and Kjela (Vesle Kjela via Foersvatn). Plants without a feeder get NaN.

Reads the relaxed_dp cache (see tokke_vinje.py).
"""

import numpy as np

from src.data import DOWNSTREAM, UPSTREAM, plant_of
from src.features import load_group


def build(data):
    dp, dp_names = load_group("relaxed_dp", data)
    plants = list(dict.fromkeys(plant_of(u) for u in data.units))
    members = {p: [u for u, unit in enumerate(data.units) if plant_of(unit) == p] for p in plants}
    feeders = {p: [q for q in plants if DOWNSTREAM[q] in UPSTREAM[p]] for p in plants}

    shape = (data.n_cases, len(data.units), dp.shape[2])
    release = np.full(shape, np.nan)
    gap = np.full(shape, np.nan)
    for p in plants:
        if not feeders[p]:
            continue
        own_capacity = sum(data.generators[data.units[u]]["p_max"] for u in members[p])
        # MW the feeders run in their DP schedules, relative to this plant's capacity
        feeder_mw = sum(
            np.asarray(dp[:, members[q][0], :, dp_names.index("dp_units_x1")])
            * sum(data.generators[data.units[u]]["p_max"] for u in members[q])
            for q in feeders[p]
        )
        feeder_gap = np.max([np.asarray(dp[:, members[q][0], :, dp_names.index("dp_gap_x1")])
                             for q in feeders[p]], axis=0)
        for u in members[p]:
            release[:, u] = feeder_mw / own_capacity
            gap[:, u] = feeder_gap

    return {"feeder_release": release, "feeder_dp_gap": gap}
