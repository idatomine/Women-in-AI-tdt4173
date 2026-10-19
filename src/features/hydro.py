"""Reservoirs per unit: fill on the start day (upstream and downstream) and the week's inflow.

Only the start-day volume is used; later volumes are outcomes of the plan (leakage).
"""

import numpy as np

from src.data import DOWNSTREAM, UPSTREAM, hourly_window, plant_of


def build(data):
    starts = data.starts
    fill, fill_down, inflow = [], [], []
    for unit in data.units:
        up = UPSTREAM[plant_of(unit)]
        down = DOWNSTREAM[plant_of(unit)]
        capacity = sum(data.max_vol[r] for r in up)

        fill.append(sum(data.volume.loc[starts, r].to_numpy() for r in up) / capacity)
        fill_down.append(data.volume.loc[starts, down].to_numpy() / data.max_vol[down]
                         if down else np.zeros(data.n_cases))
        # Week inflow in Mm³ (m³/s × 3600 s / 1e6), relative to the upstream capacity
        week_inflow = sum(hourly_window(data.inflow[r], starts).sum(axis=1) * 3600 / 1e6 for r in up)
        inflow.append(week_inflow / capacity)

    return {
        "fill": np.stack(fill, axis=1)[:, :, None],
        "fill_dn": np.stack(fill_down, axis=1)[:, :, None],
        "inflow_wk": np.stack(inflow, axis=1)[:, :, None],
    }
