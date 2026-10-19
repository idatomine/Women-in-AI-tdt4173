"""Spill pressure per unit (BRIEFING §4.1 #5): the week's inflow relative to the free space
left in the upstream and downstream reservoirs on the start day.

A nearly full reservoir with high inflow must release water (run) or spill it; a nearly
full downstream reservoir limits how much the unit can release.
"""

import numpy as np

from src.data import DOWNSTREAM, UPSTREAM, hourly_window, plant_of


def _pressure(data, reservoirs: list[str]) -> np.ndarray:
    """Week inflow (Mm³) ÷ free space on the start day, clipped to [0, 10]. Shape (case,)."""
    starts = data.starts
    capacity = sum(data.max_vol[r] for r in reservoirs)
    volume = sum(data.volume.loc[starts, r].to_numpy() for r in reservoirs)
    inflow = sum(hourly_window(data.inflow[r], starts).sum(axis=1) * 3600 / 1e6 for r in reservoirs)
    free = np.maximum(capacity - volume, 0.01 * capacity)  # volume can exceed max_vol (floods)
    return np.clip(inflow / free, 0, 10)


def build(data):
    up, down = [], []
    for unit in data.units:
        plant = plant_of(unit)
        up.append(_pressure(data, UPSTREAM[plant]))
        has_down = DOWNSTREAM[plant] and DOWNSTREAM[plant] in data.inflow.columns
        down.append(_pressure(data, [DOWNSTREAM[plant]]) if has_down else np.zeros(data.n_cases))
    return {
        "spill_pressure_up": np.stack(up, axis=1)[:, :, None],
        "spill_pressure_dn": np.stack(down, axis=1)[:, :, None],
    }
