"""Water budget per unit: how much water the plant must and can release this week, in hours
at full power, plus hour-by-hour inflow (research/02: peers' "scheduling" features).

A scheduler with water for N full-power hours runs roughly the N most expensive hours.
- must_hours: water that would otherwise spill (start volume + week inflow − capacity).
- can_hours: all water available (start volume + week inflow).
- price_rank / must_hours, price_rank / can_hours: < 1 means "within the budget".
Uses the plant's real storage lake (skipping tiny pass-through reservoirs, as in
storage_upstream) and its natural inflow only; releases from upstream plants are left to
the coupling and LP features.
"""

import numpy as np

from src.data import HOURS, hourly_window, plant_of, week_prices
from src.features.relaxed_lp import MM3_PER_M3S_HOUR, build_network
from src.features.storage_upstream import storage_reservoirs


def build(data):
    net = build_network()
    plant_info = dict(zip(net.plants, net.plant_data))
    storage = storage_reservoirs()
    starts = data.starts
    prices = week_prices(data)
    # 0 = most expensive hour of the case week
    rank_desc = (-prices).argsort(axis=1).argsort(axis=1).astype(float)

    names = ["must_hours", "can_hours", "rank_over_must", "rank_over_can",
             "inflow_vs_capacity", "idle_fill", "inflow_remaining"]
    features = {n: np.zeros((data.n_cases, len(data.units), HOURS)) for n in names}
    for u, unit in enumerate(data.units):
        plant = plant_of(unit)
        info = plant_info[plant]
        reservoirs = storage[plant]
        capacity = sum(data.max_vol[r] for r in reservoirs)  # Mm³
        volume = sum(data.volume.loc[starts, r].to_numpy() for r in reservoirs)[:, None]
        inflow = sum(hourly_window(data.inflow[r], starts) for r in reservoirs)  # m³/s, (case, hour)
        cumulative = np.cumsum(inflow, axis=1) * MM3_PER_M3S_HOUR  # Mm³ arrived by the end of hour t
        week_inflow = cumulative[:, -1:]

        # Discharge at full power with the start-day head and best efficiency (as relaxed_lp's default)
        level = np.array([np.interp(v, *net.level_curves[net.reservoirs.index(reservoirs[0])])
                          for v in volume[:, 0] * data.max_vol[reservoirs[0]] / capacity])
        k = 9.81e-3 * np.maximum(level - info.outlet, 1.0) * info.best_efficiency  # MW per m³/s
        full_power_mm3_per_hour = (info.units * info.pmax / k)[:, None] * MM3_PER_M3S_HOUR

        must = np.maximum(volume + week_inflow - capacity, 0) / full_power_mm3_per_hour
        can = (volume + week_inflow) / full_power_mm3_per_hour
        features["must_hours"][:, u] = np.minimum(must, HOURS) / HOURS
        features["can_hours"][:, u] = np.minimum(can, HOURS) / HOURS
        features["rank_over_must"][:, u] = np.minimum(rank_desc / np.maximum(must, 1), 10)
        features["rank_over_can"][:, u] = np.minimum(rank_desc / np.maximum(can, 1), 10)
        features["inflow_vs_capacity"][:, u] = inflow * MM3_PER_M3S_HOUR / full_power_mm3_per_hour
        features["idle_fill"][:, u] = (volume + cumulative) / capacity  # fill by hour t if the plant stays off
        features["inflow_remaining"][:, u] = (week_inflow - cumulative) / capacity
    return features
