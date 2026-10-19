"""Per-plant dynamic programme over the case week (BRIEFING §4.5, Level 1).

For each plant and case: the most profitable path of "number of units on" over the 168
hours, where k units earn k × p_max × (price − upstream water value at +7 d) per hour and
each start or stop costs the YAML start cost. Ignores reservoir limits (Level 2, the LP,
adds those).

Units within a plant follow a fixed priority (EDA §3.1.1, BRIEFING §2.3): the unit with
rank r is on when at least r units are on.

The start costs in the YAML may be on a different scale than SHOP effectively uses, so the
DP is solved twice: with the YAML cost (×1) and with ×10, which gives more persistence.
"""

import numpy as np

from src.data import HOURS, plant_of, upstream_wv, week_prices

# Order in which a plant's units are committed (1 = first), from the label ON rates
PRIORITY = {"Tokke_G1": 1, "Tokke_G3": 2, "Tokke_G4": 3, "Tokke_G2": 4,
            "Vinje_G1": 1, "Vinje_G2": 2, "Vinje_G3": 3}
COST_SCALES = [1, 10]


def _path_values(reward: np.ndarray, switch_cost: np.ndarray) -> np.ndarray:
    """Best total profit of any path that is in state k at hour t, shape (case, hour, state).

    reward: (case, hour, state); switch_cost: (state from, state to).
    Forward pass (best way to arrive) + backward pass (best way to continue).
    """
    forward = np.empty_like(reward)
    backward = np.zeros_like(reward)
    forward[:, 0] = reward[:, 0]
    for t in range(1, HOURS):
        forward[:, t] = reward[:, t] + (forward[:, t - 1, :, None] - switch_cost).max(axis=1)
    for t in range(HOURS - 2, -1, -1):
        ahead = reward[:, t + 1] + backward[:, t + 1]
        backward[:, t] = (ahead[:, None, :] - switch_cost).max(axis=2)
    return forward + backward


def build(data):
    prices = week_prices(data)
    price_level = np.abs(prices).mean(axis=1, keepdims=True) + 1
    wv_up_end = upstream_wv(data, 7)
    n_cases, n_units = len(prices), len(data.units)

    features = {}
    for scale in COST_SCALES:
        on = np.zeros((n_cases, n_units, HOURS))
        gap = np.zeros((n_cases, n_units, HOURS))
        units_on = np.zeros((n_cases, n_units, HOURS))

        for plant in dict.fromkeys(plant_of(u) for u in data.units):
            members = [u for u, unit in enumerate(data.units) if plant_of(unit) == plant]
            first = data.generators[data.units[members[0]]]
            n_on = len(members)
            states = np.arange(n_on + 1)

            margin = prices - wv_up_end[:, members[0]][:, None]  # EUR/MWh, (case, hour)
            reward = margin[:, :, None] * states * first["p_max"]
            switch_cost = scale * first["startcost_const"] * np.abs(states[:, None] - states[None, :])
            values = _path_values(reward, switch_cost)

            best_state = values.argmax(axis=2)
            for u in members:
                rank = PRIORITY.get(data.units[u], 1)
                value_on = values[:, :, rank:].max(axis=2)
                value_off = values[:, :, :rank].max(axis=2)
                on[:, u] = best_state >= rank
                # Regret of the opposite decision, in hours of a typical margin for this unit
                gap[:, u] = (value_on - value_off) / (first["p_max"] * price_level)
                units_on[:, u] = best_state / n_on

        suffix = f"_x{scale}"
        features["dp_on" + suffix] = on
        features["dp_gap" + suffix] = np.clip(gap, -50, 50)
        features["dp_units" + suffix] = units_on
    return features
