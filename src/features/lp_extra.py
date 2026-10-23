"""Extra features from relaxed_lp_shop2's LP (same options, solved again), for the units
that are weakest at exp 029 (Vinje G2/G3, Kjela). Prefix `lpx_`.

- `lpx_rank_gap`: the LP's units-on minus the unit's rank − 1, *unclipped*. lps2_unit_on
  clips it to [0, 1], so for Vinje G3 a plant at 1.8 units and one at 0.2 units both read
  0; the gap keeps how close the unit is to being needed. Equals units-on for one-unit
  plants.
- `lpx_plant_on`: the plant's units-on as a share of its units.
- `lpx_joint_fill`: the fill of the unit's source reservoir together with the lakes joined
  to it by tunnels (Venemo/Totak/Vaamarvatn for Vinje, Bordalsvatn/Foersvatn for Kjela).
  They share one end value, so the LP splits water between them arbitrarily and
  lps2_fill for Vinje (Vaamarvatn) is mostly 0 (BRIEFING §9); the group total is the
  honest fill. Other plants: their own source fill.
"""

import numpy as np

from src.data import plant_of
from src.features.relaxed_lp import PRIORITY, solve_all

TUNNEL_GROUPS = {"Vinje": ["Venemo", "Totak", "Vaamarvatn"], "Kjela": ["Bordalsvatn", "Foersvatn"]}


def build(data):
    lp = solve_all(data, np.arange(data.n_cases), min_flow=True, commitment=True, production_curves=True,
                   flow_costs="yaml", main_river=True, volumes=True)
    plant_index = {p: i for i, p in enumerate(lp["plants"])}
    reservoir_index = {r: i for i, r in enumerate(lp["reservoirs"])}
    n_units = {p: sum(plant_of(u) == p for u in data.units) for p in plant_index}

    features = {name: np.zeros((data.n_cases, len(data.units), lp["units_on"].shape[1]))
                for name in ["rank_gap", "plant_on", "joint_fill"]}
    for u, unit in enumerate(data.units):
        plant = plant_of(unit)
        units_on = lp["units_on"][:, :, plant_index[plant]]
        features["rank_gap"][:, u] = units_on - (PRIORITY.get(unit, 1) - 1)
        features["plant_on"][:, u] = units_on / n_units[plant]
        if plant in TUNNEL_GROUPS:
            group = [reservoir_index[r] for r in TUNNEL_GROUPS[plant]]
            features["joint_fill"][:, u] = lp["volumes"][:, :, group].sum(axis=2) / lp["max_vol"][group].sum()
        else:
            features["joint_fill"][:, u] = lp["fill"][:, :, plant_index[plant]]
    return {f"lpx_{name}": values for name, values in features.items()}
