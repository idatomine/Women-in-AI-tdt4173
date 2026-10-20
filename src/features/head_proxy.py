"""Head per unit at the start of the case (research/02: peers' "head proxies").

The height drop sets how much power each m³ gives (and SHOP's efficiency curves depend on
it). Features: start-day head relative to the turbines' mid reference head, and the
upstream reservoir's level within its regulated range (lrl … hrl).
"""

import numpy as np
import yaml

from src.data import EXTENDED_DIR, UPSTREAM, plant_of
from src.features.relaxed_lp import build_network


def build(data):
    net = build_network()
    plant_info = dict(zip(net.plants, net.plant_data))
    reservoirs = yaml.safe_load((EXTENDED_DIR / "Tokke_Vinje_topology.yaml").read_text())["model"]["reservoir"]
    head, level_share = [], []
    for unit in data.units:
        plant = plant_of(unit)
        info = plant_info[plant]
        r = UPSTREAM[plant][0]
        curve = net.level_curves[net.reservoirs.index(r)] if r in net.reservoirs else (
            reservoirs[r]["vol_head"]["x"], reservoirs[r]["vol_head"]["y"])
        level = np.interp(data.volume.loc[data.starts, r].to_numpy(), *curve)
        reference = np.median([c["ref"] for c in info.turbine_curves])
        head.append((level - info.outlet) / reference)
        lrl, hrl = reservoirs[r]["lrl"], reservoirs[r]["hrl"]
        level_share.append((level - lrl) / max(hrl - lrl, 1e-6))
    return {
        "head_rel": np.stack(head, axis=1)[:, :, None],
        "level_in_range": np.stack(level_share, axis=1)[:, :, None],
    }
