"""Each unit's real storage lake: like `hydro`, but tiny pass-through reservoirs are skipped.

Haukeli takes its water from Vatjern (0.3 Mm³), whose volume data is a constant
placeholder (0.15 every day), so `hydro`'s fill is blind for Haukeli. Vatjern is fed by
Langeidvatn (32 Mm³), and Haukeli's behaviour follows Langeidvatn's fill: it ran 94% of
the time with a negative margin in 2015 (Langeidvatn 90% full) and often stayed off with a
positive margin in 2020–22 (53–70% full). Rule: if a plant's upstream reservoir is smaller
than 1 Mm³, use the largest reservoir that releases into it instead. For the other units
this repeats their own upstream reservoir.
"""

import numpy as np

from src.data import UPSTREAM, hourly_window, plant_of
from src.features.relaxed_lp import build_network

TINY_MM3 = 1.0


def storage_reservoirs() -> dict[str, list[str]]:
    """Plant -> the storage reservoir(s) behind it, skipping tiny pass-through lakes."""
    net = build_network()
    feeders = {}  # reservoir -> reservoirs that release into it (by river or tunnel)
    for edge in net.edges:
        if edge.kind in ("river", "tunnel") and edge.source is not None and edge.target is not None:
            feeders.setdefault(net.reservoirs[edge.target], set()).add(net.reservoirs[edge.source])
    max_vol = dict(zip(net.reservoirs, net.max_vol))

    storage = {}
    for plant, reservoirs in UPSTREAM.items():
        chosen = []
        for r in reservoirs:
            if max_vol.get(r, np.inf) < TINY_MM3 and feeders.get(r):
                r = max(feeders[r], key=lambda f: max_vol[f])
            chosen.append(r)
        storage[plant] = chosen
    return storage


def build(data):
    storage = storage_reservoirs()
    starts = data.starts
    fill, inflow, free = [], [], []
    for unit in data.units:
        reservoirs = storage[plant_of(unit)]
        capacity = sum(data.max_vol[r] for r in reservoirs)
        volume = sum(data.volume.loc[starts, r].to_numpy() for r in reservoirs)
        week_inflow = sum(hourly_window(data.inflow[r], starts).sum(axis=1) * 3600 / 1e6 for r in reservoirs)
        fill.append(volume / capacity)
        inflow.append(week_inflow / capacity)
        free.append(np.maximum(capacity - volume, 0) / capacity)
    return {
        "storage_fill": np.stack(fill, axis=1)[:, :, None],
        "storage_inflow_wk": np.stack(inflow, axis=1)[:, :, None],
        "storage_free": np.stack(free, axis=1)[:, :, None],
    }
