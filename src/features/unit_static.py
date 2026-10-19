"""Unit statics: unit ID (used as a categorical feature), p_max and start cost."""

import numpy as np


def build(data):
    generators = [data.generators[u] for u in data.units]
    return {
        "unit": np.arange(len(data.units))[None, :, None],
        "p_max": np.array([g["p_max"] for g in generators])[None, :, None],
        "startcost": np.array([g["startcost_const"] for g in generators])[None, :, None],
    }
