"""Profitable blocks per unit (BRIEFING §4.1 #4): where hour t sits in a run of hours with
price above the upstream water value (+7 d).

Long profitable blocks justify a start cost; short blips don't.
"""

import numpy as np

from src.data import HOURS, upstream_wv, week_prices


def build(data):
    profitable = week_prices(data)[:, None, :] > upstream_wv(data, 7)[:, :, None]  # (case, unit, hour)

    # Hours since the last change of state, and hours until the next one
    since = np.zeros(profitable.shape, dtype=np.int32)
    until = np.zeros(profitable.shape, dtype=np.int32)
    for t in range(1, HOURS):
        same = profitable[..., t] == profitable[..., t - 1]
        since[..., t] = np.where(same, since[..., t - 1] + 1, 0)
    for t in range(HOURS - 2, -1, -1):
        same = profitable[..., t] == profitable[..., t + 1]
        until[..., t] = np.where(same, until[..., t + 1] + 1, 0)
    block = since + until + 1  # length of the block that contains hour t

    return {
        "block_len": block / HOURS,
        "block_since": since / HOURS,
        "block_until": until / HOURS,
        "block_profitable_len": np.where(profitable, block, 0) / HOURS,
        "longest_profitable": np.where(profitable, block, 0).max(axis=2, keepdims=True) / HOURS,
    }
