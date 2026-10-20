"""Forward-looking margin per unit: what the rest of the case week still offers from hour t
(research/02: peers' "remaining horizon" features). Uses the upstream margin
price − water value (+7 d), the core signal (BRIEFING §3).

Unlike margin_context (exp 005: the margin at fixed offsets around t) and profitable_blocks
(exp 006: the run containing t), these summarise the whole remaining horizon, which is what
decides whether to use water now or save it for later in the week.
"""

import numpy as np

from src.data import HOURS, upstream_wv, week_prices


def _suffix(values: np.ndarray, reduce) -> np.ndarray:
    """reduce(values[..., t:]) for every t, along the last axis."""
    return np.stack([reduce(values[..., t:], axis=-1) for t in range(HOURS)], axis=-1)


def build(data):
    prices = week_prices(data)
    scale = (np.abs(prices).mean(axis=1, keepdims=True) + 1)[:, None, :]
    margin = (prices[:, None, :] - upstream_wv(data, 7)[:, :, None]) / scale  # (case, unit, hour)
    profitable = (margin > 0).astype(float)

    remaining_share = _suffix(profitable, np.mean)
    past_share = np.concatenate([np.zeros_like(profitable[..., :1]),
                                 np.cumsum(profitable, axis=-1)[..., :-1] / np.arange(1, HOURS)], axis=-1)
    # Rank of this hour's price among the remaining hours (1 = the best still to come)
    rank_remaining = _suffix(prices[:, None, :], lambda v, axis: (v[..., :1] >= v).mean(axis=axis))
    best_next_24 = np.stack([margin[..., t:t + 24].max(axis=-1) for t in range(HOURS)], axis=-1)

    # Hours until the next profitable hour (0 if this hour is profitable, 168 if none is left)
    until = np.full(margin.shape, float(HOURS))
    for t in range(HOURS - 1, -1, -1):
        later = until[..., t + 1] + 1 if t + 1 < HOURS else np.full(margin.shape[:-1], float(HOURS))
        until[..., t] = np.where(profitable[..., t] > 0, 0, np.minimum(later, HOURS))

    return {
        "fwd_profitable_share": remaining_share,
        "past_profitable_share": past_share,
        "fwd_price_rank": np.broadcast_to(rank_remaining, margin.shape),
        "fwd_best_margin_24h": best_next_24,
        "fwd_mean_margin": _suffix(margin, np.mean),
        "hours_to_profitable": until / HOURS,
    }
