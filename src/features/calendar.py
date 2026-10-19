"""Calendar: hour within the case, hour of day (UTC), and the start day's position."""

import numpy as np

from src.data import HOURS


def build(data):
    hour = np.arange(HOURS)[None, None, :]
    return {
        "h": hour,
        "hod": hour % 24,
        "doy": data.starts.dt.dayofyear.to_numpy()[:, None, None],
        "dow": data.starts.dt.dayofweek.to_numpy()[:, None, None],
    }
