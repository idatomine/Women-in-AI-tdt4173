"""Variant of relaxed_lp with min-flow constraints and a relaxed unit commitment: units
on per plant and hour (fractional), production between on × p_min and on × p_max, and the
YAML start/stop costs on changes in units on (BRIEFING §4.0 build step 4).

Adds the persistence from start costs that only relaxed_dp provided, combined with the
water balance. Sanity check (every 6th train case): LP unit-on AUC 0.842 → 0.847 mean over
units (Vinje G1 0.834 → 0.862, Haukeli 0.852 → 0.886). Features are prefixed `lpu_`.
"""

import numpy as np

from src.features.relaxed_lp import lp_features, solve_all


def build(data):
    lp = solve_all(data, np.arange(data.n_cases), min_flow=True, commitment=True)
    return lp_features(data, lp, prefix="lpu")
