"""Variant of relaxed_lp with the six min-flow constraints (Constraint_min_flow.csv).

The EDA and BRIEFING §8 suspect Lio is driven by the min flow on r_Vest_Vassdraget
(5–14 m³/s, which Lio's releases feed). Lio is the LP's weakest unit (AUC 0.85 vs 0.90
for the simple rule). Same LP otherwise; features are prefixed `lpm_`.
"""

import numpy as np

from src.features.relaxed_lp import lp_features, solve_all


def build(data):
    return lp_features(data, solve_all(data, np.arange(data.n_cases), min_flow=True), prefix="lpm")
