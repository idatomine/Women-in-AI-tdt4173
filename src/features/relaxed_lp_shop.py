"""Most SHOP-like LP so far: relaxed_lp_uc (min flow, relaxed unit commitment) plus the
YAML production curves (turbine and generator efficiency, head losses) and the YAML
river flow costs (spill 1000, bypass 10 per m³/s·h). See BRIEFING §9.

LP sanity check (every 6th train case), mean AUC over units: unit_on 0.8473 → 0.8885,
margin 0.9207 → 0.9289 compared with relaxed_lp_uc. Features are prefixed `lps_`.
"""

import numpy as np

from src.features.relaxed_lp import lp_features, solve_all


def build(data):
    lp = solve_all(data, np.arange(data.n_cases), min_flow=True, commitment=True,
                   production_curves=True, flow_costs="yaml")
    return lp_features(data, lp, prefix="lps")
