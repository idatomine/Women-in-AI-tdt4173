"""relaxed_lp_shop with main_river: water left in Hyljelihyl after 168 h (and so in
everything upstream of it: Foersvatn, Kjelavatn, ...) is valued along the YAML's main
river r_Hyljelihyl → Venemo → Totak → Vinje ..., not the bypass b_Hyljelihyl → Vinjevatn
that skips Vinje. The old end value under-credited what the LP could earn inside the
horizon via r_Hyljelihyl, so it over-credited running Kjela and distorted the duals in
the Totak/Vinje/Tokke chain. See BRIEFING §9.

LP sanity check (every 6th train case), mean AUC over units: unit_on 0.8885 → 0.9447,
margin 0.9289 → 0.9739 compared with relaxed_lp_shop (Kjela margin 0.690 → 0.976).
Features are prefixed `lps2_`.
"""

import numpy as np

from src.features.relaxed_lp import lp_features, solve_all


def build(data):
    lp = solve_all(data, np.arange(data.n_cases), min_flow=True, commitment=True,
                   production_curves=True, flow_costs="yaml", main_river=True)
    return lp_features(data, lp, prefix="lps2")
