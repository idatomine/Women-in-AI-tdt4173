"""Value terminal water with the efficient production segment, not full-load average."""
import numpy as np
from src.features.relaxed_lp import lp_features, solve_all

def build(data):
    lp = solve_all(data, np.arange(data.n_cases), workers=6, min_flow=True, commitment=True,
                   production_curves=True, flow_costs='yaml', main_river=True,
                   terminal_efficiency='best')
    return lp_features(data, lp, prefix='lpb')
