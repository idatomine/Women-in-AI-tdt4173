"""LP accuracy loop (BRIEFING §9): per-unit AUC of the LP's own signals (unit_on, margin)
against the labels, on every 6th train case (487 cases, ~30 s on 6 cores).

Usage, from the repo root:
    PYTHONPATH=$PWD .venv/bin/python analysis/lp_loop.py '{"min_flow": true, "commitment": true}'
    PYTHONPATH=$PWD .venv/bin/python analysis/lp_loop.py '{"min_flow": true, "commitment": true}' relaxed_lp_uc lpu

The JSON holds solve_all's options (see src/features/relaxed_lp.py). The optional last two
arguments compare the result with a cached feature group (regression check).
"""
import json, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.data import load_data
from src.features import load_group
from src.features.relaxed_lp import lp_features, solve_all

if __name__ == "__main__":
    options = json.loads(sys.argv[1])
    d = load_data(); idx = np.arange(0, d.n_train, 6)
    t = time.time(); lp = solve_all(d, idx, workers=6, **options)
    print(f"options {options}: {len(idx)} LPs in {time.time() - t:.0f} s")
    sub = type("D", (), {})(); sub.n_cases, sub.units = len(idx), d.units
    sub.price, sub.starts = d.price, d.starts.iloc[idx].reset_index(drop=True)
    feats = lp_features(sub, lp, prefix="x")
    if len(sys.argv) > 2:  # regression check against a cached group
        cached, names = load_group(sys.argv[2], d)
        for n in ["unit_on", "margin"]:
            a = feats[f"x_{n}"]; b = np.asarray(cached[idx, :, :, names.index(f"{sys.argv[3]}_{n}")])
            print(f"  regression {n}: max |difference| vs cached {sys.argv[2]} = {np.nanmax(np.abs(a - b)):.2e}")
    rows = {}
    for u, unit in enumerate(d.units):
        y = d.Y[idx, u].ravel()
        rows[unit] = {s: roc_auc_score(y, np.nan_to_num(feats[f"x_{s}"][:, u]).ravel()) for s in ["unit_on", "margin"]}
    table = pd.DataFrame(rows).T
    print("  mean AUC  unit_on {:.4f}  margin {:.4f}".format(*table.mean()))
    print("  per unit (unit_on / margin): " + "  ".join(f"{u.replace('_G', ' G')} {r.unit_on:.3f}/{r.margin:.3f}" for u, r in table.iterrows()))
