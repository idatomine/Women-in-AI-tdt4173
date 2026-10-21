"""Part 2: is the end valuation consistent with what the LP can do inside the horizon?

Per reservoir, on ~50 train cases with the relaxed_lp_shop2 options:
- last-hour dual (value of 1 Mm³ at hour 167) vs _end_values (EUR per Mm³), and whether
  the end volume sits at a bound;
- mean dual over the horizon vs the end value;
- end-of-horizon arbitrage along each non-plant edge: ev(target) - cost - ev(source) > 0
  means it pays to move water across that edge at the end, whatever the price;
- total flow per river/tunnel edge (Mm³ per case).
Flags the Hyljelihyl bug with '{"main_river": false}' (BRIEFING §9). From the repo root:
    PYTHONPATH=$PWD .venv/bin/python analysis/end_value_check.py ['{"flow_costs": "yaml_curves"}'] [out.csv]
"""
import json
import sys

import numpy as np
import pandas as pd

import src.features.relaxed_lp as lp
from src.data import HOURS, load_data

extra = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
main_river = extra.pop("main_river", True)
options = dict(commitment=True, production_curves=True, flow_costs="yaml") | extra

captured = {}
_linprog, _end_values = lp.linprog, lp._end_values
lp.linprog = lambda *a, **k: captured.__setitem__("res", _linprog(*a, **k)) or captured["res"]
lp._end_values = lambda *a, **k: captured.__setitem__("ev", _end_values(*a, **k)) or captured["ev"]

data = load_data()
net = lp.build_network(lp.CURVE_SEGMENTS, main_river)
idx = np.arange(0, data.n_train, data.n_train // 50)[:50]
inputs = lp.case_inputs(data, net, idx)
n_res, n_edges = len(net.reservoirs), len(net.edges)
with_curves = options["flow_costs"] == "yaml_curves"
cost = np.array([e.flow_cost + with_curves * e.curve_cost if e.kind == "river" else 0.0 for e in net.edges])  # per m³/s·h
cost_mm3 = cost / lp.MM3_PER_M3S_HOUR  # EUR per Mm³ through the edge

rows, arb, flows_used = [], [], []
for c, case in enumerate(zip(*inputs)):
    prices, inflow, start_volume, end_wv, min_volume, required_flow = case
    out = lp.solve_case(net, prices, inflow, start_volume, end_wv, min_volume, required_flow, **options)
    res, ev = captured["res"], captured["ev"]
    dual = -res.eqlin.marginals.reshape(HOURS, n_res)
    vols = res.x[HOURS * n_edges:HOURS * (n_edges + n_res)].reshape(HOURS, n_res)
    flows = res.x[:HOURS * n_edges].reshape(HOURS, n_edges)
    upper = np.maximum(net.max_vol, start_volume)
    lower = np.minimum(min_volume[-1], start_volume)
    for r, name in enumerate(net.reservoirs):
        rows.append(dict(case=c, reservoir=name, ev=ev[r], dual_end=dual[-1, r], dual_mean=dual.mean(axis=0)[r],
                         dual_start=dual[0, r],
                         at_upper=vols[-1, r] >= upper[r] - 1e-6, at_lower=vols[-1, r] <= lower[r] + 1e-6,
                         fill_end=vols[-1, r] / net.max_vol[r]))
    for e, edge in enumerate(net.edges):
        if edge.kind in ("river", "tunnel") and edge.source is not None:
            tgt = ev[edge.target] if edge.target is not None else 0.0
            arb.append(dict(case=c, edge=edge.name, kind=edge.kind, gain=tgt - cost_mm3[e] - ev[edge.source],
                            reverse_gain=(ev[edge.source] - ev[edge.target]) if edge.kind == "tunnel" else np.nan))
            flows_used.append(dict(case=c, edge=edge.name, Mm3=flows[:, e].sum() * lp.MM3_PER_M3S_HOUR,
                                   last24=flows[-24:, e].sum() * lp.MM3_PER_M3S_HOUR, hours_used=(np.abs(flows[:, e]) > 1e-3).sum()))

df = pd.DataFrame(rows)
df["ratio_end"] = df.dual_end / df.ev.replace(0, np.nan)
df["ratio_mean"] = df.dual_mean / df.ev.replace(0, np.nan)
summary = df.groupby("reservoir", sort=False).agg(
    ev_med=("ev", "median"), dual_end_med=("dual_end", "median"), ratio_end_med=("ratio_end", "median"),
    ratio_end_p10=("ratio_end", lambda s: s.quantile(.1)), ratio_end_p90=("ratio_end", lambda s: s.quantile(.9)),
    ratio_mean_med=("ratio_mean", "median"), at_upper=("at_upper", "mean"), at_lower=("at_lower", "mean"),
    fill_end=("fill_end", "median"))
pd.set_option("display.width", 220)
print(f"options {options}, main_river {main_river}, {len(idx)} cases")
print("\nPer reservoir (EUR per Mm³; ratio = dual / end value):")
print(summary.round(3).to_string())
a = pd.DataFrame(arb).groupby("edge", sort=False).agg(kind=("kind", "first"), gain_med=("gain", "median"),
                                                       gain_pos=("gain", lambda s: (s > 0).mean()),
                                                       reverse_med=("reverse_gain", "median"))
f = pd.DataFrame(flows_used).groupby("edge", sort=False).agg(Mm3_med=("Mm3", "median"), Mm3_mean=("Mm3", "mean"),
                                                             last24_mean=("last24", "mean"), hours_used=("hours_used", "mean"))
print("\nNon-plant edges: end-of-horizon gain of moving 1 Mm³ across (EUR), share of cases > 0, and use:")
print(a.join(f).round(3).to_string())
df.to_csv(sys.argv[2] if len(sys.argv) > 2 else "/dev/null", index=False)
