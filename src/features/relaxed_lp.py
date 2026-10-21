"""Relaxed LP of the whole watercourse (BRIEFING §4.5, Level 2: the main track).

One linear programme per case, built from the topology YAML: all plants, reservoirs,
tunnels and rivers, hourly over the 168 hours. On/off status is relaxed: a plant may run
at any level between 0 and full capacity. The LP maximises

    income from selling power + value of the water left in the reservoirs after 168 h

subject to the water balance of every reservoir in every hour. Its solution (how much
each plant produces, how full each reservoir gets) and its duals (the effective hourly
value of water) become features; LightGBM learns the corrections.

Default model (each refinement below is an option, so earlier feature groups stay
reproducible):
- Production = k × discharge, k fixed per case from the start-day head
  (k = 9.81e-3 × head × best efficiency, MW per m³/s). No start costs, no p_min.
- Songavatn and Bitdalsvatn are merged (same levels and water value).
- Tunnels move water freely in both directions (they join lakes at the same level).
- River releases (spill, bypass) cost a tiny penalty; every reservoir also has an
  expensive emergency overflow, so no LP is infeasible.
- Min-volume constraints are lower bounds on the volume, capped at the start volume.
- End value: water left after 168 h is valued leg by leg down the cascade, each plant's
  production at its own upstream water value (d+7); see _end_values.

Options (solve_all / solve_case keyword arguments):
- min_flow: the six min-flow constraints, soft (shortfall costs MIN_FLOW_PENALTY).
- commitment: relaxed unit commitment (units on, p_min, start/stop costs).
- production_curves: production as a concave piecewise-linear function of discharge,
  from the turbine efficiency curves, generator efficiency and head losses in the YAML
  (instead of one constant k at the best efficiency).
- flow_costs="yaml": the rivers' flow_cost_const from the YAML (spill 1000, bypass 10 per
  m³/s·h) instead of the tiny penalty, so spilling is a last resort as in SHOP.
- head_iterations: re-solve with the head at each reservoir's mean level in the previous
  solution (SHOP's successive linearisation, in its simplest form).
- integer_units (needs commitment): after the LP, solve the same problem as a MILP with
  whole numbers of units on (BRIEFING §4.5 Level 3), with a time limit per case.
  production / units_on / fill come from the MILP; margin and water_value stay the LP's
  duals (a MILP has none).
- main_river (solve_all only; it changes the network): a reservoir's end value follows
  the river marked `main_river` in the YAML instead of the first river found. Matters
  for Hyljelihyl only: r_Hyljelihyl → Venemo → Totak → Vinje …, not the bypass
  b_Hyljelihyl → Vinjevatn that skips Vinje.

Uses only the §2.0 case inputs: prices and inflows over the 168 h, start-day volumes,
the +7 d water values and the constraint tables.
"""

import ast
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.sparse as sp
import yaml
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from src.data import EXTENDED_DIR, HOURS, hourly_window, plant_of, week_prices

MERGED = {"Bitdalsvatn": "Songavatn"}  # reservoir -> the reservoir it's merged into
MM3_PER_M3S_HOUR = 0.0036  # 1 m³/s for one hour = 0.0036 Mm³
RELEASE_PENALTY = 0.01  # EUR per m³/s·h for spill and bypass (default flow costs)
OVERFLOW_PENALTY = 1000.0  # EUR per m³/s·h for the emergency overflow
MIN_FLOW_PENALTY = 1000.0  # EUR per m³/s·h of shortfall below a min-flow constraint
TUNNEL_CAPACITY = 1000.0  # m³/s, effectively unlimited
GENERATOR_EFFICIENCY = 0.98  # default model only; production_curves uses the YAML curve
CURVE_SEGMENTS = 4  # linear pieces per plant with production_curves
MIN_FLOW_RIVERS = ["b_Kjelavatn", "b_Hyljelihyl", "b_Vinjevatn", "b_Byrtevatn", "b_Bandak", "r_Vest_Vassdraget"]
END_VALUE_BASIS = "path"  # how leftover water is valued; see _end_values
PRIORITY = {"Tokke_G1": 1, "Tokke_G3": 2, "Tokke_G4": 3, "Tokke_G2": 4,
            "Vinje_G1": 1, "Vinje_G2": 2, "Vinje_G3": 3}  # as in relaxed_dp


@dataclass
class Edge:
    """One water flow (m³/s), from a reservoir to a reservoir; None = outside the system."""
    name: str
    kind: str  # "plant", "tunnel", "river" or "overflow"
    source: int | None
    target: int | None
    upper: float  # m³/s; plant segments get their width per case
    lower: float = 0.0
    via: tuple[str, ...] = ()  # rivers the water passes on its way to the target
    flow_cost: float = 0.0  # the YAML's flow_cost_const (rivers)
    plant: int | None = None  # for plant segments: index into Network.plants


@dataclass
class Plant:
    """What the production function needs, from the YAML."""
    units: int
    pmin: float  # per unit, MW
    pmax: float
    start_cost: float  # EUR per start of one unit
    stop_cost: float
    outlet: float  # outlet level, m
    best_efficiency: float  # turbine × generator at the best point (default model)
    turbine_curves: list[dict]  # [{ref: head, x: unit flow, y: efficiency %}, ...]
    generator_curve: dict  # {x: MW, y: efficiency %}
    main_loss: float  # head loss = main_loss × Q² (plant flow) + penstock_loss × q² (penstock flow)
    penstock_loss: float
    penstocks: int


@dataclass
class Network:
    reservoirs: list[str]
    max_vol: np.ndarray  # (reservoir,)
    level_curves: list[tuple[list, list]]  # volume (Mm³) -> level (m), per reservoir
    plants: list[str]
    plant_data: list[Plant]
    edges: list[Edge]  # plant segments first (plant-major), then tunnels, rivers, overflows
    segments: int  # linear pieces per plant
    main_outlet: list[int | None]  # per reservoir: the edge its water mainly leaves by
    A_eq: sp.csr_matrix  # water balance, rows (hour, reservoir)
    plant_edges: np.ndarray = field(init=False)  # (plant, segment) -> edge index

    def __post_init__(self):
        self.plant_edges = np.arange(len(self.plants) * self.segments).reshape(len(self.plants), self.segments)

    def plant_source(self, p: int) -> int:
        return self.edges[self.plant_edges[p, 0]].source

    def plant_target(self, p: int) -> int | None:
        return self.edges[self.plant_edges[p, 0]].target


def _merged(name: str) -> str:
    return MERGED.get(name, name)


@lru_cache(maxsize=4)
def build_network(segments: int = 1, main_river: bool = False) -> Network:
    topology = yaml.safe_load((EXTENDED_DIR / "Tokke_Vinje_topology.yaml").read_text())
    model, connections = topology["model"], topology["connections"]
    targets = {}
    for c in connections:
        targets.setdefault((c["from_type"], c["from"]), []).append((c["to_type"], c["to"]))

    def route(kind: str, name: str, via: tuple = ()) -> tuple[str | None, tuple]:
        """Follow rivers downstream until a reservoir (None = leaves the system); also
        returns the rivers passed on the way."""
        for to_type, to in targets.get((kind, name), []):
            if to_type == "reservoir":
                return _merged(to), via
            if to_type == "river":
                return route("river", to, via + (to,))
        return None, via

    reservoirs = [r for r in model["reservoir"] if r not in MERGED]
    index = {r: i for i, r in enumerate(reservoirs)}
    max_vol = np.zeros(len(reservoirs))
    for r, spec in model["reservoir"].items():
        max_vol[index[_merged(r)]] += spec["max_vol"]
    level_curves = [(model["reservoir"][r]["vol_head"]["x"], model["reservoir"][r]["vol_head"]["y"])
                    for r in reservoirs]
    for r, into in MERGED.items():  # merged lake: scale the main lake's curve to the joint volume
        x, y = level_curves[index[into]]
        scale = max_vol[index[into]] / model["reservoir"][into]["max_vol"]
        level_curves[index[into]] = ([v * scale for v in x], y)

    sources_of = {}  # plant or tunnel -> reservoir that feeds it
    for c in connections:
        if c["from_type"] == "reservoir" and c["to_type"] in ("plant", "tunnel"):
            sources_of[c["to"]] = _merged(c["from"])

    edges, plants, plant_data = [], [], []
    for p, (plant, spec) in enumerate(model["plant"].items()):
        source = sources_of.get(plant)
        if source is None:  # fed through tunnels (Songa)
            source = next(sources_of[t] for t in model["tunnel"] if ("plant", plant) in targets.get(("tunnel", t), []))
        units = [g for g in model["generator"] if plant_of(g) == plant]
        first = model["generator"][units[0]]
        curves = first["turb_eff_curves"]
        curves = ast.literal_eval(curves) if isinstance(curves, str) else curves
        target, via = route("plant", plant)
        for s in range(segments):
            edges.append(Edge(f"{plant}#{s}", "plant", index[source], index[target] if target else None,
                              upper=np.inf, via=via, plant=p))
        plants.append(plant)
        plant_data.append(Plant(
            units=len(units), pmin=first["p_min"], pmax=first["p_max"],
            start_cost=first["startcost_const"], stop_cost=first.get("stopcost_const", 0.0),
            outlet=spec["outlet_line"],
            best_efficiency=max(max(c["y"]) for c in curves) / 100 * GENERATOR_EFFICIENCY,
            turbine_curves=curves, generator_curve=first["gen_eff_curve"],
            main_loss=float(spec["main_loss"][0]), penstock_loss=float(spec["penstock_loss"][0]),
            penstocks=len({model["generator"][g]["penstock"] for g in units}),
        ))

    for tunnel in model["tunnel"]:
        target = route("tunnel", tunnel)[0]
        if target is not None:  # tunnels into plants are part of the plant edge
            edges.append(Edge(tunnel, "tunnel", index[sources_of[tunnel]], index[target],
                              upper=TUNNEL_CAPACITY, lower=-TUNNEL_CAPACITY))

    for c in connections:
        if c["from_type"] == "reservoir" and c["to_type"] == "river":
            source, (target, via) = index[_merged(c["from"])], route("river", c["to"])
            target = index[target] if target else None
            if source != target:
                river = model["river"].get(c["to"], {})
                edges.append(Edge(c["to"], "river", source, target, upper=river.get("max_flow_const", np.inf),
                                  via=(c["to"],) + via, flow_cost=river.get("flow_cost_const", 0.0)))
    for r in reservoirs:
        edges.append(Edge(f"overflow_{r}", "overflow", index[r], None, upper=np.inf))

    def main_outlet(r: int) -> int | None:
        """Plant if the reservoir feeds one, else a tunnel, else the first river onwards.
        With main_river, a river marked `main_river` in the YAML wins over the others
        (Hyljelihyl: r_Hyljelihyl → Venemo, not the bypass b_Hyljelihyl → Vinjevatn)."""
        for kind in ("plant", "tunnel", "river"):
            candidates = [e for e, edge in enumerate(edges)
                          if edge.kind == kind and edge.source == r and (kind == "plant" or edge.target is not None)]
            if main_river and kind == "river":
                candidates.sort(key=lambda e: not model["river"].get(edges[e].name, {}).get("main_river"))
            if candidates:
                return candidates[0]
        return None

    incidence = np.zeros((len(reservoirs), len(edges)))
    for e, edge in enumerate(edges):
        if edge.source is not None:
            incidence[edge.source, e] -= 1
        if edge.target is not None:
            incidence[edge.target, e] += 1
    # V[t] - V[t-1] - 0.0036 × (inflows - outflows)[t] = 0.0036 × natural inflow[t]
    step = sp.eye(HOURS) - sp.eye(HOURS, k=-1)
    A_eq = sp.hstack([-MM3_PER_M3S_HOUR * sp.kron(sp.eye(HOURS), sp.csr_matrix(incidence)),
                      sp.kron(step, sp.eye(len(reservoirs)))]).tocsr()

    return Network(reservoirs, max_vol, level_curves, plants, plant_data, edges, segments,
                   [main_outlet(r) for r in range(len(reservoirs))], A_eq)


def _production_function(plant: Plant, head: float, curves: bool, segments: int) -> tuple[np.ndarray, np.ndarray]:
    """Slopes k (MW per m³/s) and widths (m³/s) of the plant's linear pieces, best first.

    Default: one piece, k at the best efficiency, up to full capacity. With curves: the
    upper concave envelope of the plant's production over discharge, for 1 … n units
    sharing the flow equally, from the turbine curve closest to this head, the generator
    curve and the head losses; split into `segments` pieces of equal width.
    """
    if not curves:
        k = 9.81e-3 * head * plant.best_efficiency
        return np.array([k]), np.array([plant.units * plant.pmax / k])

    curve = min(plant.turbine_curves, key=lambda c: abs(c["ref"] - head))
    unit_flow, efficiency = np.asarray(curve["x"], float), np.asarray(curve["y"], float) / 100
    points = [(0.0, 0.0)]
    for m in range(1, plant.units + 1):
        flow = m * unit_flow
        penstock_flow = flow / min(m, plant.penstocks)
        net_head = np.maximum(head - plant.main_loss * flow ** 2 - plant.penstock_loss * penstock_flow ** 2, 0)
        mw = 9.81e-3 * net_head * flow * efficiency
        mw *= np.interp(mw / m, plant.generator_curve["x"], plant.generator_curve["y"]) / 100
        points += list(zip(flow, np.minimum(mw, m * plant.pmax)))
    points = sorted(points)

    hull = [points[0]]  # upper concave envelope (monotone chain)
    for q, mw in points[1:]:
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (q - hull[-2][0]) <= (mw - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append((q, mw))
    hull_q, hull_mw = np.array(hull).T
    top = int(np.argmax(hull_mw))  # beyond the maximum, more flow doesn't add power
    hull_q, hull_mw = hull_q[:top + 1], hull_mw[:top + 1]

    breaks = np.linspace(0, hull_q[-1], segments + 1)
    mw_at = np.interp(breaks, hull_q, hull_mw)
    widths = np.diff(breaks)
    return np.maximum(np.diff(mw_at) / widths, 0), widths


def _end_values(net: Network, k_avg: np.ndarray, water_value: np.ndarray, basis: str,
                wv_from: dict[str, str] | None = None) -> np.ndarray:
    """Value (EUR) of 1 Mm³ left in each reservoir after 168 h, (reservoir,).

    Water left in a reservoir would later run through the plants on its way down
    (its main outlet, then the next reservoir's, ...). Each plant on the way turns
    1 Mm³ into k / 0.0036 MWh (k_avg: the plant's average MW per m³/s at full load).
    The bases differ in which water value prices that energy:
    - "path": each plant's MWh at the water value of the reservoir feeding that plant.
      Then every plant's trade-off is price vs its own upstream water value, which is
      what the labels follow (BRIEFING §3); the default.
    - "sea": all MWh down to the sea at the reservoir's own water value (fails the
      sanity check: the synthetic water values of neighbouring reservoirs aren't
      consistent with each other).
    - "local": only the first plant's MWh, at the reservoir's own water value.
    wv_from overrides, for "path", which reservoir's water value prices a plant's leg,
    e.g. {"Vinje": "Vinjevatn"}.
    """
    values = {}

    def walk(r: int, seen: frozenset) -> float:
        if r in values:
            return values[r]
        e = net.main_outlet[r]
        if e is None or r in seen:
            return 0.0
        edge = net.edges[e]
        mwh = k_avg[edge.plant] / MM3_PER_M3S_HOUR if edge.kind == "plant" else 0.0
        below = walk(edge.target, seen | {r}) if edge.target is not None else 0.0
        if basis == "path":
            plant_name = net.plants[edge.plant] if edge.kind == "plant" else None
            priced_at = net.reservoirs.index(wv_from[plant_name]) if plant_name in (wv_from or {}) else r
            values[r] = mwh * water_value[priced_at] + below
        elif basis == "local":
            values[r] = mwh * water_value[r] if edge.kind == "plant" else below * water_value[r] / max(
                water_value[edge.target], 1e-9) if edge.target is not None else 0.0
        else:  # sea: everything below priced at this reservoir's own water value
            values[r] = (mwh + (below / max(water_value[edge.target], 1e-9) if edge.target is not None else 0.0)) * water_value[r]
        return values[r]

    return np.array([walk(r, frozenset()) for r in range(len(net.reservoirs))])


def solve_case(net: Network, prices: np.ndarray, inflow: np.ndarray, start_volume: np.ndarray,
               end_water_value: np.ndarray, min_volume: np.ndarray,
               required_flow: np.ndarray | None = None, basis: str = END_VALUE_BASIS,
               wv_from: dict[str, str] | None = None, commitment: bool = False,
               production_curves: bool = False, flow_costs: str = "penalty",
               head_iterations: int = 0, integer_units: bool = False,
               milp_time_limit: float = 10.0) -> dict[str, np.ndarray] | None:
    """Solve one case. prices (hour,), inflow and min_volume (hour, reservoir) in m³/s and
    Mm³, start_volume and end_water_value (reservoir,), required_flow (hour, MIN_FLOW
    column) in m³/s or None for no min-flow constraints. Options: see the module
    docstring. Returns per-plant arrays of shape (hour, plant)."""
    levels = np.array([np.interp(v, *curve) for v, curve in zip(start_volume, net.level_curves)])
    result = None
    for _ in range(head_iterations + 1):
        result = _solve_once(net, levels, prices, inflow, start_volume, end_water_value, min_volume,
                             required_flow, basis, wv_from, commitment, production_curves, flow_costs,
                             integer_units, milp_time_limit)
        if result is None:
            return None
        mean_volume = result.pop("_volumes").mean(axis=0)
        levels = np.array([np.interp(v, *curve) for v, curve in zip(mean_volume, net.level_curves)])
    return result


def _solve_once(net, levels, prices, inflow, start_volume, end_water_value, min_volume,
                required_flow, basis, wv_from, commitment, production_curves, flow_costs,
                integer_units=False, milp_time_limit=10.0):
    n_res, n_edges, n_plants = len(net.reservoirs), len(net.edges), len(net.plants)
    heads = np.array([max(levels[net.plant_source(p)] - net.plant_data[p].outlet, 1.0) for p in range(n_plants)])
    pieces = [_production_function(net.plant_data[p], heads[p], production_curves, net.segments) for p in range(n_plants)]
    k = np.zeros(n_edges)  # MW per m³/s for plant segments
    lower = np.array([e.lower for e in net.edges], dtype=float)
    upper = np.array([e.upper for e in net.edges], dtype=float)
    for p, (slopes, widths) in enumerate(pieces):
        k[net.plant_edges[p]] = slopes
        upper[net.plant_edges[p]] = widths
    capacity_flow = np.array([w.sum() for _, w in pieces])
    capacity_mw = np.array([(s * w).sum() for s, w in pieces])
    k_avg = capacity_mw / capacity_flow  # MW per m³/s at full load

    if flow_costs == "yaml":
        cost = np.array([e.flow_cost if e.kind == "river" else OVERFLOW_PENALTY if e.kind == "overflow" else 0.0
                         for e in net.edges])
    else:
        cost = np.array([{"river": RELEASE_PENALTY, "overflow": OVERFLOW_PENALTY}.get(e.kind, 0.0) for e in net.edges])
    flow_cost = np.tile(cost, (HOURS, 1)) - prices[:, None] * k[None, :]  # income from production
    volume_cost = np.zeros((HOURS, n_res))
    volume_cost[-1] = -_end_values(net, k_avg, end_water_value, basis, wv_from)  # EUR per Mm³ left at the end
    vol_upper = np.maximum(net.max_vol, start_volume)
    vol_lower = np.minimum(min_volume, start_volume)  # (hour, reservoir); feasible by construction

    b_eq = MM3_PER_M3S_HOUR * inflow
    b_eq[0] += start_volume
    bounds = np.vstack([np.column_stack([np.tile(lower, HOURS), np.tile(upper, HOURS)]),
                        np.column_stack([vol_lower.ravel(), np.tile(vol_upper, HOURS)])])

    # Variable blocks after the flows and volumes are added below; each block has a size,
    # a cost, bounds, and its columns in the inequality rows (A_ub x ≤ b_ub).
    sizes = {"flow": HOURS * n_edges, "volume": HOURS * n_res}
    costs, bound_parts = [flow_cost.ravel(), volume_cost.ravel()], [bounds]
    ub_blocks, ub_rhs = [], []

    def add_variables(name, size, cost, low, high):
        sizes[name] = size
        costs.append(np.broadcast_to(cost, size).astype(float))
        bound_parts.append(np.column_stack([np.broadcast_to(low, size), np.broadcast_to(high, size)]))

    if required_flow is not None:
        # Shortfall z ≥ 0 per constraint and hour: −(flows through the river) − z ≤ −required
        n_c = required_flow.shape[1]
        through = np.zeros((n_c, n_edges))
        for c, river in enumerate(MIN_FLOW_RIVERS):
            through[c, [e for e, edge in enumerate(net.edges) if river in edge.via]] = -1
        add_variables("shortfall", HOURS * n_c, MIN_FLOW_PENALTY, 0, np.inf)
        ub_blocks.append({"flow": sp.kron(sp.eye(HOURS), sp.csr_matrix(through)), "shortfall": -sp.eye(HOURS * n_c)})
        ub_rhs.append(-required_flow.ravel())

    if commitment:
        # Relaxed unit commitment per plant and hour: "on" units in [0, n_units] (fractional),
        # production between on × p_min and on × p_max, start/stop costs on changes in "on".
        # The state before the case is unknown, so hour 0 starts free.
        plant_units = np.array([d.units for d in net.plant_data], float)
        add_variables("on", HOURS * n_plants, 0, 0, np.tile(plant_units, HOURS))
        add_variables("start", HOURS * n_plants, np.tile([d.start_cost for d in net.plant_data], HOURS), 0, np.inf)
        add_variables("stop", HOURS * n_plants, np.tile([d.stop_cost for d in net.plant_data], HOURS), 0, np.inf)
        hours, plants = sp.eye(HOURS), sp.eye(n_plants)
        mw_of_plant = np.zeros((n_plants, n_edges))
        for p in range(n_plants):
            mw_of_plant[p, net.plant_edges[p]] = k[net.plant_edges[p]]
        production = sp.kron(hours, sp.csr_matrix(mw_of_plant))  # MW per plant and hour
        change = sp.kron((sp.eye(HOURS) - sp.eye(HOURS, k=-1)).tocsr()[1:], plants)  # on[t] − on[t−1]
        later = sp.kron(sp.eye(HOURS).tocsr()[1:], plants)
        ub_blocks += [
            {"flow": production, "on": -sp.kron(hours, sp.diags([d.pmax for d in net.plant_data]))},  # MW ≤ on × p_max
            {"flow": -production, "on": sp.kron(hours, sp.diags([d.pmin for d in net.plant_data]))},  # MW ≥ on × p_min
            {"on": change, "start": -later},                                                          # start ≥ increase
            {"on": -change, "stop": -later},                                                          # stop ≥ decrease
        ]
        ub_rhs += [np.zeros(HOURS * n_plants)] * 2 + [np.zeros((HOURS - 1) * n_plants)] * 2

    def row_block(columns: dict) -> sp.csr_matrix:
        n_rows = next(iter(columns.values())).shape[0]
        return sp.hstack([columns.get(name, sp.csr_matrix((n_rows, size))) for name, size in sizes.items()])

    n_extra = sum(sizes.values()) - HOURS * (n_edges + n_res)
    A_eq = sp.hstack([net.A_eq, sp.csr_matrix((net.A_eq.shape[0], n_extra))]).tocsr() if n_extra else net.A_eq
    A_ub = sp.vstack([row_block(b) for b in ub_blocks]).tocsr() if ub_blocks else None
    b_ub = np.concatenate(ub_rhs) if ub_rhs else None
    res = linprog(np.concatenate(costs), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq.ravel(),
                  bounds=np.vstack(bound_parts), method="highs")
    if res.status != 0:
        return None

    flows = res.x[:HOURS * n_edges].reshape(HOURS, n_edges)
    volumes = res.x[HOURS * n_edges:HOURS * (n_edges + n_res)].reshape(HOURS, n_res)
    plant_mw = np.stack([(flows[:, net.plant_edges[p]] * k[net.plant_edges[p]]).sum(axis=1) for p in range(n_plants)], axis=1)
    # Duals: change in the (minimised) objective per extra Mm³ in a reservoir at hour t,
    # so the value of water there is −dual. Convert to EUR per MWh of this plant.
    water_value = -res.eqlin.marginals.reshape(HOURS, n_res)
    to_mwh = MM3_PER_M3S_HOUR / k_avg
    sources = np.array([net.plant_source(p) for p in range(n_plants)])
    targets = [net.plant_target(p) for p in range(n_plants)]
    wv_source = water_value[:, sources] * to_mwh
    wv_target = np.column_stack([water_value[:, t] if t is not None else np.zeros(HOURS) for t in targets]) * to_mwh
    result = {
        "production": plant_mw / capacity_mw,  # share of plant capacity
        "margin": prices[:, None] - wv_source + wv_target,  # EUR/MWh: profit of one more MWh
        "water_value": wv_source,
        "fill": volumes[:, sources] / net.max_vol[sources],
        "_volumes": volumes,
    }
    on_offset = sum(size for name, size in sizes.items() if name in ("flow", "volume", "shortfall"))
    if commitment:
        result["units_on"] = res.x[on_offset:on_offset + HOURS * n_plants].reshape(HOURS, n_plants)
    if integer_units and commitment:
        integrality = np.zeros(len(res.x))
        integrality[on_offset:on_offset + HOURS * n_plants] = 1
        all_bounds = np.vstack(bound_parts)
        constraints = [LinearConstraint(A_eq, b_eq.ravel(), b_eq.ravel())]
        if A_ub is not None:
            constraints.append(LinearConstraint(A_ub, -np.inf, b_ub))
        mip = milp(np.concatenate(costs), integrality=integrality, bounds=Bounds(all_bounds[:, 0], all_bounds[:, 1]),
                   constraints=constraints, options={"time_limit": milp_time_limit, "mip_rel_gap": 1e-3})
        if mip.x is not None:  # best solution found within the time limit
            flows = mip.x[:HOURS * n_edges].reshape(HOURS, n_edges)
            volumes = mip.x[HOURS * n_edges:HOURS * (n_edges + n_res)].reshape(HOURS, n_res)
            plant_mw = np.stack([(flows[:, net.plant_edges[p]] * k[net.plant_edges[p]]).sum(axis=1)
                                 for p in range(n_plants)], axis=1)
            result.update(production=plant_mw / capacity_mw, fill=volumes[:, sources] / net.max_vol[sources],
                          units_on=mip.x[on_offset:on_offset + HOURS * n_plants].reshape(HOURS, n_plants),
                          _volumes=volumes)
    return result


def _solve_chunk(args):
    options, inputs = args  # inputs: case_inputs' arrays, sliced to a chunk
    options = dict(options)
    net = build_network(CURVE_SEGMENTS if options.get("production_curves") else 1, options.pop("main_river", False))
    return [solve_case(net, *case, **options) for case in zip(*inputs)]


def case_inputs(data, net: Network, case_idx: np.ndarray):
    """The §2.0 inputs per case: prices, inflows per (merged) reservoir, start volumes,
    +7 d water values, min-volume constraints and the flow required by the min-flow
    constraints."""
    starts = data.starts.iloc[case_idx].reset_index(drop=True)
    prices = week_prices(data)[case_idx]
    index = {r: i for i, r in enumerate(net.reservoirs)}

    topology = yaml.safe_load((EXTENDED_DIR / "Tokke_Vinje_topology.yaml").read_text())
    river_target = {c["from"]: c["to"] for c in topology["connections"] if c["from_type"] == "river"}
    inflow = np.zeros((len(case_idx), HOURS, len(net.reservoirs)))
    for column in data.inflow.columns:
        target = column if column in data.max_vol else river_target[column]  # e.g. r_Vest_Vassdraget → Bandak
        inflow[:, :, index[_merged(target)]] += hourly_window(data.inflow[column], starts)

    start_volume = np.zeros((len(case_idx), len(net.reservoirs)))
    for r in data.max_vol:
        start_volume[:, index[_merged(r)]] += data.volume.loc[starts, r].to_numpy()
    end_days = starts + pd.Timedelta(days=7)
    end_wv = np.column_stack([data.water_value.loc[end_days, r].to_numpy() for r in net.reservoirs])
    constraint = pd.read_csv(EXTENDED_DIR / "Constraint_min_volume.csv", parse_dates=["date"]).set_index("date")
    min_volume = np.zeros((len(case_idx), HOURS, len(net.reservoirs)))
    for r in constraint.columns:
        min_volume[:, :, index[_merged(r)]] = hourly_window(constraint[r], starts)
    flows = pd.read_csv(EXTENDED_DIR / "Constraint_min_flow.csv", parse_dates=["date"]).set_index("date")
    required = np.stack([hourly_window(flows[river], starts) for river in MIN_FLOW_RIVERS], axis=2)
    # r_Vest_Vassdraget also carries its own natural inflow, which counts towards its minimum
    natural = np.stack([hourly_window(data.inflow[river], starts) if river in data.inflow else np.zeros((len(starts), HOURS))
                        for river in MIN_FLOW_RIVERS], axis=2)
    required_flow = np.maximum(required - natural, 0)
    return prices, inflow, start_volume, end_wv, min_volume, required_flow


def solve_all(data, case_idx: np.ndarray, workers: int | None = None, min_flow: bool = False,
              **options) -> dict[str, np.ndarray]:
    """Solve the given cases in parallel. Returns arrays (case, hour, plant); NaN if an LP
    failed. `options` are solve_case's keyword arguments (basis, commitment, ...) and
    main_river."""
    net = build_network(CURVE_SEGMENTS if options.get("production_curves") else 1, options.get("main_river", False))
    inputs = case_inputs(data, net, case_idx)
    if not min_flow:
        inputs = inputs[:-1]  # drop required_flow
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = np.array_split(np.arange(len(case_idx)), workers * 4)
    with ProcessPoolExecutor(workers) as pool:
        results = [r for chunk in pool.map(_solve_chunk, [(options, [a[c] for a in inputs]) for c in chunks]) for r in chunk]

    failed = sum(r is None for r in results)
    if failed:
        print(f"    {failed} of {len(results)} LPs failed; their features are NaN")
    keys = ["production", "margin", "water_value", "fill"] + (["units_on"] if options.get("commitment") else [])
    empty = np.full((HOURS, len(net.plants)), np.nan)
    return {key: np.stack([r[key] if r is not None else empty for r in results]) for key in keys} | {
        "plants": net.plants}


def build(data):
    return lp_features(data, solve_all(data, np.arange(data.n_cases)), prefix="lp")


def lp_features(data, lp: dict, prefix: str) -> dict[str, np.ndarray]:
    """Per-unit features from solve_all's per-plant results."""
    plant_index = {p: i for i, p in enumerate(lp["plants"])}
    scale = np.abs(week_prices(data)).mean(axis=1, keepdims=True) + 1  # (case, 1)
    n_units = {p: sum(plant_of(u) == p for u in data.units) for p in plant_index}

    features = {name: np.zeros((data.n_cases, len(data.units), HOURS))
                for name in ["production", "unit_on", "margin", "water_value", "fill"]}
    for u, unit in enumerate(data.units):
        p = plant_index[plant_of(unit)]
        production = lp["production"][:, :, p]
        features["production"][:, u] = production
        # Units are committed in priority order: the r-th unit runs once r-1 are fully on.
        # With commitment the LP has its own units-on; otherwise derive it from production.
        units_on = lp["units_on"][:, :, p] if "units_on" in lp else production * n_units[plant_of(unit)]
        rank = PRIORITY.get(unit, 1)
        features["unit_on"][:, u] = np.clip(units_on - (rank - 1), 0, 1)
        features["margin"][:, u] = lp["margin"][:, :, p] / scale
        features["water_value"][:, u] = lp["water_value"][:, :, p] / scale
        features["fill"][:, u] = lp["fill"][:, :, p]
    return {f"{prefix}_{name}": values for name, values in features.items()}
