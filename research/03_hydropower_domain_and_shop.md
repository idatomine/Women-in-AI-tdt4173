# 03 – Hydropower domain, SHOP economics and the Tokke–Vinje topology

Purpose: domain background for feature engineering on the SINTEF/TDT4173 unit-commitment task
(14 units, 168-hour cases, labels from SHOP v17.17).

Sources: the local files `data/extended/Tokke_Vinje_topology.yaml` and `Tokke_topology.pdf`, the
`data/kernel/*.csv` and `data/extended/Constraint_*.csv` files (all numbers below that come from the data
were computed by me with plain python3), SHOP documentation (docs.shop.sintef.energy), and a few web sources
listed at the end. Wherever I infer something rather than read it from a source, I say so.

---

## 0. Key empirical finding (read this first)

I ran a simple rule-based check on the 2015–2022 training labels:
**"unit is ON in hour t if price(t) > water value of the reservoir directly upstream of its plant"**.

| Unit | WV at case start date | **WV at start + 7 days (end of horizon)** | WV at +6 d | WV at +8 d | majority-class baseline |
|---|---|---|---|---|---|
| Hogga_G1 | 0.645 | **0.839** | 0.802 | 0.794 | 0.595 |
| Tokke_G1 | 0.574 | **0.788** | 0.700 | 0.699 | 0.671 |
| Tokke_G2 | 0.566 | **0.831** | 0.736 | 0.738 | 0.522 |
| Tokke_G3 | 0.570 | **0.829** | 0.730 | 0.732 | 0.564 |
| Tokke_G4 | 0.568 | **0.825** | 0.730 | 0.734 | 0.502 |
| Lio_G1 | 0.738 | **0.786** | 0.781 | 0.782 | 0.747 |
| Byrte_G1 | 0.816 | **0.957** | 0.940 | 0.941 | 0.542 |
| Vinje_G1 | 0.678 | **0.720** | 0.719 | 0.720 | 0.515 |
| Vinje_G2 | 0.667 | **0.707** | 0.705 | 0.707 | 0.580 |
| Vinje_G3 | 0.630 | **0.651** | 0.650 | 0.650 | 0.741 |
| Haukeli_G1 | 0.755 | **0.781** | 0.780 | 0.781 | 0.547 |
| Songa_G1 | 0.859 | **0.947** | 0.937 | 0.939 | 0.515 |
| Kjela_G1 | 0.807 | **0.916** | 0.904 | 0.910 | 0.529 |
| Vesle Kjela_G1 | 0.828 | **0.950** | n/a* | n/a* | 0.533 |

(*the +6/+8 day run's column parsing broke on the space in "Vesle Kjela"; the start/+7 runs are correct.)

Conclusions:
1. **The water value to use for a case starting on day D is the row dated D+7 days** in
   `Synthetic_water_value_2015_2024.csv`, i.e. the end-of-horizon value. D+7 beats both D+6 and D+8, which
   strongly suggests the file is indexed by the date the value applies to (the horizon end). Using the
   start-date row loses 5–25 accuracy points.
2. **Water values are "local" energy values in EUR/MWh, so you compare them to the price directly.**
   Comparing `price` with the upstream reservoir's WV works much better than a "global" conversion
   `(WV_up·G_up − WV_down·G_down)/e_plant` (column 3 in the first run: about 0.57–0.75). This matches SHOP's
   `energy_value_input` semantics (see §1.2).
3. A single-threshold rule already gets about 0.95 on Byrte, Songa and Vesle Kjela (single-unit plants with
   large upstream reservoirs). The hard cases are the **multi-unit plants (Vinje, Tokke G1)**, **Lio**,
   **Haukeli** and **Hogga**. For those, how many units run depends on the size of the margin, the water
   balance of small buffer reservoirs, and the constraints.
4. **Test caveat:** the WV file ends at 2025-01-01. Test cases starting after 2024-12-25 have no D+7 row,
   so you need a fallback (the latest available value, or a forward-fill).

Also found: the 17 reservoirs' synthetic water values fall into **9 groups whose series are exactly
identical**, and each group matches a set of reservoirs with the same "energy equivalent to the sea" (§2.4):

| Identical-WV group | mean WV 2015–2024 (EUR/MWh) |
|---|---|
| Kjelavatn = Staavatn | 67.0 |
| Langesae = Foersvatn = Bordalsvatn | 64.2 |
| Venemo = Vaamarvatn = Totak = Hyljelihyl | 67.1 |
| Botnedalsvatn | 65.4 |
| Bitdalsvatn = Songavatn | 74.9 |
| Langeidvatn = Vatjern | 75.6 |
| Vinjevatn | 61.5 |
| Bandak | 63.3 |
| Byrtevatn | 70.5 |

So the water-value data carries only 9 independent series, not 17. Deduplicate them before feature creation.

---

## 1. How SHOP decides unit commitment

### 1.1 The optimisation problem
SHOP (SINTEF) is a short-term hydro scheduling tool for cascaded systems. It maximises
*revenue from selling energy at the given price, plus the value of water left in the reservoirs at the end,
minus start/stop costs and penalties*, subject to water balance, plant/unit limits and constraints.
SINTEF describes the method as successive linear programming. Per the 2018 workshop slides, it alternates two
modes: a MILP "unit commitment" mode (binary on/off per unit and time step; "full" iterations) and an LP
"load dispatch" mode ("incremental" iterations), updating head/reservoir trajectories until convergence.
The plant docs confirm `mip_flag`/`mip_length` (which time steps get binary start-up/min-production
variables) and the full vs incremental PQ-curve segmentation (`n_mip_seg_*` for full mode, `n_seg_*` for
incremental mode around the unit's best-efficiency point, BEP).

For a single hour and a single unit, the economics reduce to (my simplification):

> run the unit if `price_t × P(Q)` > `water value of upstream reservoir (EUR/m³) × Q` − `value of the same
> water arriving in the downstream reservoir`, and the margin over the horizon covers start + stop cost.

With a price-taker (no bids, a given price series) and a 168-hour horizon, the solution is close to
"bang-bang": run at or near BEP or p_max in hours where price is above the water value, and shut down when it
is below. Exceptions come from constraints, start costs and small reservoirs.

### 1.2 Water value: EUR/MWh → EUR/m³
- SHOP reservoir attribute `energy_value_input` (doc unit NOK/MWh, the currency is configurable) is
  *"the input local energy value of the reservoir which will be converted to a global water value
  internally"*. `energy_conversion_factor` (MWh/Mm³) is *"calculated ... based on the first plant below the
  reservoir and initial start head"*, and `calc_global_water_value` (NOK/Mm³) is the result.
- The docs do not spell out the arithmetic. My reading of the definitions: global WV(r) ≈ local EV(r) ×
  conv_factor(first plant below r) + global WV(downstream reservoir). Under that reading the
  **break-even price for the first plant below r is simply local EV(r)**. Run the plant when price > EV(r),
  adjusted by efficiency at the operating point. The empirical test in §0 supports this.
- Energy equivalent of a plant: e ≈ ρ·g·H_net·η / 3.6e9 MWh per m³, i.e. **e [MWh/Mm³] ≈ 9.81 · H_net[m] ·
  η_turb · η_gen / 3.6**. The values for this system are in §2.4.
- `water_value_input` (an alternative to energy_value_input) gives water values "referred to the sea" in
  NOK/Mm³ as a constant, a table or cuts. The task statement says per-reservoir EUR/MWh values are used, which
  matches energy_value_input.
- **End-of-horizon valuation**: the water value only prices the *terminal* storage. Inside the week, SHOP
  moves water freely to the highest-price hours. If the reservoir is large compared to weekly discharge
  (Songavatn 640 Mm³, Bandak 174, Totak 258), the decision is close to an hourly comparison of price with the
  end WV. If the reservoir is small (Vinjevatn 9.4 Mm³, Vaamarvatn 26 Mm³, Hyljelihyl 0.26, Vatjern 0.3), the
  weekly water balance binds and the *ranking* of hours within the week matters more than the absolute level.

### 1.3 YAML attributes that matter for on/off decisions

| Attribute (object) | Meaning (SHOP docs) | Relevance to on/off |
|---|---|---|
| `p_min`, `p_max` (generator) | static min / max production, MW | **High.** A unit cannot run below p_min, so small-margin hours can still be "off" (the binary choice is between 0 and ≥ p_min). p_min ≈ 34–36 % of p_max for all units here. |
| `p_nom` | rated capacity, only used for droop | none |
| `startcost_const`, `stopcost_const` | cost of turning on / off in one time step | **High** for short price spikes. Values 92–222 (applied symmetrically, so a full on/off cycle costs 2×). Small compared to hourly revenue (e.g. Tokke at about 80 MW × 50 EUR = 4000 EUR/h), so they mostly suppress 1–2 hour blips. |
| `gen_eff_curve` | generator efficiency vs MW (97 % at p_min → 99 % at p_max for all units) | low |
| `turb_eff_curves` | turbine efficiency vs flow, one curve per net head (`ref`) | **Medium.** Defines the BEP (at about 75 % of max flow for all units except Hogga at 85 %). Efficiency drops 5–10 points at minimum flow, so p_min operation is "expensive", which helps explain all-or-nothing behaviour. |
| `penstock` (gen), `penstock_loss` (plant) | shared penstock and quadratic loss | low/medium. Tokke G1–G2 share penstock 1 and G3–G4 penstock 2. Vinje has one per unit. |
| `outlet_line`, `main_loss` (plant) | tailwater level; quadratic tunnel loss | define head; Haukeli main_loss 2.93 and Byrte 0.23 are large. |
| `max_vol`, `lrl`, `hrl`, `vol_head` (reservoir) | volume limits and volume→level curve | **High**: fill level → head (efficiency) and spill risk (forced running when close to max_vol). |
| `s_*` rivers with `flow_cost_const: 1000` | spillways, heavily penalised | spill avoidance: when the reservoir is near full and inflow is high, units run even at low price. |
| `b_*` rivers, `flow_cost_const: 10` | bypass gates (cheap) | can carry min-flow requirements instead of running units |
| `r_*` rivers with `flow_cost_curve` | natural river with flow cost | Staavatn / Langesae / Langeidvatn / Hyljelihyl outflows are unregulated "main rivers" |
| `min_uptime`/`min_downtime` | minutes | **Not set in this YAML**, so no explicit min up/down-time. Persistence comes only from start/stop costs. |
| `maintenance_flag`, `committed_in` | forced status | not present in the YAML |

---

## 2. Tokke–Vinje topology parsed from the YAML

### 2.1 Plants and generators

| Plant | Unit(s) | p_min / p_max (MW) | start = stop cost | Max flow per unit (m³/s) | Approx. net head (m)† | BEP (approx MW) | Takes water from | Discharges to |
|---|---|---|---|---|---|---|---|---|
| **Vesle Kjela** | G1 | 2.9 / 8.5 | 92 | 20 | 27–70 (≈61) | ≈8 | **Kjelavatn** | Foersvatn |
| **Kjela** | G1 | 20.6 / 60 | 171 | 40 | 122–185 (≈174) | ≈45 | **Foersvatn** | Hyljelihyl |
| **Songa** | G1 | 41.2 / 120 | 222 | 52 | 252–287 (≈280) | ≈95 | **Songavatn + Bitdalsvatn** (two tunnels) | Totak |
| **Haukeli** | G1 | 0.3 / 4.9 | 143 | 2.1 | 265–268 | ≈3.6 | **Vatjern** | Vinjevatn |
| **Vinje** | G1, G2, G3 | 34.4 / 100 each | 198 | 55 each | 212–225 (≈222) | ≈79 each | **Vaamarvatn** (fed by tunnel from Totak) | Vinjevatn |
| **Byrte** | G1 | 6.9 / 20 | 135 | 8 | 259–295 (≈287) | ≈15 | **Botnedalsvatn** | Byrtevatn |
| **Lio** | G1 | 14.8 / 43 | 163 | 14 | 334–352 (≈346) | ≈32 | **Byrtevatn** | r_Vest_Vassdraget → Bandak |
| **Tokke** | G1, G2, G3, G4 | 36.9 / 107.5 each | 214 | 32 each | 389–394 (≈392) | ≈81 each | **Vinjevatn** | Bandak |
| **Hogga** | G1 | 3 / 17 | 94 | 168.6 | 9.5–13 (≈12) | ≈15 | **Bandak** | out of system (sea side) |

† Range of the `ref` heads of the turbine-efficiency curves. Bracket = the curve with highest peak
efficiency. BEP MW = 9.81·H·Q_BEP·η/1000 (my calculation).

Installed: 4×107.5 + 3×100 + 120 + 60 + 43 + 20 + 17 + 8.5 + 4.9 ≈ 1003 MW (Statkraft quotes about
1017 MW and 13 generator sets for the regulation area).

### 2.2 Reservoirs (max_vol, Mm³) and outflows

| Reservoir | max_vol | lrl–hrl (m a.s.l.) | Outflow(s) | Feeds plant |
|---|---|---|---|---|
| Songavatn | 640.0 | 939–974 | tunnel → Songa; bypass/spill → Totak | Songa |
| Totak | 257.8 | 680–687.3 | tunnel → Vaamarvatn; bypass/spill → r_Vest_Vassdraget | (Vinje via Vaamarvatn) |
| Bordalsvatn | 184.2 | 852–891 | tunnel → Foersvatn; bypass/spill → Hyljelihyl and → Venemo | (Kjela via Foersvatn) |
| Bandak | 174.2 | 69.46–73 | Hogga; bypass/spill → out | Hogga |
| Foersvatn | 122.3 | 828.5–891 | Kjela; bypass/spill → Hyljelihyl | Kjela |
| Bitdalsvatn | 110.0 | 939–974 | tunnel → Songa; bypass/spill → Totak | Songa |
| Kjelavatn | 84.5 | 918–944 | Vesle Kjela; bypass/spill → Hyljelihyl | Vesle Kjela |
| Langesae | 78.0 | 1067–1090 | river/spill → Foersvatn (unregulated) | – |
| Byrtevatn | 76.71 | 427.3–445.6 | Lio; bypass/spill → r_Vest_Vassdraget | Lio |
| Botnedalsvatn | 54.17 | 705–740 | Byrte; bypass/spill → r_Vest_Vassdraget | Byrte |
| Staavatn | 48.0 | 966–978.5 | river/spill → Kjelavatn (unregulated) | – |
| Langeidvatn | 32.1 | 878.5–885.5 | river/spill → Vatjern | – |
| Vaamarvatn | 26.2 | 677–687.3 | Vinje; bypass/spill → Vinjevatn | Vinje |
| Venemo | 23.0 | 666–703 | tunnel → Totak; bypass/spill → Vinjevatn | – |
| Vinjevatn | 9.36 | 462–465.5 | Tokke; bypass/spill → r_Vest_Vassdraget | Tokke |
| Vatjern | 0.30 | 835–838 | Haukeli; bypass/spill → Vinjevatn | Haukeli |
| Hyljelihyl | 0.26 | 702.4–706 | river → Venemo (main); bypass → Vinjevatn | – |

Sum of max_vol = 1921 Mm³ (Statkraft states about 1909 Mm³ for the system).
`r_Vest_Vassdraget` is a river node (no storage) that collects Lio plus bypass/spill from Totak, Vinjevatn,
Botnedalsvatn and Byrtevatn, and flows into Bandak. Inflow series exist for all 17 reservoirs plus
r_Vest_Vassdraget.

### 2.3 Cascade diagram (upstream → downstream; `==>` plant, `-->` tunnel, `~~>` river/bypass)

```
 Staavatn ~~> Kjelavatn ==[Vesle Kjela]==> Foersvatn <~~ Langesae
                                             ^   ^
                       Bordalsvatn ---------+   |   (tunnel)
                           |  ~~> Hyljelihyl, ~~> Venemo
                                             |
                                    Foersvatn ==[Kjela]==> Hyljelihyl ~~> Venemo --> Totak
                                                                                      ^
 Songavatn --+                                                                        |
             +--> ==[Songa]==> Totak <------------------------------------------------+
 Bitdalsvatn-+                    |
                                  +--> Vaamarvatn ==[Vinje x3]==> Vinjevatn
 Langeidvatn ~~> Vatjern ==[Haukeli]==========================> Vinjevatn
                                                                  |
                                                     Vinjevatn ==[Tokke x4]==> Bandak ==[Hogga]==> (out)
                                                                                 ^
 Botnedalsvatn ==[Byrte]==> Byrtevatn ==[Lio]==> r_Vest_Vassdraget -------------+
   (Totak, Vinjevatn, Botnedalsvatn, Byrtevatn bypass/spill also go to r_Vest_Vassdraget)
```

Main stem (by energy): **Songa/Bitdal → Songa → Totak → Vaamarvatn → Vinje → Vinjevatn → Tokke → Bandak →
Hogga**. Western branch: **Staavatn → Kjelavatn → Vesle Kjela → Foersvatn → Kjela → Hyljelihyl → Venemo →
Totak**. Eastern side branch: **Botnedalsvatn → Byrte → Byrtevatn → Lio → Bandak**. Small branch:
**Langeidvatn → Vatjern → Haukeli → Vinjevatn**.

### 2.4 Energy equivalents (my estimates, MWh/Mm³ = kWh per 1000 m³)

| Plant | e_plant | Reservoir | Global equivalent to sea G (sum of plants downstream) |
|---|---|---|---|
| Hogga | ≈29 | Bandak | 29 |
| Tokke | ≈943 | Vinjevatn | ≈972 |
| Lio | ≈831 | Byrtevatn | ≈860 |
| Byrte | ≈690 | Botnedalsvatn | ≈1550 |
| Vinje | ≈533 | Vaamarvatn, Totak, Venemo, Hyljelihyl | ≈1505 |
| Haukeli | ≈644 | Vatjern, Langeidvatn | ≈1616 |
| Kjela | ≈418 | Foersvatn, Langesae, Bordalsvatn | ≈1923 |
| Songa | ≈673 | Songavatn, Bitdalsvatn | ≈2178 |
| Vesle Kjela | ≈149 | Kjelavatn, Staavatn | ≈2072 |

The reservoir groups with equal G match **exactly** the identical-WV groups in §0. This is how the synthetic
water values were evidently built: one series per hydraulic "energy level".

### 2.5 Coupled decisions
- **Tokke G1–G4** share Vinjevatn (9.36 Mm³). At full load Tokke draws 128 m³/s ≈ 0.46 Mm³/h, which
  empties Vinjevatn in about 20 h without inflow. Tokke is therefore tied to **Vinje + Haukeli
  discharge + Vinjevatn inflow**. The number of Tokke units online tracks the inflow rate into Vinjevatn.
- **Vinje G1–G3** share Vaamarvatn (26 Mm³), refilled from Totak through a tunnel. Vinje 3×55 = 165 m³/s.
- Identical units are interchangeable, so SHOP's choice between them is arbitrary/tie-broken. Training
  on-rates show a consistent ordering (Tokke G1 > G3 > G4 ≈ G2; Vinje G1 > G2 > G3). Model **"number of
  units on per plant"** and then assign by that priority order.
- **Songa** draws from two reservoirs (Songavatn + Bitdalsvatn) with identical WV.
- **Kjela/Vesle Kjela** form a chain. Vesle Kjela fills Foersvatn for Kjela.
- **Hyljelihyl (0.26 Mm³) and Vatjern (0.3 Mm³)** are tiny. Haukeli is essentially run-of-river on the
  Langeidvatn/Vatjern inflow.
- **Hogga** (low head, 168.6 m³/s) passes the outflow of Tokke + Lio + r_Vest_Vassdraget out of Bandak. Its
  commitment correlates with upstream activity and Bandak level as well as price.

### 2.6 Constraint files (seasonal; same every year)
- Min flow (m³/s): b_Bandak 4 (Oct–Apr) / 5 (May–Sep). r_Vest_Vassdraget 5 / 8 (16 Jun–14 Sep) / 14
  (15 Sep–15 Nov). b_Vinjevatn 1 / 2 (16 Jun–15 Nov). b_Byrtevatn 0 / 0.2 (15 May–15 Sep). b_Kjelavatn and
  b_Hyljelihyl vary smoothly (summer peaks of 0.75 and 2.0).
- Min volume (Mm³): **Totak ≥ 211.89 (82 % of max) 1 Jul–15 Aug, ≥ 194.23 until 1 Nov**; **Byrtevatn ≥
  67.68 (88 %) 1 Jul–15 Aug**; **Staavatn ≥ 46.66 (97 %) 1 Jul–1 Oct**; otherwise 0. These summer
  (recreational/landscape) constraints block drawdown. When the volume is near the floor, the plants that
  drain those reservoirs (Vinje via Totak→Vaamarvatn, Lio) are forced to hold back.
- Lio feeds r_Vest_Vassdraget, which has a min flow of up to 14 m³/s. Running Lio (14 m³/s max) is one way
  to meet it (hypothesis, not verified). This may explain Lio's high on-rate (98 % in 2017, 97 % in 2019).

---

## 3. Market context

- **Price area.** Tokke–Vinje is in Vest-Telemark. Southern Norway is split between NO1 and NO2, and Telemark
  is normally in **NO2 (south-west)**. I could not find a primary source in this session that confirms the
  zone. However, the price file's profile (2020 mean 9.3 EUR/MWh, 2022 mean 211 EUR/MWh) fits the NO2
  pattern; news reports gave NO2 at about 193 EUR/MWh year-to-date in August 2022. Treat this as "very likely
  NO2".
- **Price statistics from `Historical_day_ahead_price_2015_2025.csv` (EUR/MWh)**

| Year | mean | median | p5 | p95 | max | sd | # negative hours |
|---|---|---|---|---|---|---|---|
| 2015 | 19.8 | 22.0 | 7.8 | 29.0 | 69.7 | 7.6 | 0 |
| 2016 | 25.2 | 23.6 | 17.1 | 37.0 | 95.1 | 6.6 | 0 |
| 2017 | 28.8 | 28.9 | 23.1 | 33.8 | 114.7 | 4.6 | 0 |
| 2018 | 43.3 | 43.6 | 28.5 | 55.7 | 105.0 | 9.4 | 0 |
| 2019 | 39.3 | 38.8 | 28.4 | 53.2 | 109.5 | 8.2 | 0 |
| 2020 | 9.3 | 7.0 | 1.2 | 24.3 | 99.9 | 8.3 | 5 |
| 2021 | 75.1 | 59.3 | 33.6 | 150.8 | 600.2 | 47.2 | 5 |
| 2022 | 211.3 | 181.1 | 50.1 | 474.9 | 844.0 | 125.8 | 0 |
| **2023 (test)** | 79.4 | 80.5 | 5.6 | 135.3 | 261.9 | 36.3 | **174** |
| **2024 (test)** | 50.1 | 47.2 | 4.1 | 97.1 | 898.2 | 34.2 | **206** |
| 2025 | 67.2 | 67.6 | 16.0 | 114.9 | 493.3 | 31.2 | 59 |

- **Regime shift.** 2015–2019 had low, stable prices (sd 5–9). 2020 had extremely low prices (wet year,
  COVID). From H2 2021 through 2022 came the energy crisis (gas prices, low southern reservoir levels, strong
  coupling to continental prices) with a mean of 211 and sd 126. Average water values follow:
  2015–19 ≈ 18–35, 2020 ≈ 8, 2021 ≈ 89, **2022 ≈ 297**, 2023 ≈ 97, 2024 ≈ 52 EUR/MWh.
- **Train/test shift implications.**
  - Absolute price and WV levels in test (about 50–80) sit between the calm years and the crisis. Every
    level-based feature is out-of-distribution for parts of the training set. **Use relative features**
    (price − WV, price / WV, within-case rank/z-score of price) rather than raw price.
  - **Negative prices** appear frequently only in 2023–2024 (380 hours compared with 10 in all of training).
    The rule "price < WV ⇒ off" still holds. Ratio features misbehave when the price is ≤ 0 or the WV is
    near 0, so prefer differences, or clip.
  - Intra-day volatility (solar-driven midday dips and evening peaks) is larger in 2023–24 than in
    2015–19, which means more starts/stops per week. Start-cost effects matter more.
  - 2022 is the closest analogue for high-volatility hours. Consider weighting 2021–2022, or validate on
  2022 (time-based CV).
- **15-minute resolution.** The price file switches to 15-minute steps at **2025-09-30 22:15 UTC**
  (consistent with the move of the Nordic day-ahead market to 15-minute MTU around 1 Oct 2025). This is
  **outside the 2023–2024 test range**, but resample to hourly if you ever use 2025 data. Timestamps are UTC
  (`+00:00`). Note that the case `starttime` is a date: check whether hour t0 is 00:00 UTC or local time.

---

## 4. Seasonal hydrology

Total system inflow (sum of the 18 inflow series, mean m³/s, 1958–2025):

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 27 | 20 | 24 | 73 | **305** | **293** | 147 | 102 | 113 | 130 | 84 | 42 |

System fill (sum vol / 1921 Mm³, monthly mean 2015–2024):

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.68 | 0.57 | 0.46 | **0.36** | 0.40 | 0.65 | 0.76 | 0.80 | 0.82 | **0.85** | 0.85 | 0.77 |

- **Snowmelt flood in May–June** (10–15× winter inflow), a second, smaller autumn rain peak in Sep–Oct, and
  very low inflow Dec–Mar. Reservoirs are drawn down through winter to a minimum in April/early May, then
  refill. The fill on 1 May ranged from 0.21 (2015) to 0.58 (2019). The fill on 1 Oct was 0.86–0.95 in normal
  years, but **only 0.60 (2021) and 0.68 (2022)**, the dry crisis years, which fed the high water values.
- Implications for commitment:
  - **Spring flood**: high inflow plus filling reservoirs means a spill risk in small/medium reservoirs, so
    units run more regardless of price (Tokke G1 on 83 % in May). Water values may also fall.
  - **Summer**: min-volume constraints on Totak, Byrtevatn and Staavatn plus refilling mean less production
    from Vinje (Vinje G3 on only 13 % in May and 19 % in June) and lower Lio in May–Jun (58 %).
  - **Winter**: little inflow, so production relies on storage. Decisions are almost purely price-vs-WV,
    with the morning/evening peak hours chosen.
  - Annual on-rates swing strongly with hydrology and prices (e.g. Kjela 14 % in 2021 vs 73 % in 2018;
    Songa 12 % in 2022 vs 71 % in 2020). Inflow and fill should therefore enter as case-level features.
- Inflow is given hourly but is mostly piecewise constant (daily series repeated hourly). Use daily or
  weekly sums.

---

## 5. Implications for feature engineering (prioritised)

Notation: the case starts on day D (t = 0..167). up(u) = reservoir directly feeding unit u's plant
(§2.1). WV_end(r) = water value row at **D+7** (fallback: last available).

1. **`margin_t = price_t − WV_end(up(u))`** (EUR/MWh), plus the sign indicator. On its own it reaches about
   0.79–0.96 accuracy on most units (§0). Also add `price_t − WV_start(up(u))`, and the WV change
   (end − start) as a trend signal.
2. **Within-case relative price**: rank/percentile of price_t within the 168 h, z-score, and
   price_t − case median. These capture "which hours get the water" when water is limited, and they
   transfer well across the 2015–19 / 2022 / 2023–24 regimes.
3. **Margin normalised by level**: margin / max(|WV|, 5) and margin / case price sd. These are robust to the
   absolute-level shift. Avoid raw price/WV ratios because of negative prices.
4. **Count of profitable hours in the case**: number of hours with margin > 0, and the weekly sum of positive
   margin. This links to how much water SHOP can release, and so to how many units run (Tokke/Vinje).
5. **Upstream reservoir state at D**: fill ratio vol/max_vol for up(u) and for the main storages (Totak,
   Songavatn, Bandak, Foersvatn, Kjelavatn, Byrtevatn, Botnedalsvatn); headroom (max_vol − vol);
   **distance to the min-volume constraint** (vol − min_vol_constr) for Totak, Byrtevatn, Staavatn.
6. **Inflow over the horizon** (sum over the 168 h, from the given inflow series) for up(u) **and all
   reservoirs upstream of it**. Key aggregates: Vinjevatn local inflow plus the upstream groups for Tokke,
   Vatjern + Langeidvatn for Haukeli, and Totak + Vaamarvatn + Venemo + Hyljelihyl for Vinje. Also add
   inflow / plant max discharge (a spill-pressure index) and "weeks to fill" = headroom / inflow.
7. **Unit static features** (if you train one model across units): p_min, p_max, start cost, e_plant, head,
   number of units in the plant, the unit's index within the plant (priority order), and the global energy
   equivalent G.
8. **Calendar**: hour-of-day, weekday/weekend, month or day-of-year (sin/cos), and the seasonal constraint
   flags (min-flow levels, min-vol active). Inflow/fill already carry most of the seasonal signal.
9. **Temporal context of price**: price at t−1, t+1 and rolling 3–6 h mean/max of the margin (start-cost
   persistence: SHOP avoids 1-hour on blips). Distance to the case end (end effects in the last hours).
10. **Structure-aware target handling**: for Tokke (4) and Vinje (3), predict the *number of units on*,
    then map to units using the observed priority (G1 first). Enforce consistency, e.g. G3 on ⇒ G1 on, if
    the data supports it (check the label co-occurrence).
11. **Group-level water values**: use the 9 deduplicated WV series. Also add the WV difference between up(u)
    and the downstream reservoir (water "parked" downstream has a different value). This is useful for
    Hogga/Tokke/Lio where Bandak's WV differs.
12. **Post-processing**: smooth predictions to remove isolated 1-hour flips (start/stop cost ≈ 92–222 per
    transition). On average there are only 0.2–1.9 starts per unit per week in training.

Pitfalls:
- Do not use the case-start WV as "the" water value. Use D+7.
- Watch time zones (UTC in all files).
- Validate on 2022 (or 2021–2022) as a proxy for test, given the regime shift.

---

## Sources
- SHOP docs, reservoir object: https://docs.shop.sintef.energy/objects/reservoir/reservoir.html
- SHOP docs, generator object: https://docs.shop.sintef.energy/objects/generator/generator.html
- SHOP docs, plant object: https://docs.shop.sintef.energy/objects/plant/plant.html
- SINTEF SHOP workshop slides (two-mode MILP/LP iteration, coupling to water values): https://www.sintef.no/contentassets/174bfb8434fb49eeae186d403e165678/p3_4_skjelbred.pdf
- SINTEF SHOP overview (successive LP, 7–14 day horizon): https://ewh.ieee.org/conf/powertech/2009/papers/621.pdf
- Tokke power station (430 MW, 2140 GWh, 394 m head Vinjevatn→Bandak): https://en.wikipedia.org/wiki/Tokke_Hydroelectric_Power_Station
- Statkraft Tokke–Vinje brochure (13 units, 1017 MW, ≈1909 Mm³ storage): https://www.statkraft.com/globalassets/0/.com/1-about-statkraft/where-we-operate/norway/norwaytokke-vinje_eng.pdf
- Vinje kraftverk (3 Francis units, 300 MW, Totak main reservoir): https://snl.no/Vinje_kraftverk
- NO2 price 2022 context (≈193 EUR/MWh YTD Aug 2022): https://www.euronews.com/next/2022/08/20/norway-electricity
- SSB on 2022 record prices: https://www.ssb.no/en/energi-og-industri/energi/statistikk/elektrisitetspriser/article-for-electricity-prices/record-high-electricity-price-in-2022--curbed-by-electricity-support-for-households
