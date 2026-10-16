# Briefing: SINTEF unit-commitment prediction (TDT4173)

Handoff document for whoever (human or Claude session) trains and tunes the model. It
summarises what is **verified in the data** (✅), what comes from **research** (📚, see
`research/0*.md` for sources), and what is still a **hypothesis** (❓).

Status 2026-10-15: Part 1 (understanding + direction) done. No final pipeline exists yet;
`analysis/` holds throwaway EDA/baseline scripts only.

---

## 1. Problem in one paragraph

SINTEF's SHOP optimiser schedules the Tokke–Vinje hydropower system (Norway, 17 reservoirs,
9 plants, 14 generating units). For each **case** (a 168-hour week starting at 00:00 UTC on
a given day, one case per day) we predict P(unit ON) for each of 14 units × 168 hours =
**2352 outputs per case**. Train: 2922 cases (2015-01-01 → 2022-12-31). Test: 725 cases
(2023-01-01 → 2024-12-25). Metric: **micro-averaged ROC-AUC** over all 2352 columns
(`kaggle_metric.py`). Only ranking matters, but **the ranking is global across units and
hours**: a 0.7 for Haukeli must mean the same as a 0.7 for Tokke.

## 2. Verified data facts ✅

### 2.0 What one case "knows" (the inputs SHOP had for a week starting on day d)
| Input | Which time points | Notes |
|---|---|---|
| Day-ahead price | **all 168 hours** (d … d+6) | perfect-hindsight prices, not forecasts |
| Inflow per reservoir | **all 168 hours** | |
| Min-flow / min-volume constraints | **all 168 hours** | |
| Reservoir volume | **start of day d only** | initial state; later days = observed outcome → do not use |
| Water value | **day d** (start) and **day d+7** (end-of-week valuation) | the +7 value is the important one |
| Topology / unit data (YAML) | static | |

Kaggle gives these **inputs for the test cases too** (2023–24). Only the on/off labels are
hidden. The 725 test cases are split by Kaggle into a *public* part (scored on the live
leaderboard) and a *private* part (final ranking). Both lie inside 2023-01-01 … 2024-12-25.
How they are split (random cases or by time) is unknown; ask the course staff or check the
competition's Evaluation page.

- All files are UTC, hourly (price switches to 15-min on 2025-09-30, which is outside the test
  range; resample to hourly with `.resample('h').mean()`). There are no gaps, duplicates or
  NaNs in 2015–2024.
- Volume and water value are **daily** and go up to **2025-01-01**, which covers what the last
  test case (2024-12-25) needs at +7 days.
- Label columns are unit-major: `result_committed_<Plant>_<Unit>_t<h>`, i.e. all 168 hours of
  Hogga_G1, then all of Tokke_G1, and so on. The plant name `Vesle Kjela` contains a space.
- `total_calculation_time` exists only in train. **Do not use it as a feature.**

### 2.1 Water values are synthetic and built from a formula ✅📚
`water_value[day d, reservoir] = c × mean(price over the 168 h starting at d)`, where
c ∈ [0, 2] is a scarcity coefficient (from Reite et al.: 2 at the historical-min volume, 1 at
the median, 0 at the max). The ratio has median 1.00, the 1% quantile is 0 and the 99%
quantile is 2.00, which matches exactly.
- The **17 reservoir columns are just 9 distinct series** (exact duplicates). Deduplicate.
- **SHOP's end-of-horizon water value is the value at start + 7 days**. This is documented in
  `Dataset_definitions_and_explanation.pdf` and confirmed empirically: the rule "ON if
  price > upstream water value at +7 d" scores AUC ≈ 0.95 for several units and is much worse
  with the start-day value.
- **Water values are in EUR/MWh and directly comparable to price.** Do not convert them via
  energy equivalents.
- The +7 d water value therefore contains the *mean price of the following week*, which is
  fine: it is an official SHOP input, not leakage.
- The table is **fixed**: one number per reservoir per day, never revised. The value for a
  given Thursday is identical in every case that uses it. Each case uses two rows: start day
  (initial) and start + 7 d (end-of-week valuation).

### 2.2 Topology (from the YAML connections) ✅
| Plant (units) | Takes water from | Discharges into | p_min–p_max MW | Start cost |
|---|---|---|---|---|
| Hogga (G1) | Bandak | out of system | 3–17 | 94 |
| Tokke (G1–G4) | Vinjevatn | Bandak | 36.9–107.5 each | 214 |
| Lio (G1) | Byrtevatn | river r_Vest_Vassdraget → Bandak | 14.8–43 | 163 |
| Byrte (G1) | Botnedalsvatn | Byrtevatn | 6.9–20 | 135 |
| Vinje (G1–G3) | Vaamarvatn (tunnel from Totak) | Vinjevatn | 34.4–100 each | 198 |
| Haukeli (G1) | Vatjern | Vinjevatn | 0.3–4.9 | 143 |
| Songa (G1) | Songavatn + Bitdalsvatn (tunnels) | Totak | 41.2–120 | 222 |
| Kjela (G1) | Foersvatn | Hyljelihyl → Venemo → Totak | 20.6–60 | 171 |
| Vesle Kjela (G1) | Kjelavatn | Foersvatn | 2.9–8.5 | 92 |

Main cascade: Songa → Totak → Vaamarvatn → **Vinje → Vinjevatn (9.4 Mm³, tiny) → Tokke** →
Bandak → Hogga. Tokke can empty Vinjevatn in about 20 h, so Tokke and Vinje are tightly
coupled (label correlation 0.6–0.8). There are no min up/down times in the YAML; persistence
comes only from start/stop costs. Full details: `research/03_hydropower_domain_and_shop.md`.

### 2.3 Label structure ✅
- ON rate is about 0.5 overall, but **varies a lot by unit and year**. Examples: Haukeli 0.87
  in 2015 → 0.14 in 2022; Kjela 0.24 → 0.73 → 0.14. 2022 has the lowest rates.
- **59% of unit-weeks are constant** (32% all-ON, 27% all-OFF). Median ON run is 19 h,
  median OFF run 14 h, and 1-hour blips are rare.
- **Strict priority order within plants:**
  - Vinje G1 ≥ G2 ≥ G3: violations ≤ 0.1%.
  - Tokke G1 is always first, then G3, then G2/G4 (G2-on-while-G1-off = 0.0%).
  - So "how many units at the plant are on" is the real variable.
- **Overlapping cases are solved independently.** Case d's hours 24–167 agree with case d+1's
  hours 0–143 on only 87.8% of labels. There is no initial-state input, and the first hours
  (t0–t3) are the hardest.

### 2.4 Market regime ✅
Mean price per year: 2015–19 20–43 EUR/MWh; 2020 9; 2021 76; **2022 211**; test 2023 ≈ 79,
2024 ≈ 50. Test has about 380 negative-price hours versus about 10 in all of train.
**Use relative features** (price ÷ water value, price minus water value scaled by the week's
price level, within-week ranks and z-scores), not raw EUR.

## 3. What we measured (EDA baselines, `analysis/`) ✅

All results use year holdouts, with training cases whose window overlaps the validation year
removed (7-day embargo).

| Model | Val 2022 | Val 2021 | Val 2019 |
|---|---|---|---|
| Unit base rate | 0.555 | | |
| Raw price | 0.521 | | |
| Price − (wv_up − wv_down) at +7 d, single rule | 0.592 | | |
| LightGBM pooled, generic features (`baseline_lgbm.py`) | 0.910 | 0.951 | 0.961 |
| + physics features: price vs +7 d wv, scarcity coef, profitable-hour share (`baseline_v2.py`) | **0.970** | **0.964** | |
| Oracle that knows each unit-week's true ON fraction (level only) | 0.960 | 0.971 | |

Takeaways:
1. **Domain features matter far more than model choice:** +0.06 AUC from physics features
   alone.
2. **Getting each unit's weekly ON level right** (how much a unit runs in a given week) is
   about as important as timing within the week. Both need work.
3. **Validation years differ by about 0.01–0.05.** Never trust a single holdout year.

📚 Two other teams in this same course have public GitHub repos and self-report public-LB
scores of about 0.988–0.991 with per-unit XGBoost/LightGBM plus engineered features, and about
**0.9957** with features from an **LP relaxation** of the watercourse. Treat these as an
indication of what is achievable. **Check the course rules before looking at their code;
copying is likely not allowed.** Details: `research/02_kaggle_and_modeling_techniques.md`.

## 4. Recommended direction

**Core model: gradient-boosted trees in long format (one row = case × unit × hour), with
strong physics features.** Neural nets are a secondary, diversity member only.

Why not neural nets as the core:
- With about 2.9k cases, trees win on tabular data (Grinsztajn 2022).
- A peer team's TCN reached only 0.927.
- SINTEF's own paper (Barjaktarevic et al., EEM 2026, DOI 10.1109/eem68581.2026.11589730)
  compared only neural nets.
  - **Abstract (verified):** "convolutional architectures exhibit the most consistent
    results"; all models reached F1 > 97.35%.
  - **Full text verified** (PDF in the repo root):
    - It is the same Tokke–Vinje 3,647-case, 2,352-label dataset, generated with SHOP
      v17.15.1. Ours says v17.17.0, so it's a newer regeneration.
    - It used a **random 80/20 split** (leaky).
    - Its inputs included the **relaxed-LP (No_MIP) solution's production and UC outputs**.
  - **168 h micro-F1:** CDIL-CNN 0.9778, CNN 0.9774, TCN 0.9770, MLP 0.9762, LSTM 0.9760.
    **The relaxed LP alone scores 0.9735.**
  - The No_MIP rule over-commits (recall 0.995, precision 0.953).
  - No trees were tested.
  - **Main lesson:** a relaxed LP of the watercourse already reproduces SHOP's labels almost
    perfectly, and the networks only add a small correction on top. **Building our own LP
    relaxation as a feature (§4.1 #7) is the highest-value step.**
  - ⚠️ The paper states that the complete training dataset is public:
    *KSP InterOpt Project Open Dataset SHOP_17_15_1_Tokke_3_history*, data.sintef.no. It
    very likely contains labels for 2023–24. **Do not download or use it** without explicit
    written permission from the course staff; that would be test-label leakage.

### 4.1 Feature backlog, roughly in priority order
1. Price vs the **upstream water value at +7 d**: difference scaled by the week's price
   level, and the ratio. Add the downstream reservoir's value too (net water value).
2. **Scarcity coefficients:** c_start = wv_start / mean price of the case week; c_end = wv_+7 /
   mean price of the following week. Use the 9 deduplicated series.
3. **Within-case price shape:** rank, z-score, rolling 3/6/12/24 h means, lead/lag
   prices, and the distance to the next/previous price peak.
4. **Case-level economics per unit:** number and share of hours where the price beats the
   water value, the summed positive margin, and the length of the longest profitable block.
5. **Hydrology:** reservoir fill (vol / max_vol from the YAML) for the upstream and downstream
   reservoirs; inflow over the horizon ÷ free space (spill pressure); distance to the
   min-volume floor.
6. **Constraints:** min-flow on r_Vest_Vassdraget (likely drives **Lio**), plus
   Totak/Byrtevatn/Staavatn min-volume periods.
7. **Relaxed-optimiser features (the biggest peer gain).** Start with a per-unit two-state
   dynamic program over the 168 prices with the start cost, using the +7 d water value as the
   cost of water. That gives the optimal ON/OFF path and the value gap per hour. Later, an LP
   relaxation of the full cascade with `scipy.optimize.linprog`/HiGHS. Using topology for
   features like these is how the "extended data" should be used, rather than our own
   full reservoir optimisation.
8. **Weekly-level model:** a separate case-level model predicting each unit's weekly ON
   fraction. Feed it out-of-fold as a feature or logit offset.
9. Unit statics: p_min, p_max, start cost, priority within the plant.
10. Calendar: day of year, hour of day, day of week (weak but cheap).

### 4.2 Modelling ideas, roughly in order
- Pooled LightGBM with unit as a category versus **one model per unit**. Peers found per-unit
  slightly better; test both.
- XGBoost / CatBoost / LightGBM blends with several seeds. **Blend raw probabilities with
  global weights; never rank-normalise per unit** (that breaks micro-AUC).
- **Use the plant structure.**
  - Predict "number of units on" per plant (ordinal), then assign units by the priority order.
  - Or post-process so that P(Vinje G2) ≤ P(Vinje G1).
- **Stage 2** stacking: each unit's own predictions at ±1/3/6/12/24 h plus the other units'
  predictions.
- Temporal smoothing of predictions (rolling median/mean). Validate the window size.
- Weight recent years more; test 2015–16 down-weighting.
- k-NN over historical weeks as a cheap baseline or feature.
- Optional: a small 1D-CNN over (14 × 168) as an ensemble member.

### 4.3 How to search features (not exhaustive combinations)
Testing all combinations is infeasible: 30 features give about 10⁹ subsets at about 3 min per
3-fold run. It would also overfit the validation folds, because with thousands of comparisons
some win by luck. Use this instead:

1. **Organise features into about 8–12 named groups**, following the §4.1 backlog: `price_shape`,
   `wv_margin`, `scarcity`, `case_economics`, `hydro_fill`, `inflow`, `constraints`,
   `relaxed_dp`, `weekly_level`, `unit_static`, `calendar`. One group is one switch in config.
2. **Forward by group:** start from a base (unit + price_shape + wv_margin). Add one group per
   experiment and keep it only if it passes the §5b acceptance rule.
3. **Ablation pass** every ~10 experiments: remove one kept group at a time. If the score
   doesn't drop, remove the group (simpler wins).
4. **Inside a group:** prune with permutation importance on the validation folds, or SHAP.
   GBDTs tolerate useless features, so pruning is for robustness and speed, not score.
5. **Adversarial validation** (once, early): train a classifier to tell train cases from
   test cases using the inputs only. Features that separate them easily (e.g. raw price
   level) are regime-dependent and risky. Replace them with relative versions.
6. **Make features better, don't just select them.** The large gains so far came from
   *new* physics features (+0.06), not from choosing among existing ones.

### 4.4 Leakage-proof feature code
- **One function builds all features from a `CaseInputs` object.** It contains only the §2.0
  inputs: price, inflow and constraints for d … d+6; volume at d; water value at d and d+7;
  static YAML. The function never receives the raw data frames or labels, so it is
  structurally impossible for it to touch later volumes or answers.
- A unit test asserts that changing the data *outside* the allowed window leaves the
  features unchanged.
- Anything **learned from labels** must be fit inside each fold, on that fold's training
  part only. Examples: target or base-rate encodings, the weekly-level model, stacking
  models, blend weights (use out-of-fold predictions).
- The harness (folds, metric, embargo) is frozen; experiments can't change it (§7).

### 4.5 Spec: our own "relaxed SHOP" as a feature generator
Why: in the SINTEF paper the relaxed LP *alone* scores F1 0.9735, and NNs add only about
0.004. A peer team reported LP features as their biggest gain. This is **not** a full SHOP
rebuild. It's a rough, fast approximation whose outputs become features; the ML model learns
the corrections. It uses only the §2.0 case inputs, so it is leakage-safe.

**Level 1: per-plant dynamic program (start here, about 1–2 h of work, milliseconds per case)**
- For each plant and each hour: profit per MWh = price_t − wv_up(d+7) + wv_down(d+7). The
  water values are already in EUR/MWh.
- State = number of units on (0 … n_units; Tokke 4, Vinje 3, others 1). Running k units
  earns k × p_max × margin_t. Each start or stop costs startcost_const (from the YAML).
- Run a DP / Viterbi over the 168 hours to get the optimal on/off path.
- **Features:**
  - the DP state per hour (units on);
  - the margin;
  - the **regret**: how much worse the opposite decision would be at that hour (a soft
    confidence);
  - hours to the next and from the previous switch.
- It ignores reservoir limits (no water balance). It's fine for big reservoirs and wrong when
  a reservoir is nearly full or empty. Level 2 fixes that.

**Level 2: one LP per case over the whole cascade (about 1 day; `scipy.optimize.linprog`,
method="highs" or `highspy`; about 5k variables, well under 1 s per case, so all 3,647 cases
take minutes)**

- **Variables** (per hour t):
  - production per plant: 0 ≤ P ≤ n_units × p_max (binaries relaxed);
  - discharge per plant;
  - spill and bypass per reservoir;
  - volume per reservoir: 0 … max_vol, with the min-volume constraint.
- **Water balance:** v_{t+1} = v_t + inflow_t + upstream releases − own releases − spill.
  - Convert m³/s → Mm³/h with ×0.0036.
  - Follow the routing in the YAML `connections` (plants, tunnels, rivers).
- **Production:** P = k_plant × Q, with k ≈ 9.81 × head × η / 1000 MW per m³/s.
  - head ≈ mean reservoir level − outlet_line;
  - η ≈ 0.9, or take it from `turb_eff_curves`.
- **Other constraints:** min-flow on rivers (`Constraint_min_flow.csv`).
- **Objective:** maximise Σ price_t × P_t + end-of-week value of water left in each reservoir.
  - Convert volume to energy with the reservoir's energy equivalent to the sea (sum of
    k × head down the cascade), then multiply by wv(d+7).
  - Optionally add a start-cost penalty via a relaxed "units on" variable.
- **Features:**
  - relaxed production ÷ (p_min, p_max) per plant per hour;
  - implied units on (No_MIP-style rule: ON if production ≥ 50% of p_min, plus the raw
    fraction);
  - the LP reservoir trajectory (fill, hitting the max → spill pressure);
  - **shadow prices (duals) of the water balance**, i.e. the effective hourly water value.
    These are often better than the static water value.
- **Sanity check:** the AUC of the LP-implied UC alone over 2015–22 should be clearly above
  the single-rule baseline (0.59). Report it per unit.

**Level 3 (optional): the same model as a MILP with binaries** (`scipy.optimize.milp` or
HiGHS). It's closer to SHOP but slower; only consider it if Level 2 features plateau.

Pitfalls:
- **Units.** Volumes are in Mm³, inflow in m³/s and water values in EUR/MWh. Unit-test the
  water balance on one case.
- **Duplicated reservoirs** in the water-value file (§2.1).
- **Tiny reservoirs** (Vinjevatn, Vatjern, Hyljelihyl) make the LP stiff; use hourly
  resolution.
- **Cache** all outputs to `features/lp_*.parquet` once; never recompute inside the
  experiment loop.

**Deprioritise:** time-series foundation models (Chronos, TimesFM: we are not forecasting),
large Transformers, per-unit calibration, class weights, and averaging predictions across
overlapping cases (the labels disagree on 12%).

## 5. Validation protocol (non-negotiable)

- **Never use a random K-fold over cases.** Neighbouring cases share 6/7 days and 88% of
  labels, so a random split inflates the score heavily (SINTEF's own paper makes this
  mistake).
- **We only have labels for 2015–2022.** 2023–24 is the Kaggle test set, and Kaggle keeps both
  its public and private labels. So 2023 can **only** be checked through the public
  leaderboard, never locally.
- **Why the year split needs a buffer (embargo).** A case covers 7 days. The case starting
  2021-12-28 runs to 2022-01-03. If 2022 is the validation year, that training case contains
  SHOP's decisions for 1–3 January 2022. Those overlap almost exactly with the validation
  case starting 2022-01-01, since overlapping cases agree on about 88% of their shared hours.
  The model would effectively have seen part of the answer. The same happens at the other end:
  when 2021 is validated, the validation case starting 2021-12-31 runs to 2022-01-06, which
  overlaps the training cases starting 2022-01-01 … 2022-01-07. Therefore:
  **drop every training case whose start date is within 7 days of the validation year**, on
  both sides.
- **Folds (primary): leave one year out**, using the exact `kaggle_metric.score`. Training
  always excludes cases that start within 7 days of the validation year, on both sides.

  | Fold | Validate on | Train on |
  |---|---|---|
  | A | 2020 (very low prices) | 2015-01-01 … 2019-12-25 and 2021-01-08 … 2022-12-31 |
  | B | 2021 (rising prices) | 2015-01-01 … 2020-12-25 and 2022-01-08 … 2022-12-31 |
  | C | 2022 (crisis prices) | 2015-01-01 … 2021-12-25 |

  The three years are deliberately very different regimes, which tests generalisation.
  **Score = mean of A, B, C**, and always report each fold too.
- **Secondary check (forward-only, "expanding window"):** train only on years *before* the
  validation year (2015–19 → 2020, 2015–20 → 2021, 2015–21 → 2022). This mimics the real
  situation (train on the past, predict 2023–24). Run it for shortlisted candidates, not every
  experiment.
- **Optional extra folds:** 2018 and 2019, for a 5-fold check on final candidates (more
  stable, slower).
- **Final model:** after choosing settings with the folds, retrain on **all of 2015–2022**
  with the same settings. For early-stopped GBDTs, use about the mean best iteration from the
  folds × 1.1, since there is more data.
  - The fold models exist only to *score* settings. The "mean of A, B, C" is an average of
    **scores**, not of models.
  - Averaging *predictions* happens only for ensembles: several seeds, or different model
    types (LGBM/XGB/CNN), each retrained on all of 2015–2022. Combine them as a weighted
    average of probabilities, with the weights chosen on the out-of-fold CV predictions.
  - An alternative is averaging the three fold models' test predictions, a common Kaggle
    trick. It's weaker here because fold C never saw 2022.
- Treat differences < 0.001–0.002 as noise; measure seed variance early.
- **Early stopping:** on an inner time-tail of the training folds, never on the validation
  year itself.
- **Locked holdout:** if running a long autonomous loop, keep one extra year (e.g. 2018, or the
  second half of 2022) that the loop never sees, and score candidates on it only at the end.
  Over a 24 h search the agent *will* overfit its validation folds (Meta AIRA findings).
- **Kaggle submissions:** humans choose them, a few per day, as a sanity check against the
  public LB (the 2023–24 regime differs from every validation year).

## 5b. Model selection and targets (how to decide what is "better")

The final ranking uses Kaggle's **private** test labels, which nobody sees. Everything we
decide must therefore generalise, not chase a number.

**Leakage, precisely:**
- Test *labels* cannot leak; we don't have them, public or private.
- What *can* go wrong:
  - (a) leaky validation (random split, no buffer) gives a misleadingly high local score;
  - (b) inputs a real system wouldn't have, especially observed reservoir volumes after the
    start day;
  - (c) external data that contains 2023–24 answers;
  - (d) overfitting the public leaderboard through many submissions.
- (a)–(c) are forbidden by §5/§6. (d) is controlled by the rules below.

**Acceptance rule for a change** (feature, model, parameter):
1. The mean leave-one-year-out AUC (folds A–C) improves by **≥ 0.001**, **and**
2. it improves (or is within noise) on **at least 2 of 3** folds; no fold may drop
   > 0.002, **and**
3. it is not a narrow trick. Prefer changes with a physical or economic reason (e.g. a
   water-value margin) over hyperparameter fiddling. Among equal scores, choose the simpler
   model.

**Targets (realistic, honest):**
- Local mean A–C ≈ **0.975–0.985** is a strong, honest result. Baseline v2 is about 0.967.
  The 2022 fold will likely stay the lowest.
- Public LB will probably be somewhat **higher** than local CV, since 2023–24 prices are
  closer to normal than 2022. Peers report about 0.99; 0.995+ came with LP-relaxation
  features.
- Do **not** set a leaderboard number as the goal. The goal is the highest and most
  *stable* local CV.

**Using the public leaderboard:**
- Submit sparingly (e.g. ≤ 2 per day). Treat it as a sanity check for whether local CV gains
  carry over to 2023–24.
- **Warning sign:** CV improves but LB drops, or the reverse, across several submissions.
  That suggests the change is fold-specific. Prefer the CV signal.
- Whether public/private split by random days or by time (2023 vs 2024) is unknown. If it's by
  time, the public LB is an even weaker guide for the private score.

**Final submissions** (Kaggle usually lets you pick 2):
1. The best mean CV (A–C) that also holds up in the forward-only check.
2. A robust choice: a simple multi-seed blend of the top 2–3 diverse models (e.g. LGBM +
   XGB + optional CNN), even if its CV is slightly below #1.

Never pick a submission because it has the highest public LB alone.

## 6. Pitfalls checklist

- [ ] Use the +7 d water value, not the start-day value.
- [ ] Don't use raw EUR price levels as the main signal, because of regime shift and negative
      prices in test.
- [ ] Don't use future *observed* reservoir volumes (days d+1…d+7 of
      `Historical_volume`). SHOP only knew the initial volume, so later volumes are leakage.
- [ ] Don't use `total_calculation_time`.
- [ ] No external SINTEF datasets that might contain 2023–24 labels (check the course rules).
- [ ] Submission: `Run No` + 2352 columns in exactly the `sample_submission.csv` order,
      725 rows, probabilities in [0, 1], no NaN. Validate with `kaggle_metric.score` on a fake
      solution before uploading.
- [ ] Memory: the long format is about 6.9M rows. Keep features in float32 and subsample rows
      when iterating. The machine is an M3 with 16 GB RAM, so per-unit models (~490k rows
      each) are lighter.
- [ ] The base conda environment is broken (NumPy 1/2 mix). Use the project venv `.venv/`.

## 7. Plan for the long autonomous run (Part 2)

See `research/04_autonomous_experiment_loop.md` for sources and a full scaffold. Summary:

1. **A human sets up the harness first** (about 1–2 h, supervised):
   - `harness/folds.py` + `harness/evaluate.py` (frozen) and a timeout wrapper.
   - `experiments/leaderboard.csv`, written only by the evaluator, and `experiments/journal.md`.
   - `src/` is the only directory the agent edits.
   - Git init and one commit per experiment.
2. Protect the harness:
   - Use a `PreToolUse` hook that blocks edits to `harness/` and `data/`, plus the sandbox.
     Prompt text alone is not enough; agents have edited their own timeouts.
   - Use `acceptEdits` or auto mode with an explicit allow list. ⚠️ `bypassPermissions` is
     only for throwaway containers or VMs.
3. Choose a driver:
   - `/loop` with `.claude/loop.md` (re-read every iteration).
   - Or, more robust, a shell loop around `claude -p` so each iteration starts with a fresh
     context and all state lives on disk.
   - On Max, expect usage-limit pauses (5-hour windows plus weekly caps), so the driver must
     back off and retry.
   - Keep the Mac awake with `caffeinate -i`.
   - Verify CLI flags against `claude --help` before relying on them.
4. Rough schedule:
   - 0–3 h: diverse baselines.
   - 3–18 h: one-hypothesis-per-experiment improvements, working down the §4.1 backlog.
   - 18–22 h: ensembling and multi-seed runs.
   - Then a human scores the locked holdout and makes the final submissions.

### Useful Claude Code skills
- **Built-in:**
  - `/loop`: the self-paced experiment loop.
  - `/schedule`: cloud routines. These get a fresh git clone without local data, so they're a
    poor fit unless the data is in the repo.
  - `/code-review` or `/simplify`: periodic review of `src/` for leakage bugs.
  - `dataviz`: diagnostic plots.
  - `update-config`: hooks and permissions for unattended runs.
  - `fewer-permission-prompts`: build an allow list after a supervised dry run.
- **Worth creating with `skill-creator`** (project skills in `.claude/skills/`):
  - `run-experiment`: the exact steps to run, log, keep or revert.
  - `check-submission`: format validation plus a fake-solution score.
  - `leakage-audit`: a checklist from §5–6 that runs against a diff.

## 8. Open questions
- ❓ What drives Lio (the min-flow constraint on r_Vest_Vassdraget?) and Haukeli (a tiny unit,
  close to run-of-river)? These are the weakest units at about 0.90–0.92.
- ❓ Does the cheap DP/LP relaxation reproduce SHOP well enough to be a strong feature?
  The peer evidence says yes.
- ❓ How do the 2023–24 public-LB scores relate to our CV folds? Calibrate with 1–2 early
  submissions.
- ❓ Course rules on using other teams' public repos and external SINTEF data.
