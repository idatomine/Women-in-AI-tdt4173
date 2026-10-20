# Briefing: SINTEF unit-commitment prediction (TDT4173)

Handoff document for whoever (human or Claude session) trains and tunes the model. It
summarises what is **verified in the data** (✅), what comes from **research** (📚, see
`research/0*.md` for sources), and what is still a **hypothesis** (❓).

Status 2026-10-16: Part 1 (understanding + direction) done. No final pipeline exists yet;
`analysis/` holds throwaway EDA/baseline scripts only.

**Revised 2026-10-16.** The main track is now **relaxed-MIP/LP features + GBDT** (§4.0).
Other changes:
- the NN assessment is corrected (§4);
- validation adds 2019 as fold D, with 2018 as a locked holdout (§5);
- the oracle numbers are reconciled (§3);
- the autonomous loop is replaced by a lean manual plan with a timeline (§7).

**Revised 2026-10-19.** The §3 single-rule baseline is corrected (upstream-only ≈ 0.85–0.91,
not 0.59), the DP margin in §4.5 now uses the upstream water value only, and findings from
`analysis/01_eda.ipynb` are added (§2, §2.3, §4.1 #5, §4.2, §6).

**Deadlines:**
- VTs are frozen after **2026-10-18**.
- Project deadline: **2026-11-08 22:00** (strict; late = −30 points).

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
- From `analysis/01_eda.ipynb`:
  - **Volumes go above `max_vol`** in 14 of 17 reservoirs, by up to 18% (Bandak), on 0–7%
    of days (floods). It's real, so don't clip the feature. The LP's volume bounds must
    allow it, or start states above `max_vol` make the LP infeasible.
  - **The Hyljelihyl and Vatjern volumes are constant** (placeholders, no records). Drop
    them as features and don't trust them as LP start states.
  - Several **inflow series are scaled copies** of each other (correlation ≈ 1); a few
    representative series carry the information.
  - Water-value deduplication (keep → drop): Kjelavatn → Staavatn; Langesae → Foersvatn,
    Bordalsvatn; Venemo → Vaamarvatn, Totak, Hyljelihyl; Bitdalsvatn → Songavatn;
    Langeidvatn → Vatjern.

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
  (t0–t3) are the hardest. Tokke and Vinje are the least stable (≈ 82% agreement), the
  small plants the most (≈ 95%).
- **Possible label noise (EDA §2.2.3):** slow solver runs come in clusters of consecutive
  start days, with a spike at ≈ 1000 s that looks like a time limit, so those labels may be
  suboptimal. It's worst in late 2022.

### 2.4 Market regime ✅
Mean price per year: 2015–19 20–43 EUR/MWh; 2020 9; 2021 76; **2022 211**; test 2023 ≈ 79,
2024 ≈ 50. Test has about 380 negative-price hours versus about 10 in all of train.
**Use relative features** (price minus water value scaled by the week's price level,
within-week ranks and z-scores), not raw EUR.

⚠️ **Prefer scaled differences over ratios.** Price ÷ water value blows up when the water
value is near 0 (2020, and scarcity coef ≈ 0 at full reservoirs) and flips sign with
negative prices (test). If ratios are used, clip them, and always keep the
difference ÷ week-level version alongside.

## 3. What we measured (EDA baselines, `analysis/`) ✅

All results use year holdouts, with training cases whose window overlaps the validation year
removed (7-day embargo).

| Model | Val 2022 | Val 2021 | Val 2019 |
|---|---|---|---|
| Unit base rate | 0.555 | | |
| Raw price | 0.521 | | |
| **Price − upstream wv at +7 d**, single rule | **0.911** | **0.849** | **0.863** |
| Price − (wv_up − wv_down) at +7 d, single rule | 0.592 | 0.591 | 0.566 |
| LightGBM pooled, generic features (`baseline_lgbm.py`) | 0.910 | 0.951 | 0.961 |
| + physics features: price vs +7 d wv, scarcity coef, profitable-hour share (`baseline_v2.py`) | **0.970** | **0.964** | |
| Oracle that knows each unit-week's true ON fraction (level only) | 0.960 | 0.971 | |

An independent re-implementation (forward-only folds, per-unit LightGBM, ~115 generic
features, water value as the *average* of start and +7 d) scored **0.941 (2021) / 0.927
(2022)**. That confirms that using the +7 d value directly, not a blend, is worth about
+0.03.

⚠️ **The two single-rule rows (corrected 2026-10-19).** Both are one number per row,
ranked across all units and hours together (micro-AUC):
- **Upstream only** (price − the water value of the reservoir the plant takes water from)
  is the strong rule: 0.85–0.91 pooled, and 0.73–0.996 per unit in 2022 (EDA §4.2).
- **Subtracting the downstream value** makes it much worse (≈ 0.59; Vinje 0.33 per unit,
  worse than random). Every water value is ≈ c × next week's mean price, so
  wv_up − wv_down is the difference of two similar numbers and close to 0. Then
  price − (wv_up − wv_down) ≈ price, and raw price carries almost no signal.
- An earlier version of this table listed only the net rule (0.592) as "the single rule",
  which understated the baseline. Use the upstream rule as the bar to beat.

**Harness results (2026-10-19, details in `experiments/journal.md`).** Mean AUC over folds
A–D (2020, 2021, 2022, 2019):
- exp 001, baseline_v2 in the harness: 0.9593. Seed noise: ≤ 0.0017 per fold, ≤ 0.0008 on
  the mean.
- exp 002, raw EUR levels removed: 0.9642 (2020 +0.016). Kept.
- exp 007, + spill pressure: 0.9652. Kept.
- exp 008, + per-plant DP (§4.5 Level 1): 0.9663. Kept; DP regret is the top feature by gain.
- **exp 009, + Tokke–Vinje coupling: 0.9789.** Kept. When Vinje G1 runs, Tokke G1 runs 99.5%
  of the time (Vinje's water passes through the tiny Vinjevatn), otherwise 37%. Tokke
  G2–G4 gained 0.02–0.08.
- No gain: margin in neighbouring hours (005), profitable-block lengths (006), one model per
  unit (010, −0.0016). Cascade coupling for the other plants (011): Hogga +0.02–0.04 on all
  folds, pooled within noise; pending.
- Lesson: re-expressing the margin adds nothing the trees can't combine. Gains come from new
  information, i.e. the physics (DP, coupling, reservoirs). That supports the LP track.
- exp 012, XGBoost instead of LightGBM (same features): 0.9780 vs 0.9789, within noise and
  2× slower. LightGBM stays the main model; XGBoost is in `src/models.py` for the report
  and a blend.
- **exp 013, + relaxed LP of the whole watercourse (`src/features/relaxed_lp.py`): 0.9829.**
  Kept; `lp_margin` (from the LP duals) is the #1 feature. Gains on Hogga (+0.03), Haukeli
  (+0.05), Kjela (+0.02).

**LP lesson: the synthetic water values aren't consistent across reservoirs.** Valuing
leftover water physically ("sea": energy down to the sea × the reservoir's own water
value) failed the sanity check (LP AUC ≈ 0.70 vs 0.886 for the simple rule): Vinje's
trade-off becomes 1.93·price + 3.5·wv_Vinjevatn − 5.43·wv_Vaamarvatn, which amplifies the
difference between two series that only correlate 0.82 (AUC 0.52). The labels instead
follow "price vs the plant's **own upstream** water value" for every plant (AUC 0.86–0.99),
except Vinje, which follows Vinjevatn (0.88 vs 0.76 for its own Vaamarvatn). The working
**"path" valuation** prices each plant's leg down the cascade at that plant's own upstream
water value, so every plant's LP trade-off is that rule, while the LP still enforces the
water balance. Sanity check with "path": LP margin 0.905 mean over units (Tokke
0.92–0.93, Vinje 0.88–0.91).

Later in the same round:
- Ablation (exp 014a–c): the DP (−0.0011 without it) and the Tokke–Vinje coupling (−0.0071)
  are still needed next to the LP; spill pressure isn't (−0.0002), so it was removed.
- exp 015: pricing Vinje's LP leg at Vinjevatn's water value made the LP *worse* at the
  sanity check (Tokke 0.88–0.91, Vinje 0.85–0.88); not run.
- exp 016, more capacity (60% of rows, 1500 trees): −0.0001 at 3× the time. The model is
  feature-limited, not capacity-limited.
- **exp 017, LP with the six min-flow constraints (soft): 0.9840.** Kept. LP "Lio on" AUC
  0.853 → 0.955 at the sanity check, and Lio +0.02–0.07 in the model: the min flow on
  r_Vest_Vassdraget drives Lio.
- Per-unit AUC at exp 017 (mean over folds): Haukeli 0.907 is the weakest by far, then
  Vinje G3 0.963; all others ≥ 0.975.
- Public LB for exp 017: **0.99042** (VT1, the highest VT: 0.99040). Local CV 0.9840.
- No gain: smoothing predictions within a case (exp 018, worse at every window), a
  LightGBM + XGBoost blend (exp 020, the two make the same errors), Haukeli's real storage
  lake Langeidvatn as a feature (exp 019; the LP already has it), enforcing the plant
  priority order (exp 022; the model already respects it).
- **exp 021, LP with a relaxed unit commitment (units on, p_min, start/stop costs): 0.9850.**
  Kept after a two-seed check (+0.0010, all folds up).

Two more checks from the same session:
- Averaging predictions over overlapping cases **hurts**: 0.934 → 0.926 at decay 0.5, and
  → 0.897 with equal weights.
- Future observed volumes (forbidden anyway) carry almost no signal, with single-feature
  AUC ≈ 0.45–0.60.

Takeaways:
1. **Domain features matter far more than model choice:** +0.06 AUC from physics features
   alone.
2. **Weekly level and within-week timing both matter, and timing is not a detail.**
   Knowing each unit-week's true ON fraction alone gives only 0.960–0.971 (measured twice,
   independently). So even a perfect level model caps far below the 0.99+ on the LB, and
   the remaining gap is hour-level timing. This is what optimisation features (DP/LP) supply.
   - ⚠️ `research/02` quotes "oracle level ≈ 0.997" from a peer repo. That contradicts our
     measurement. It is most likely level + the model's within-week ranking, not level
     alone. Trust our number.
3. **Validation years differ by about 0.01–0.05.** Never trust a single holdout year.

📚 **Leaderboard picture (public LB, 2026-10-16):** #1 is 0.99768, the top 10 are all
≥ 0.9961, #25 is 0.9948 and #50 is about 0.990.
- The leader ([67]) told us verbally that they solve a **relaxed MIP of the watercourse** and
  feed the solution into LightGBM/XGBoost (see §4.0).
- None of the top-10 teams has a public repo for this task (checked by Kaggle member name
  and team name).
- Public peer repos self-report about 0.988–0.991 with per-unit XGBoost/LightGBM plus
  engineered features, and **0.9957** with LP-relaxation features ([119]).
- Never copy code from other teams. Use these numbers only as an indication of what is
  achievable. Details: `research/02_kaggle_and_modeling_techniques.md`.

## 4. Recommended direction

### 4.0 Priority update (2026-10-16): relaxed-MIP/LP features are the main track

**Evidence:**
- **The LB leader 📚 (verbal, not verified in code):** [67] SIX-SEVEN (0.99768) solves a
  **relaxed MIP** of the watercourse and feeds its solution into LightGBM/XGBoost.
- **SINTEF's paper:** the relaxed solution alone (No_MIP) gives micro-F1 0.9735 at 168 h.
  NNs that also see the No_MIP output only reach 0.9778, and that is with a random split,
  so it is optimistic.
- **The [119] peer repo** (0.9957 public) reports LP features as its biggest gain.
- **Our oracle numbers (§3):** within-week timing is the remaining gap, and an optimiser
  supplies exactly that.

**Conclusion:** the score comes from the optimisation features, not the model type.
- §4.5 Level 2 moves from "later" to **the main track**.
- Level 1 (per-plant DP) is a quick stepping stone (≤ 1 day) that also gives a feature in
  its own right. It is not the destination.

**What "relaxed" means:** SHOP solves a MIP in which each unit's on/off status is binary.
Relaxing it lets on/off take any value in [0, 1], which turns the problem into an LP that
solves in well under a second per case.
- The LP solution is *almost* right: a unit that runs in the MIP produces in the LP, and a
  unit that is off produces about 0.
- The GBDT then learns the corrections. For example, the LP over-commits: No_MIP has
  recall 0.995 and precision 0.953.

**Build order (risk-staged):**
1. **Level 1 DP** (§4.5): half a day. Measure it.
2. **LP for Tokke + Vinje only**: 7 of 14 units, coupled via the tiny Vinjevatn (9.4 Mm³).
   Reservoirs: Vaamarvatn, Vinjevatn and Bandak.
   - **Sanity check:** the AUC of LP production ÷ p_max against the labels for these 7
     units, per year. It must be clearly above the upstream single rule for the same units
     (per unit: Tokke ≈ 0.96, Vinje ≈ 0.82–0.84 in 2022), and ideally above the DP. If not,
     fix the LP before adding units.
3. **Full cascade** following the YAML `connections`, plus min-flow / min-volume.
4. Optional: a relaxed start-cost term, to capture the persistence SHOP gets from start
   costs.
5. Optional: a real MILP (Level 3) only if Level 2 plateaus.

**Evaluation:** for every fold, report three numbers:
- (a) the LP rule alone;
- (b) GBDT without LP features;
- (c) GBDT with LP features.

Keep LP features only if (c) beats (b) under §5b.

**Runtime check:** the LP over all 3,647 cases must fit well inside the short notebook's
12 h / 4-core limit. Measure seconds per case on day one.

### 4.0b Core model and the role of neural nets

**Core model: gradient-boosted trees (LightGBM/XGBoost) on top of physics + LP features.**
The leader and both public peer repos use GBDTs.

The NN question, assessed fairly:
- With about 2.9k cases, trees are the safe default on tabular data (Grinsztajn 2022).
- A peer team's TCN reached only 0.927. ⚠️ That is *not* evidence against NNs as such: a
  GBDT with generic features scores about the same (0.927–0.941, §3). It is evidence
  against models without physics features.
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
    perfectly, and the networks only add a small correction on top (+0.004 F1). The paper
    shows *the LP* is strong. It says nothing about NN vs trees. **Building our own LP
    relaxation as a feature (§4.0, §4.5) is the highest-value step.**
- **Verdict:** NNs are optional. Add a small 1D-CNN over (channels × 168 h) only if time
  allows after the LP track, and only fed the **same physics/DP/LP channels** as the
  GBDT. Its value is as a diverse ensemble member (errors differ from per-hour trees). Keep
  it only if it reaches about 0.97 alone and improves the blend out-of-fold.
- ⚠️ The paper states that the complete training dataset is public:
  *KSP InterOpt Project Open Dataset SHOP_17_15_1_Tokke_3_history*, data.sintef.no. It
  very likely contains labels for 2023–24. **Do not download or use it** without explicit
  written permission from the course staff; that would be test-label leakage.

### 4.1 Feature backlog, roughly in priority order
**#7 (relaxed optimiser) is the main track (§4.0)** and runs in parallel with #1–#6, which
are cheap. Get #1–#4 in first: they are the base the LP features are measured against.

1. Price vs the **upstream water value at +7 d**: the difference scaled by the week's price
   level, and a clipped ratio (see §2.4). The downstream reservoir's value can be added as
   an *extra* feature, but never in place of the upstream margin: the net version alone is
   much weaker (§3).
2. **Scarcity coefficients:** c_start = wv_start / mean price of the case week; c_end = wv_+7 /
   mean price of the following week. Use the 9 deduplicated series.
3. **Within-case price shape:** rank, z-score, rolling 3/6/12/24 h means, lead/lag
   prices, and the distance to the next/previous price peak.
4. **Case-level economics per unit:** number and share of hours where the price beats the
   water value, the summed positive margin, and the length of the longest profitable block.
5. **Hydrology:** reservoir fill (vol / max_vol from the YAML) for the upstream and downstream
   reservoirs; inflow over the horizon ÷ free space (spill pressure); distance to the
   min-volume floor. **The unit's own upstream fill is strong** (EDA §4.3: Songa ON 19% of
   the time in the emptiest 10% of weeks vs 93% in the fullest; Kjela 7% → 76%), while
   system-wide fill and inflow are weak. Get it in early, with #1–#4.
6. **Constraints:** min-flow on r_Vest_Vassdraget (likely drives **Lio**), plus
   Totak/Byrtevatn/Staavatn min-volume periods.
7. **Relaxed-optimiser features: the main track (§4.0, spec in §4.5).** This is the
   biggest gain for the leader and for peers.
   - Step 1: a per-plant DP over the 168 prices with the start cost, using the +7 d water
     value as the cost of water. It gives the optimal path and the value gap per hour.
   - Step 2: an LP relaxation of the cascade with `scipy.optimize.linprog`/HiGHS, Tokke +
     Vinje first, then the full cascade.
   - Using topology for features like these is how the "extended data" is meant to be used.
8. **Weekly-level model:** a separate case-level model predicting each unit's weekly ON
   fraction. Feed it out-of-fold as a feature or logit offset. **Lower priority than #7.**
   The level-only oracle caps at 0.96–0.97 (§3), and the LP already encodes level and
   timing jointly.
9. Unit statics: p_min, p_max, start cost, priority within the plant.
10. Calendar: day of year, hour of day, day of week (weak but cheap).

### 4.2 Modelling ideas, roughly in order
- Pooled LightGBM with unit as a category versus **one model per unit**. Peers found per-unit
  slightly better; test both.
- XGBoost / CatBoost / LightGBM blends with several seeds. **Blend raw probabilities with
  global weights; never rank-normalise per unit** (that breaks micro-AUC).
- **Use the plant structure.** It deserves a proper test; one peer attempt is not enough
  to rule it out.
  - The priority order is near-strict (§2.3), so the real target for Tokke and Vinje is
    "number of units on".
  - Model it as cumulative binaries P(n ≥ k) for k = 1…4 (Tokke) and 1…3 (Vinje).
    P(G-k on) then follows from the priority order, and monotonicity is guaranteed by
    construction.
  - G2/G3 then learn from all of the plant's data instead of only their own sparse ON
    hours.
  - Cheap fallback: post-process so that P(Vinje G2) ≤ P(Vinje G1).
- **Stage 2** stacking: each unit's own predictions at ±1/3/6/12/24 h plus the other units'
  predictions.
- Temporal smoothing *within* a case (a rolling mean over that case's own hours). Validate
  the window size. Do not average *across* overlapping cases; that was measured to hurt
  (§3).
- Weight recent years more; test 2015–16 down-weighting.
- Down-weight cases from slow solver runs (possible label noise, §2.3). This uses
  `total_calculation_time` as a *training weight* only, never as a feature, so it's
  allowed; validate it like any other change.
- k-NN over historical weeks as a cheap baseline or feature.
- Optional, after the LP track: a small 1D-CNN with the same feature channels (§4.0b).

### 4.3 How to search features (not exhaustive combinations)
Testing all combinations is infeasible: 30 features give about 10⁹ subsets at about 3 min per
4-fold run. It would also overfit the validation folds, because with thousands of comparisons
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
0.004. A peer team reported LP features as their biggest gain, and the LB leader uses
relaxed-MIP features (§4.0). This is **not** a full SHOP
rebuild. It's a rough, fast approximation whose outputs become features; the ML model learns
the corrections. It uses only the §2.0 case inputs, so it is leakage-safe.

**Level 1: per-plant dynamic program (start here, about 1–2 h of work, milliseconds per case)**
- For each plant and each hour: profit per MWh = price_t − wv_up(d+7). The water values are
  already in EUR/MWh. Don't add back wv_down: the net margin scores ≈ 0.59 against ≈ 0.85–0.91
  for upstream only (§3). Test the net version as a variant at most.
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

**Level 2 (MAIN TRACK): one LP per case over the cascade (about 1–2 days;
`scipy.optimize.linprog`, method="highs" or `highspy`; about 5k variables, well under 1 s per
case, so all 3,647 cases take minutes).** Build it for **Tokke + Vinje first** and pass the
sanity check (§4.0) before extending it to the full cascade.

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
  the upstream single-rule baseline (pooled ≈ 0.85–0.91, §3). Report it per unit.

**Level 3 (optional): the same model as a MILP with binaries** (`scipy.optimize.milp` or
HiGHS). It's closer to SHOP but slower; only consider it if Level 2 features plateau.

Pitfalls:
- **Units.** Volumes are in Mm³, inflow in m³/s and water values in EUR/MWh. Unit-test the
  water balance on one case.
- **Duplicated reservoirs** in the water-value file (§2.1).
- **Tiny reservoirs** (Vinjevatn, Vatjern, Hyljelihyl) make the LP stiff; use hourly
  resolution.
- **Cache** all outputs to `features/lp_*.parquet` once; never recompute inside the
  experiments.

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
  | D | 2019 (normal prices) | 2015-01-01 … 2018-12-25 and 2020-01-08 … 2022-12-31 |

  **Score = mean of A–D**, and always report each fold too.

  Why D was added: A–C are all extreme regimes, and none of them resembles the test period
  (2023 ≈ 79 and 2024 ≈ 50 EUR/MWh, with negative-price hours). The 2022 fold in
  particular can dominate decisions. A "normal" year keeps the average from steering
  towards what only helps in crisis or near-zero-price years.
- **Secondary check (forward-only, "expanding window"):** train only on years *before* the
  validation year (2015–19 → 2020, 2015–20 → 2021, 2015–21 → 2022). This mimics the real
  situation (train on the past, predict 2023–24). Run it for shortlisted candidates, not every
  experiment.
- **Locked holdout: 2018.** It is never used while choosing features or settings. Score it
  once, on the 2–3 final candidates only, as an honest check for overfitting the A–D folds.
- **Final model:** after choosing settings with the folds, retrain on **all of 2015–2022**
  with the same settings. For early-stopped GBDTs, use about the mean best iteration from the
  folds × 1.1, since there is more data.
  - The fold models exist only to *score* settings. The "mean of A–D" is an average of
    **scores**, not of models.
  - Averaging *predictions* happens only for ensembles: several seeds, or different model
    types (LGBM/XGB/CNN), each retrained on all of 2015–2022. Combine them as a weighted
    average of probabilities, with the weights chosen on the out-of-fold CV predictions.
  - An alternative is averaging the three fold models' test predictions, a common Kaggle
    trick. It's weaker here because fold C never saw 2022.
- Treat differences < 0.001–0.002 as noise; measure seed variance early.
- **Early stopping:** on an inner time-tail of the training folds, never on the validation
  year itself.
- **Kaggle submissions:** humans choose them, a few per day, as a sanity check against the
  public LB (the 2023–24 regime differs from every validation year).
  - **Submit the baseline_v2 equivalent early (day 1).** That gives one (local CV, public
    LB) pair to calibrate against.
  - Then submit after each big step (DP, LP), so you can see whether local gains carry over
    to 2023–24.

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
1. The mean leave-one-year-out AUC (folds A–D) improves by **≥ 0.001**, **and**
2. it improves (or is within noise) on **at least 3 of 4** folds; no fold may drop
   > 0.002, **and**
3. it is not a narrow trick. Prefer changes with a physical or economic reason (e.g. a
   water-value margin) over hyperparameter fiddling. Among equal scores, choose the simpler
   model.

**Targets (realistic, honest):**
- Local mean A–D ≈ **0.975–0.985** is a strong, honest result. Baseline v2 is about 0.967.
  The 2022 fold will likely stay the lowest.
- Public LB will probably be somewhat **higher** than local CV, since 2023–24 prices are
  closer to normal than 2022.
- LB reference points (2026-10-16): #50 ≈ 0.990, #25 ≈ 0.995, the top 10 ≥ 0.996, and the
  leader 0.9977, with relaxed-MIP features.
- Expect: physics features alone give about 0.99 public. Reaching 0.995+ requires the LP
  track.
- Base points depend on how many VTs we beat, and the VTs are frozen after 2026-10-18.
  Watch where the VT entries sit on the LB.
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
1. The best mean CV (A–D) that also holds up in the forward-only check.
2. A robust choice: a simple multi-seed blend of the top 2–3 diverse models (e.g. LGBM +
   XGB + optional CNN), even if its CV is slightly below #1.

Pick two that **differ**: two near-identical models fail on the same cases, which wastes
the second pick as insurance. Both need their own short notebook that reproduces them from
raw data (fixed seeds, ≤ 12 h on 4 cores / 32 GB).

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
- [ ] Ratio features (price ÷ water value) blow up near zero and flip with negative prices.
      Clip them and keep scaled-difference versions (§2.4).
- [ ] Don't average predictions across overlapping cases. It was measured to hurt (§3).
- [ ] LP volume bounds must allow start volumes above `max_vol` (floods, §2), and must not
      rely on the constant Hyljelihyl/Vatjern volumes.
- [ ] Use the **upstream** water value as the core margin. The net (upstream − downstream)
      margin alone is much weaker (§3).
- [ ] The LP must use only §2.0 inputs: start-day volume, the +7 d water value, and prices
      and inflows inside the case. Assert this in a unit test, like the feature code (§4.4).

## 7. Plan and timeline (Part 2)

**Decision (2026-10-16): no long autonomous loop.** The earlier plan (an unattended 24 h
`/loop` with hooks, see `research/04_autonomous_experiment_loop.md`) is dropped:
- It costs the most usage, and we have a usage budget to protect.
- It overfits the validation folds by construction: with thousands of comparisons, some win
  by luck.
- The gains we need come from 3–4 well-defined features (DP, LP), not from a broad search.

Work instead as **targeted manual experiments**: one hypothesis, one run on folds A–D, one
line in `experiments/journal.md` (date, change, per-fold AUC, keep/drop and why).

**Timeline** (deadline 2026-11-08 22:00; VTs frozen after 2026-10-18):

| Dates | Work | Done when |
|---|---|---|
| Oct 16–18 | Harness: folds A–D + locked 2018, `kaggle_metric.score`, journal. Reproduce baseline_v2. Backlog #1–#6. **First Kaggle submission.** | One (CV, public LB) pair |
| Oct 19–21 | Level 1 DP features (§4.5). Submit. | DP gain measured on A–D |
| Oct 22–28 | Level 2 LP: Tokke + Vinje, sanity check, then the full cascade. Cache features. Submit. | LP rule AUC per unit; GBDT with vs without LP |
| Oct 29–Nov 1 | Models on the final features: per-unit LightGBM vs XGBoost, the plant-count variant (§4.2), multi-seed blend. Optional CNN. | Two candidate submissions |
| Nov 1–7 | Report (EDA, ≥ 2 model types, feature engineering, interpretation/SHAP), two short notebooks, a clean-environment reproducibility run, the locked 2018 check, final picks. | All four −3 items covered; both notebooks reproduce |
| Nov 8 | Buffer only. Upload to Canvas well before 22:00. | |

**Report material is collected as you go:** EDA figures from the harness week, the
per-step score table from the journal, and SHAP on the final model. The four −3 deductions
are the cheapest points in the project.

**Saving usage when working with Claude:**
- Use one task per session, and point it to this file.
- Keep all state on disk (journal, cached features), not in the chat.
- Run heavy training locally and paste only the results.

### Useful Claude Code skills
- `/code-review` or `/simplify`: periodic review of `src/` for leakage bugs.
- `dataviz`: diagnostic plots and report figures.
- **Worth creating with `skill-creator`:**
  - `check-submission`: format validation plus a fake-solution score;
  - `leakage-audit`: the §5–6 checklist run against a diff.

## 8. Open questions
- ✅ Lio is driven by the min flow on r_Vest_Vassdraget (exp 017). ❓ What drives Haukeli (a
  tiny unit, close to run-of-river)? Now the weakest unit at 0.907.
- ❓ Does the cheap DP/LP relaxation reproduce SHOP well enough to be a strong feature?
  The peer evidence and the LB leader say yes. Answer it with the Tokke + Vinje sanity check
  (§4.0).
- ❓ Is a linear production function (P = k × Q at a fixed head) close enough, or is a
  head-dependent piecewise version needed? Try a fixed head first.
- ❓ Which are stronger as features: the LP duals (the hourly water value) or the primal
  production?
- ❓ Does the plant-count target (cumulative P(n ≥ k)) beat 14 separate unit models
  (§4.2)?
- ❓ Which leaderboard entries are the VTs, and how many are there? This decides how many
  base points each step is worth.
- ❓ How do the 2023–24 public-LB scores relate to our CV folds? Calibrate with 1–2 early
  submissions.
- ❓ Course rules on using other teams' public repos and external SINTEF data, and whether
  the public/private split is random or by time. **Ask the course staff.**

## 9. Handoff: LP accuracy loop (2026-10-19, in progress)

**Where we are.** Best model: exp 021 (`configs/exp_021_lp_commitment.yaml`), local CV
**0.9850**, public LB **0.99067** (VT1, the highest VT: 0.99040; leader 0.99795). Details per
run in `experiments/journal.md`. Model tuning, blends, smoothing and stacking-style
post-processing no longer help (exps 016, 018, 020, 022); every real gain has come from
making the LP more like SHOP (water balance +0.004, min flow +0.0013, start costs +0.001).

**Why the LP.** SHOP's own relaxed solution reproduces the labels with F1 ≈ 0.97; our LP's
best signal reaches AUC ≈ 0.92 per unit. Closing that gap is the most likely way to close
the gap to the leader. SHOP itself is not an option for the deliverable (license, and the
short notebooks must run offline in ≤ 12 h on a normal PC).

**Code state.** `src/features/relaxed_lp.py` was rewritten with three new options, all off
by default. With the defaults it reproduces exp 021's LP exactly (regression check:
max difference ≈ 1e-7). The new options are implemented but **not yet tested**:
- `production_curves=True`: production as a concave piecewise-linear function of
  discharge (4 pieces), from the YAML turbine efficiency curves (nearest reference head),
  generator efficiency curve and head losses (`main_loss`, `penstock_loss`), for 1 … n
  units sharing the flow. Replaces one constant k at the best efficiency, which overrates
  part-load running.
- `flow_costs="yaml"`: rivers cost their `flow_cost_const` (spill 1000, bypass 10 per
  m³/s·h) instead of 0.01, so spilling is a last resort as in SHOP. Likely matters for the
  negative-price hours in 2023–24 (the LP can currently dump water for free).
- `head_iterations=1`: re-solve with each reservoir's head at its mean level in the first
  solution (SHOP's successive linearisation, simplest form).

**The loop (fast, ~30 s per variant).** `analysis/lp_loop.py` scores an LP variant's own
signals against the labels on every 6th train case, without training any model:

    PYTHONPATH=$PWD .venv/bin/python analysis/lp_loop.py '{"min_flow": true, "commitment": true, "production_curves": true}'

Baseline to beat (exp 021's LP, options `{"min_flow": true, "commitment": true}`): mean AUC
**unit_on 0.8473, margin 0.9207**. Weakest units: Kjela (0.573 / 0.714), Vinje G3
(0.799 / 0.888), Tokke G2/G4 unit_on (≈ 0.81).

**Loop results so far (2026-10-20).** Mean AUC over units (unit_on / margin):

| LP options (on top of min_flow + commitment) | unit_on | margin |
|---|---|---|
| none (exp 021's LP) | 0.8473 | 0.9207 |
| production_curves | 0.8745 | 0.9237 |
| flow_costs "yaml" | 0.8626 | 0.9251 |
| head_iterations 1 | 0.8452 | 0.9195 (dropped) |
| **production_curves + flow_costs "yaml"** | **0.8885** | **0.9289** |
| same + basis "sea" | 0.7264 | 0.7886 ("sea" still fails) |

| curves + yaml costs + integer_units (MILP) | 0.8470 | 0.9289 (dropped) |

The best combination is the feature group `relaxed_lp_shop`. **exp 024 (it replaces
relaxed_lp_uc): CV 0.9865, +0.0015 over exp 021, mostly 2020 (+0.006).** Public LB **0.99099** (021: 0.99067, VT1: 0.99040).

MILP trial (step 4) done: fast (median 0.7 s per case, all cases ≈ 6 min on 6 cores), but
its whole-number units-on ranks *worse* than the LP's fractional one (0.847 vs 0.889):
a hard 0/1 decision loses the "how close" information that AUC rewards. Dropped; the
`integer_units` option stays in relaxed_lp.py.

Peer-inspired hand features (exps 025–027: water budget, forward-looking margin, head
proxies): `water_budget` helps LightGBM *without* the LP (+0.0010, exp 025a, best LP-free
model at 0.9798) but adds nothing on top of the LP (the LP computes the same thing). The
other two add nothing on either base.

LightGBM tuning on exp 024 (exps 028a–e: num_leaves 31/255, min_child_samples 50/1000,
colsample 0.5): all −0.0006 … −0.0000. The original settings are near-optimal; gains must
come from the inputs. (028f, one model per unit with the full LP features: see the journal.)

Open lead: Kjela's LP margin is poor (0.69 vs 0.955 for its simple rule). Water Kjela
releases into the tiny Hyljelihyl can continue via r_Hyljelihyl → Venemo → Totak → Vinje,
a more valuable route than the one its end value assumes, so its downstream credit is
probably inflated (the same problem as the "net" rule in §3).

**Steps, in order:**
1. Each new option alone on top of the baseline: `production_curves`, `flow_costs: "yaml"`,
   `head_iterations: 1`. Keep an option if the mean margin or unit_on AUC rises clearly
   (≥ ~0.003) without Tokke/Vinje dropping.
2. Combine the winners. Then re-test `basis: "sea"` with them: it failed with the
   simple LP, but may behave differently once efficiency and spill are realistic.
3. For the best combination, add a variant group (copy `src/features/relaxed_lp_uc.py`,
   new file name and prefix, pass the options) and run a full experiment against exp 021
   (copy its config, swap `relaxed_lp_uc` for the new group). Judge by the §5b rule; for
   borderline results run a second seed (as exps 003 and 023 did).
4. Time-boxed MILP trial (BRIEFING §4.5 Level 3): the LP's fractional units-on is its
   weakest output. Solve ~50 cases with real binaries (`scipy.optimize.milp`, HiGHS, short
   time limit) and measure seconds per case and the unit_on AUC. Only pursue it if all
   3647 cases fit in a few hours on 4 cores.
5. Submit a new best to Kaggle only after a clear local gain (≤ 2 submissions per day) and
   record the public score in the journal.

**Pitfalls when changing the LP:**
- Feature caches are keyed on the group's own file and `src/data.py` only. The variant
  groups (`relaxed_lp_minflow`, `relaxed_lp_uc`, `tokke_vinje`, `cascade`,
  `storage_upstream`) import `relaxed_lp.py`, so **after changing `relaxed_lp.py`, delete
  their caches** (`rm features/relaxed_lp_uc__*` etc.) or they keep stale features.
- Don't edit `relaxed_lp.py` while an experiment is building an LP cache: the worker
  processes read the file from disk.
- `solve_all` uses all cores but one. Close the laptop lid only with the charger in, or
  runs stall.

**After the loop (by about Nov 1):** freeze the model and choose the two final submissions
(§5b), check them once on the locked 2018 holdout, then write `Short_notebook_1/2.ipynb`
(all code inside, from the raw data, ≤ 12 h), the model-interpretation section (SHAP /
feature importance; −3 points if missing) and the report. Deadline 2026-11-08 22:00.
