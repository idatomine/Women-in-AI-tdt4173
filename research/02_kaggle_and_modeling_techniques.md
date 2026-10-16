# 02 – Kaggle and modeling techniques (web research sweep)

Compiled 2026-10-15. Covers techniques for TDT4173 / SINTEF hydropower unit commitment
(14 units × 168 h, micro-averaged ROC-AUC). Every source below was found and opened
(or seen in search results) during this sweep. Where I could only see an abstract or a
search snippet, I say so. Numbers attributed to other student teams are **their own
reported numbers** from public repos. I did not verify them.

> **Academic integrity note.** Two public GitHub repos from *other teams in this same
> competition* (autumn 2026) were found (section 1). Use them as idea sources at most, and
> cite them if an idea is used. Do not copy their code. Check the course rules on using
> other teams' public material.

---

## 1. This competition and previous TDT4173 competitions

### 1.1 henrikremmen/modernemaskinl-ring (another team, same SINTEF task, public)
https://github.com/henrikremmen/modernemaskinl-ring (files `RESEARCH_*.md`, last commit 2026-10-05)
- **Their score history (self-reported public LB, micro-AUC):** raw 20 features with
  logistic/LightGBM ≈ 0.82. Engineered features: logistic 0.974, LightGBM 0.981. Per-unit
  XGBoost 0.988. "Physics" features (water-value margins, cumulative/remaining inflow,
  head proxies) took XGB/LGBM to ≈ 0.990, and a 50/50 blend reached 0.99005. An LP
  relaxation of the whole Tokke–Vinje system was used as features (notebook 19), and its
  blend was reported at **0.99571 public**. The leader was about 0.9956 around late
  September. **Conclusion: the big jumps came from feature representation and domain
  physics, not from model choice.**
- **What did not help (for them):** multitask TCN 0.927, multi-output ExtraTrees 0.945,
  EBM 0.79, greedy OOF blend and logistic stacking (both below the best single XGB),
  per-unit logistic calibration (it lowered AUC), and plant-state multiclass LightGBM
  (0.989, not better than the per-unit models). A temporal smoothing ensemble gave only
  +0.00001.
- **Error analysis:**
  - Hardest units: Haukeli G1, Lio G1, Vinje G2/G3.
  - A simple rule, "price > terminal water value of the source reservoir", already gets
    AUC ≈ 0.98–0.99 for Byrte and Vesle Kjela but only ≈ 0.76–0.79 for Haukeli and Vinje.
    The errors sit in hydraulically coupled units.
  - Lio is driven by a **minimum-flow constraint** in the river downstream, which acts as a
    dual/shadow price rather than a price signal.
  - Haukeli is small (4.9 MW) and has start costs, so start/stop economics matter.
  - An "oracle" that knows each case/unit's true weekly ON fraction gives AUC ≈ 0.997.
    **Errors in whole-week operating level dominate**, followed by timing within the week.
  - The first hours of the horizon (t0–t3) score worst.
- **Validation lesson:** use expanding-window, year-based folds (2020, 2021, 2022 as outer
  years) and purge cases whose 168-h horizon crosses the fold boundary. Don't use the outer
  year for early stopping; use an inner tail of the training period instead. Regime shift
  is large: mean price was about 9 EUR/MWh in 2020 and 211 in 2022, and 2023/24 fall in
  between, so 2022 is an outlier year.
- **Overlap pitfall:** adjacent cases disagree on **about 12% of the true states for the
  same calendar hour**, because the initial state, the terminal value and the horizon all
  change. Averaging predictions across overlapping cases is risky.

### 1.2 BilalS04/TDT4173-Group-71 (another team, same task, public)
https://github.com/BilalS04/TDT4173-Group-71 (notebook `xgboost_two_stage.ipynb`)
- Per-unit XGBoost in long format (case × hour rows) on 262 engineered "scheduling"
  features. Their self-reported public scores: 0.99053 for a single model and 0.99107 for a
  50/50 blend of baseline and deeper XGBoost.
- **Stage 2:** a small depth-3, 120-tree XGBoost per unit. Its inputs are stage-1
  probabilities of all 14 units, the unit's own predictions at ±1/3/6/12/24 h, rolling
  means, and the weekly mean/std/min/max of its predictions. It is trained only on
  out-of-fold stage-1 predictions and blended in with a weight chosen on 2021, then
  confirmed on 2022.
- Validation: training cases must **end** before the validation year starts
  (start + 168 h ≤ Jan 1). They also drop selection-year cases that overlap the
  confirmation year.
- Also confirms the metric: the Kaggle script is `roc_auc_score(..., average="micro")`
  over the flattened label matrix.

### 1.3 Previous TDT4173 Kaggle competitions (other partners)
- **2023: Aneo solar PV forecasting (MAE).**
  - power-predictor, https://github.com/SverreNystad/power-predictor: the team got an A
    and beat all baseline bots. Their lessons were that "data is the most important part"
    and that simpler models are better; some models ran for more than 24 h. A bug in how
    they built the test data inflated their apparent results until they fixed it. They
    tried CatBoost, LightGBM, XGBoost, AutoGluon, LSTM, Prophet and stacking.
  - https://github.com/sleipner01/tdt4173-machine-learning is the same 2023 task (private
    Kaggle competition).
- **2024: AIS vessel trajectory forecasting.**
  - https://github.com/arikraut/ais-vessel-trajectory-forecasting (grade A): one XGBoost
    per horizon bucket, a strict time cutoff (anchors before the cutoff, targets after) and
    Optuna per horizon.
  - https://github.com/markus0800/AIS-Vessel-prediction- (group 8, fall 2024): random
    forest + XGBoost.
- **Common pattern across years:** gradient boosting with heavy feature engineering, a
  strict time-based split and a small blend wins. Neural sequence models rarely ended up in
  the final submission.

---

## 2. Domain literature: ML that imitates a unit-commitment solver

### 2.1 Barjaktarevic, Kong, Yang, Naversen (2026). "An Evaluation of Neural Network Architectures for Predicting Hydro Unit Commitment Decisions", IEEE EEM 2026
https://www.sintef.no/en/publications/publication/019f93eb7024-9db6a5de-cdc7-4ad4-9d28-d051824cd633/ (abstract page only)
- Written by SINTEF, very likely the people behind our dataset. They frame the problem as
  **multilabel UC prediction** over **24 h and 168 h** horizons for a real Norwegian
  watercourse.
- They compared simple, convolutional, recurrent and hybrid NNs. All reached F1 > 97%, and
  **convolutional architectures were the most consistent** across horizons.
- Caveat: they optimize F1, not micro-AUC, and they may have far more (synthetic) data
  than we do; see 2.2. Use this as the motivation for a 1D-CNN baseline, not as proof that
  CNNs beat GBDT here.

### 2.2 Reite, Kong, Naversen (2026). "Synthetic Data Generation for ML-Accelerated Hydropower Scheduling", IEEE EEM 2026
https://www.sintef.no/en/publications/publication/019f93c0595c-4f9905ae-3c25-4b3a-8f8c-c47ac61f6eb8/ (abstract)
- Same system (Tokke–Vinje) and **about 3,650 historical cases**, which matches our
  2922 + 725 = 3647. This is almost certainly the data source.
- Their synthetic cases draw initial reservoir states from a KDE and price and inflow
  profiles from history, then compute water values from a normalized availability
  coefficient. This tells us how SINTEF thinks about the drivers: **initial fill relative
  to season, price profile and water value**.
- We cannot run SHOP to label synthetic cases. Still, the idea is useful for augmentation
  ideas, and for spotting which inputs are "free" and which are derived.

### 2.3 SINTEF news: "How AI could squeeze more value from Norwegian hydropower" (InterOpt, 2026-06-26)
https://www.sintef.no/en/latest-news/2026/how-ai-could-squeeze-more-value-from-norwegian-hydropower/
- The research agenda: run a fast, **relaxed** UC model first, then let ML predict the
  exact on/off decisions.
- For us, this suggests **features from a cheap LP/heuristic relaxation**, such as a
  price-vs-water-value margin or a simple dispatch LP. That is exactly what gave the other
  team their largest late gain (section 1.1).

### 2.4 Xavier, Qiu & Ahmed (2021). "Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems", INFORMS J. Computing 33(2)
https://arxiv.org/abs/1902.01697
- They learn from previously solved, slowly varying daily instances to predict
  warm-starts and the subspaces where the optimum is likely to lie. They report 4.3× to
  10.2× speed-ups.
- Takeaway: solver decisions are highly **repeatable given similar inputs**.
  Nearest-neighbour or similar-week features (for example "label of the most similar
  training case") are a cheap, strong idea. They must be computed out-of-fold.

### 2.5 Pineda & Morales (2021). "Is learning for the unit commitment problem a low-hanging fruit?"
https://arxiv.org/abs/2106.11687
- A *naive* algorithm that reuses past UC solutions already gets optimal or near-optimal
  results. Complex ML has to beat simple baselines.
- For us: build a "copy the most similar past week" baseline (kNN on standardized inputs)
  and a "price > water value" rule baseline before anything fancy.

### 2.6 Pourahmadi & Kazempour (2023). "Unit Commitment Predictor With a Performance Guarantee: A SVM Classifier"
https://arxiv.org/abs/2310.08601
- Per-unit classifiers of on/off status; a kernel SVM was best on the IEEE test systems.
- Confirms that the "one classifier per unit" decomposition is standard in UC learning.

### 2.7 Related: "Neural Two-Stage Stochastic Optimization for Solving UC" (arXiv 2507.09503)
https://arxiv.org/abs/2507.09503 (seen in search snippet)
- Units that are constant across all training samples carry no learnable signal. Check
  for units that are (almost) always ON or OFF. For micro-AUC, those units still need
  correctly scaled probabilities, but they need no fancy model.

---

## 3. Similar Kaggle competitions and lessons

### 3.1 ASHRAE Great Energy Predictor III, lessons-learned paper
https://arxiv.org/abs/2202.02898 ("Gradient boosting machines and careful pre-processing work best")
and the overview paper https://arxiv.org/abs/2007.06933
- Winners were ensembles of GBDT, mostly LightGBM. Participants rated **preprocessing and
  feature extraction** as the most important part.
- Energy tabular-plus-time tasks with few thousand to millions of rows follow the same
  pattern: GBDT plus domain features first.

### 3.2 M5 Accuracy competition (Makridakis et al., IJF 2022)
https://ideas.repec.org/a/eee/intfor/v38y2022i4p1346-1364.html
- The winner was an equal-weighted ensemble of LightGBM models, and almost all of the top
  50 used LightGBM. **Global models** (one model across many series, with the series ID as
  a feature) work well.
- This supports the long format: rows = (case, hour[, unit]) with hour-of-horizon and unit
  as features.

### 3.3 Mechanisms of Action (MoA), Kaggle 2020, multi-label (206 binary targets)
https://www.kaggle.com/c/lish-moa (the write-ups could not be fetched; Kaggle pages render
via JS). Secondary notes: https://scrapbox.io/regonn-curry/105_MoA%E3%82%B3%E3%83%B3%E3%83%9A%E3%82%92%E7%B5%82%E3%81%88%E3%81%A6
- The top solutions were multi-output MLP and TabNet ensembles with label smoothing. That
  was for **log-loss** on about 24k samples with many sparse targets, and the shared
  hidden layer helped rare labels.
- Their CV respected the grouping (drug groups → GroupKFold). The analogue here is
  grouping by time block.
- Label smoothing helps log-loss. For AUC it is a monotone squeeze, so its only effect is
  through regularizing training. Expect little gain.

### 3.4 Google Brain Ventilator Pressure Prediction (Kaggle 2021, 1st place)
https://towardsdatascience.com/winning-the-kaggle-google-brain-ventilator-pressure-prediction-2d4c90d831ec
and code https://github.com/GillesVandewiele/google-brain-ventilator
- Fixed-length sequence (80 steps) → per-step target, solved with LSTM and
  CNN-Transformer models plus **physics-based post-processing** (PID controller matching).
- That competition had about 75k sequences. Sequence nets win with that much data **and**
  when the target is a deterministic function of a controller. Our case is analogous (a
  deterministic solver) but has only about 2.9k sequences. Domain post-processing or
  features from a relaxed optimizer are the transferable trick.

### 3.5 Child Mind Institute – Detect Sleep States (Kaggle 2023)
Repo of a top-2% team: https://github.com/TeamEpochGithub/cmi-detect-sleep-states
- Per-time-step labeling of long sequences, typically done with 1D-CNN/GRU/U-Net encoders.
- Shows the "1D U-Net/GRU over the sequence" pattern for per-hour outputs. That pattern
  pays off with many long sequences. With 2.9k × 168 it is plausible only as an ensemble
  member.

### 3.6 Enefit – Predict Energy Behavior of Prosumers (Kaggle 2024)
Leaderboard mirror: https://clist.by/standings/enefit-predict-energy-behavior-of-prosumers-energy-time-series-analysis-tabular-mae-47550689/
- I could not retrieve the winning write-up. It is generally reported as GBDT-dominated,
  but I have not verified that, so treat it as unconfirmed.

---

## 4. Model families for our output shape (14 × 168 binary)

### 4.1 Tabular evidence: trees vs. NN
Grinsztajn, Oyallon, Varoquaux (2022). "Why do tree-based models still outperform deep learning on tabular data?" https://arxiv.org/abs/2207.08815
- On medium-sized tabular data (about 10k rows), XGBoost and random forests still beat
  NNs. NNs struggle with uninformative features and irregular functions.
- Our case-level dataset has about 2.9k rows (cases). In long format it becomes about
  490k rows (case × hour) per unit, or about 6.9M with units stacked.

### 4.2 TabPFN v2 (Hollmann et al., Nature 2025) and TabPFN-2.5
https://pubmed.ncbi.nlm.nih.gov/39780007/ and https://huggingface.co/Prior-Labs/TabPFN-v2
- TabPFN v2 is strong on datasets of 10k rows or fewer; v2.5 handles about 50k rows.
- **Not a fit for the long format** (hundreds of thousands of rows). It could work on a
  case-level sub-problem, such as predicting each unit's **weekly ON fraction** from about
  2.9k cases. That is the "level" problem the oracle analysis in 1.1 says dominates. It is
  worth one experiment as a level feature or prior.

### 4.3 Time-series foundation models (Chronos, TimesFM, Moirai, MOMENT)
- MOMENT paper: https://arxiv.org/abs/2402.03885. In its comparison table, Chronos, Moirai
  and Lag-Llama are listed for forecasting only, and only MOMENT for classification.
- EPF benchmark of TSFMs: https://arxiv.org/abs/2506.08113. Chronos-Bolt and Time-MoE
  were best among TSFMs, but no TSFM statistically beat a simple MSTL baseline on
  day-ahead prices.
- Chronos-Bolt has no covariate support; Chronos-2 adds it (ECB slide deck:
  https://www.ecb.europa.eu/press/conferences/shared/pdf/20260323_forecasting/Guerron_presentation.pdf).
- **Verdict: not relevant as the main model.** Our task is supervised classification of a
  solver decision, and all exogenous inputs (168-h price and inflow) are already given, so
  nothing needs to be forecast. At most, frozen MOMENT embeddings of the price curve could
  be used as features, which is low priority.

### 4.4 Hugging Face
- I found no hydropower or UC dataset or model on HF. The closest is the
  microsoft/GridSFM_US_power_grid dataset (https://huggingface.co/datasets/microsoft/GridSFM_US_power_grid),
  a US grid with UC decommit counts, which is not useful here.
- A hydro UC case-study dataset (not on HF) exists: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5996229/.
  It has a different system and is not usable as training data.

### 4.5 Output decomposition options (synthesis; evidence from sections 1–3)
| Option | Rows | Pros | Cons | Evidence |
|---|---|---|---|---|
| **A. One GBDT per unit, long format** (row = case×hour, hour-of-horizon + case features + hour-specific price/inflow/margins) | ~490k/unit | Proven: ~0.988–0.991 for two peer teams; per-unit physics features | 14 models; weaker cross-unit coupling | Peer repos 1.1, 1.2 |
| B. One pooled GBDT, unit as categorical | ~6.9M | Shares data; probabilities on common scale (good for micro-AUC) | Unit-specific features become messy; pooled CatBoost 0.983 < per-unit 0.985 (peer, sampled 35%) | 1.1 |
| C. One model per (unit, hour) = 2352 models | ~2.9k each | — | Throws away sharing; too many tiny models | none — avoid |
| D. Multi-output NN over 168 h (MLP / 1D-CNN / U-Net / TCN / Transformer) | 2.9k sequences | Natural joint/temporal structure; SINTEF found CNN best (F1) | Small data; peer TCN only 0.927 | 2.1, 1.1 |
| E. Plant-state multiclass (Tokke 16 states, Vinje 8) | ~490k/plant | Captures sibling coupling | Rare states; peer 0.989 not better | 1.1 |
| F. Stage-2 stacker on OOF probs (sibling units, ±k h, weekly stats) | as A | Cheap; adds temporal + cross-unit context | Easy to leak; peer results mixed | 1.1, 1.2 |

**With about 3,000 cases, option A wins, and its gain comes from features.** Use D as an
ensemble-diversity member only if it gets above about 0.97 alone. In the peer blend, a
5%-weight TCN still helped OOF AUC.

### 4.6 Micro-AUC specific tricks
Metric definition: sklearn `roc_auc_score(average="micro")` treats every element of the
label matrix as one binary decision
(https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html).
The following points are my own reasoning from that definition, not from a cited source:
- **Cross-unit ranking matters.** All 2352 × N_test probabilities are ranked jointly, so a
  unit with a high base rate must get higher probabilities than a mostly-OFF unit. Train
  each per-unit model with **log-loss and no per-unit `scale_pos_weight` or class
  weights**. Weighting distorts the unit's probability scale relative to other units and
  can lower micro-AUC even if per-unit AUC rises.
- Monotone global transforms (temperature scaling, label smoothing, global Platt) **do not
  change** AUC. Per-unit calibration *can* change micro-AUC, but it helped nobody in the
  peer tests (it lowered AUC from 0.986 to 0.984).
- When blending models, blend **probabilities or logits with global weights** (not per
  unit) unless validated. Never rank-normalize per unit.
- Evaluate locally with the exact Kaggle metric on the pooled matrix. Also log per-unit
  AUC and **within-week AUC versus weekly-level error** to tell level errors from timing
  errors.
- Temporal smoothing (rolling mean of logits along the horizon) gives at best tiny gains.
  Don't average predictions across overlapping cases (12% label disagreement, 1.1).

---

## 5. Validation for overlapping rolling-horizon cases

### 5.1 Purged cross-validation (López de Prado), Wikipedia summary
https://en.wikipedia.org/wiki/Purged_cross-validation
- Purging drops training samples whose label span overlaps the test label span. The
  embargo also drops samples just after the test fold.
- Here a case's label span is [start, start + 167 h], so neighbouring cases share up to
  144 hours of the same calendar period. Inputs are nearly identical too: same prices,
  shifted by one day.

### 5.2 Implementations
- sklearn `TimeSeriesSplit(gap=...)`: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html.
  It is forward-only and `gap` removes samples just before the test fold. Set
  `gap ≥ 7` cases (days).
- skfolio `CombinatorialPurgedCV` (`purged_size`, `embargo_size`): https://skfolio.org/generated/skfolio.model_selection.CombinatorialPurgedCV.html
- temporalcv `PurgedKFold` (`purge_gap`, `embargo_pct`; it suggests an embargo of at
  least the prediction horizon): https://temporalcv.readthedocs.io/en/latest/api/cv_financial.html

### 5.3 Recommended scheme for us (synthesis)
- **Primary:** expanding-window year folds. Train on cases with `start + 168 h ≤ Jan 1 of
  the validation year`, then validate on years 2019, 2020, 2021 and 2022, which gives 4
  folds. Report the mean and the worst fold.
- **Secondary, for more training data per fold:** leave-one-year-out (GroupKFold by year),
  with **purge plus a 7-day embargo on both sides** of the held-out year. This is
  acceptable because the targets are solver decisions given inputs rather than
  autoregressive forecasts. It is still optimistic if regimes drift, which they do
  strongly (2022).
- Early stopping must use an inner time-tail of the training data, **not** the validation
  year. Otherwise the CV is optimistically biased (peer team flagged this, 1.1).
- **Never random KFold over cases:** neighbouring cases are near-duplicates.
- Expect fold variance (for example 2020 ≈ 0.92–0.95 vs 2021 ≈ 0.98 for the same model in
  peer tests). Small differences (< 0.001) are noise. The public LB is a subset of
  2023–24, so don't tune to it.
- Leakage pitfalls:
  - Rolling or normalization statistics computed over the full dataset.
  - Target encodings or kNN-label features not computed out-of-fold.
  - Features from *observed* volumes after the case start, if any exist in the data.
  - Run No or calculation-time columns.
  - Using next-case labels as "lags": cases overlap, and at test time the labels are
    unknown anyway.

---

## Implications for our project (prioritized)

1. **Set up the harness first.**
   - Use the exact Kaggle `score` function (micro-AUC on the flattened matrix).
   - Use expanding year folds with a horizon purge (5.3). Early-stop on an inner tail.
   - Log per-unit AUC, within-week AUC and weekly-level MAE.
2. **Baselines, no ML.**
   - Per-unit base rate (expect micro-AUC around 0.5–0.6; the peer climatology baseline
     scored 0.54).
   - The rule `price_t − terminal water value of the unit's source reservoir`, converted
     to a score. For each unit, map the plant to its feeding reservoir(s) from the
     topology YAML.
   - kNN "most similar past week" label copy (Pineda & Morales).
3. **Main model: per-unit LightGBM/XGBoost in long format** (row = case × hour; about
   490k rows per unit). Features:
   - Hour-of-horizon, hour-of-day, weekday, season.
   - Price at t, price lags and leads (±1, 3, 6, 12, 24 h), and price rank within the day
     and within the week.
   - Price minus the water value of the source reservoir (local margin), normalized
     margin, and the number of hours in the week that are better than t.
   - Initial fill as a fraction of capacity, plus cumulative and remaining inflow for the
     week relative to capacity and free storage.
   - Seasonal minimum-flow and minimum-volume constraints active at t. This is crucial for
     Lio.
   - Upstream and downstream reservoir states for coupled plants (Tokke, Vinje, Haukeli).

   Train with log-loss and no class weights. Expect about 0.985–0.99 public.
4. **Add "relaxed optimizer" features.** This gave the biggest gain for peers.
   - Start with a per-plant two-state dynamic program over the 168-h price path with a
     start-cost penalty, giving the ON-vs-OFF value difference per hour.
   - Then try a small LP of the watercourse (scipy `linprog` / HiGHS) per case. Use flows,
     reduced costs and duals per station-hour as features.
   - This follows the SINTEF/InterOpt "relax then learn" approach (2.3).
5. **Target the weekly level explicitly.** Build a case-level model of each unit's weekly
   ON fraction (GBDT or TabPFN on about 2.9k cases). Use it out-of-fold as a feature or
   logit offset in the hourly model. The oracle analysis suggests level errors dominate.
6. **Ensemble.**
   - Blend XGB + LGBM (+ CatBoost) with 2–3 seeds each, using global weights on
     probabilities.
   - Optionally add a stage-2 stacker on OOF probabilities: sibling-unit probabilities,
     own ±k h and weekly stats, with shallow trees and a weight validated forward in time.
7. **Optional NN member.** Try a 1D-CNN/U-Net over 168 h with per-hour inputs and static
   context, a 14-channel sigmoid output and BCE loss (SINTEF found CNN most consistent).
   Include it only if it adds OOF AUC in the blend. Don't spend the main effort here.
8. **Skip or deprioritize:**
   - Time-series foundation models.
   - Per-unit class weights and per-unit calibration.
   - Smoothing across overlapping cases.
   - Large Transformers or GNNs.
   - Public-leaderboard-driven tuning.
9. **Reproducibility:** respect the course runtime limits (the peer notes mention a
   12-hour and 32 GB grading target for the short notebooks). Fix seeds and thread counts.
