# 01 – ML for Unit Commitment (UC) / Predicting Binary Decisions of Optimization Problems

Literature/web sweep for the TDT4173 SINTEF InterOpt Kaggle task (Tokke–Vinje, 14 units, 168 h, micro ROC-AUC).
Compiled 2026-10-15. Every source below was actually located online. Statements marked **[uncertain]** are my inference, not something a source states.

---

## TL;DR

* **The Kaggle task is almost certainly a version of a published SINTEF/NTNU dataset.** Barjaktarevic, Kong, Yang & Naversen (EEM 2026) describe the same system (17 reservoirs, 9 plants, 14 units), a daily rolling 168-h horizon with **3,647 cases** (2,922 + 725 = 3,647), and **2,352 label columns** (14 × 168). That is exactly our setup.
* **Their results:** every neural net they tried reached micro-F1 above 97.4% on 168 h. 1D CNNs (plain CNN, CDIL-CNN, TCN) were the most consistent, a plain MLP came close, and RNNs were weakest. They used a **random 80/20 split**, which is optimistic for overlapping rolling-horizon cases. Our split is temporal (2023–24).
* **How the labels were made:** each case was solved three ways in SHOP (All_MIP, Mixed_MIP, No_MIP). The label is the UC schedule with the best objective after a 1-second nonlinear simulation. So some labels come from a *relaxed LP plus a 50%-of-minimum-load threshold*, not from the exact MILP.
* **How water values were made:** in this dataset they are most likely *not* from a mid-term model. SINTEF used a heuristic: `WV = scarcity coefficient(volume vs. historical min/median/max for that date) × 168-h forward rolling mean price`. So **price ÷ water value** (the hourly price relative to the coming week's mean price, scaled by scarcity) is likely the single most informative signal. **[uncertain whether the historical cases use exactly this formula; the paper says the time-series computation is documented in that work]**
* **General UC-learning literature:** simple methods (k-NN over past instances, per-unit classifiers) are strong baselines (Pineda & Morales). Exploiting temporal structure (x_t = x_{t+1}, min up/down) helps (Xavier et al.). Distribution shift is the main known weakness.

---

## A. Directly relevant: SINTEF / NTNU / InterOpt

### 1. An Evaluation of Neural Network Architectures for Predicting Hydro Unit Commitment Decisions
*D. Barjaktarevic, J. Kong, Z. Yang, C. Ø. Naversen. 2026 22nd Int. Conf. European Energy Market (EEM), IEEE. DOI 10.1109/eem68581.2026.11589730*
Links: https://www.sintef.no/en/publications/publication/019f93eb7024-9db6a5de-cdc7-4ad4-9d28-d051824cd633/ · NVA record (open AAM PDF, CC-BY): https://nva.sikt.no/registration/019f93eb7024-9db6a5de-cdc7-4ad4-9d28-d051824cd633
I read the full text. **This is the closest thing to our task: same watercourse, same horizon, same label shape.**
- **Data.** Tokke–Vinje with NO2 prices from ENTSO-E and hydrology from NVE. The 168-h horizon has 3,647 cases, 7,930 feature columns and 2,352 label columns. Solved with SHOP v17.15.1 + CPLEX 20.1.
- **Features they used:** n prices, n×r reservoir inflows, n×l catchment inflows, r initial volumes, r *end* water values, plus the **No_MIP (relaxed LP) solution's production and UC outputs** (n×k each). The relaxed-LP outputs are a very strong feature, and we probably do *not* have them in Kaggle.
- **Label construction.** Three UC-mode settings were run:
  - All_MIP: 5 MILP iterations.
  - No_MIP: 5 LP iterations; a unit counts as "on" if its production ≥ 50% of its minimum limit.
  - Mixed_MIP: 3 LP iterations, then 2 MILP iterations.
  
  The label is the schedule with the highest objective in the nonlinear simulation. Ties are broken in the order All → Mixed → No_MIP. **So labels are a mix of exact-MILP and heuristic schedules.** Expect some label noise and over-commitment patterns that look like the No_MIP rule.
- **Results (168 h, micro-averaged).** No_MIP baseline: F1 0.9735 (precision 0.953, recall 0.995). The NNs reached F1 0.975–0.978: CDIL-CNN 0.9778, CNN 0.9774, TCN 0.9770, DLinear 0.9768, MLP 0.9762, LSTM 0.9760, RNN 0.9760, CNN-LSTM 0.9748.
  - Differences between architectures are small. CNNs are the most consistent.
  - DLinear does well at 24 h but degrades at 168 h.
  - Models are small (MLP ~60k parameters, CNN ~260–390k) and train in about 2–3 minutes. They tuned with Optuna.
- **Pitfalls to note:**
  - They used a *random* 80/20 split. Neighbouring rolling cases overlap by 6 of 7 days, so that split leaks information. Our temporal split will be harder.
  - Better classification metrics did *not* always give a better objective value.
  - They report no ROC-AUC, only threshold metrics.

### 2. Synthetic Data Generation for Machine Learning-Accelerated Hydropower Scheduling: A Scalable Approach to Overcome Historical Data Limitations
*T. Å. Reite, J. Kong, C. Ø. Naversen. EEM 2026, IEEE. DOI 10.1109/eem68581.2026.11589808*
Links: https://www.sintef.no/en/publications/publication/019f93c0595c-4f9905ae-3c25-4b3a-8f8c-c47ac61f6eb8/ · NVA (open AAM): https://nva.sikt.no/registration/019f93c0595c-4f9905ae-3c25-4b3a-8f8c-c47ac61f6eb8
I read the full text. It documents how our input time series were built.
- **Inflows.** Reservoir inflows are *scaled copies of measuring-station series*: `I_res = (mean_res / mean_station) · I_station`. Several reservoirs share one station, so their **hourly shapes are identical** and only the magnitude differs. Expect strongly collinear inflow features; aggregating them, or keeping one per station, loses little.
- **Water values.** `W_{k,t} = F_{k,t} · mean(P_{t+1..t+168})`. The scarcity coefficient F is:
  - 2 at the historical minimum volume for that calendar date,
  - 1 at the median,
  - 0 at the maximum (if W = 0 it is set to 0.01).
  
  Reservoirs with no plant directly downstream copy values from comparable reservoirs. The water values of Songavatn and Bitdalsvatn are averaged because they share the Songa plant. **Consequence: water values encode both (a) the coming week's mean price and (b) reservoir fullness relative to season.** The decision "run when price > water value" is therefore roughly "run when hourly price > F × weekly mean price".
- **Prices.** NO2 prices show a strong regime shift: peaks up to roughly 800 EUR/MWh in 2022 (their Fig. 3). In their synthetic scenarios they restrict price sampling to the most recent 11 years, acknowledging the structural shift.
- **Data volume.** They argue that roughly 3,650 historical cases are too few and too homogeneous. Their synthetic dataset (KDE over 17-D reservoir states, bandwidth 0.05) covers about 100k states. A larger "Tokke-Vinje Open Dataset v3_2026_3" may exist. **[uncertain whether it includes SHOP UC labels, and whether competition rules allow external data]**

### 3. KSP InterOpt Project Open Dataset (SHOP_17_15_1_Tokke_3_history) and Tokke-Vinje Watercourse Open Dataset
*J. Kong, D. Barjaktarevic, C. Ø. Naversen (2026), SINTEF datasets*
Links: https://nva.sikt.no/registration/01a0f39c8505-5ceb7fe1-2bb3-4cfb-a87c-e4bcbfd36d4b · https://nva.sikt.no/registration/01a0f3a602bf-ae0fae77-3e5a-437c-9c39-239aa6c365df · data.sintef.no feature pages cited in paper 1 (I could not render them).
- The description reads: "datasets for developing machine learning models that can support or accelerate operational hydropower decision making". The watercourse dataset includes 10 years of hourly inputs (2015–2024) and several versions with generation logs.
- **Integrity warning:** if this public dataset contains labels for 2023–2024, using them for test predictions would be cheating. **Do not use test-period labels.** Check the course/Kaggle rules before using *any* external data. That includes the topology and the No_MIP outputs, which could otherwise be a legitimate and very strong feature *if* allowed and reproducible.

### 4. SINTEF InterOpt project page and news article
Links: https://www.sintef.no/en/projects/2024/interopt/ · https://www.sintef.no/en/latest-news/2026/how-ai-could-squeeze-more-value-from-norwegian-hydropower/
- InterOpt is a Research Council of Norway KSP project (2024–2028) led by Jiehong Kong. Partners include NTNU, UPM (Madrid), UQAC (Canada) and CEPEL (Brazil), with industry partners ANEO, Hydro Energi and Å Energi.
- The stated concept: run a fast *relaxed* model first, then use ML to predict the decisions the exact MIP would produce. The rationale is that the same UC problem is solved every day with only small input changes.

### 5. Why Optimal Solutions Differ: Mathematical Explanation of NL, MILP, and LP Hydropower Production in SHOP
*J. Kong, C. Ø. Naversen (2026), SINTEF report (open PDF, CC BY-NC)*
Link: https://www.sintef.no/en/publications/publication/01a0f3805ef8-2645f514-8c93-4994-af5b-3f76bc65c52f/
- Background on why the LP relaxation and the MILP give different on/off decisions (minimum production limits, piecewise-linear production curves). Useful for understanding *when* units are on. I did not read the full report.

### 6. An intelligent decision-making process for hydro scheduling (NTNU MSc thesis)
*P. Babayev (supervisor Z. Yang), NTNU 2022*
Link: https://hdl.handle.net/11250/3027677
- An earlier NTNU/SINTEF attempt to predict SHOP *settings/commands* (not UC) from reservoir volume, inflow and prices, using data from 6 producers.
- Used TPOT AutoML, imbalanced-data resampling and feature engineering. The authors describe the performance as "moderate".
- Lesson: AutoML or GBDT pipelines on hand-crafted aggregate features are a reasonable baseline. Class imbalance matters.

### 7. Machine Learning for Hydropower Scheduling: State of the Art and Future Research Directions
*C. Bordin, H. I. Skjelbred, J. Kong, Z. Yang. Procedia Computer Science 176 (2020) 1659–1668. DOI 10.1016/j.procs.2020.09.190*
Link: https://hdl.handle.net/10037/19622 (also on core.ac.uk)
- A SINTEF/NTNU review that sets out the agenda (ML integrated into short-term scheduling) leading to InterOpt. Context only; it contains no UC-prediction results.

### 8. The Value of Considering Uncertainty in Short-Term Hydropower Scheduling – Combining Machine Learning and Stochastic Programming (NTNU MSc thesis)
*K. K. Fleten, M. F. Bjørdal (supervisors H. Andersson, G. Klæboe, O. E. Gundersen), NTNU 2020*
Link: https://hdl.handle.net/11250/2776928
- Combines ML with SHOP-like short-term scheduling. I found only the metadata (the full text was embargoed until 2023). Low priority.

---

## B. General ML-for-UC / learning MILP binaries

### 9. Is learning for the unit commitment problem a low-hanging fruit?
*S. Pineda, J. M. Morales. Electric Power Systems Research 207 (2022) 107851*
Link: https://arxiv.org/abs/2106.11687
- **Method:** a naive k-NN (K = 50, L2 distance on the input load profile). Fix the binary decisions from each neighbour, solve the resulting LPs, and keep the best.
- **Results:** gives near-optimal UC (average gap ≤ 0.13%) with 40–200× speedups on 9 European-scale systems.
- **Message:** sophisticated models must *statistically beat k-NN*. **For us, k-NN over historical cases (soft-voting on neighbours' labels) is a mandatory baseline**, and it is also a good ensemble member.
- Caveat: they used leave-one-out on synthetic, i.i.d. instances with no temporal split.

### 10. Learning to Solve Large-Scale Security-Constrained Unit Commitment Problems
*Á. S. Xavier, F. Qiu, S. Ahmed. INFORMS J. Computing (2020/21). DOI 10.1287/ijoc.2020.0976*
Link: https://arxiv.org/abs/1902.01697
- **k-NN warm start:** average x_gt over the k nearest past solutions. Set it to 1 if the average is > p, to 0 if it is < 1 − p, and leave it free otherwise. p = 0.9 worked best. This averaged neighbour vote is effectively a **probability estimate, which is directly usable for AUC.**
- **"Affine subspace" hints:** linear SVMs predict x_gt = 0, x_gt = 1, or **x_gt = x_g,t+1** (no switch between consecutive hours). Strong temporal persistence is the structural prior to exploit.
- **Results:** 4.3× faster with optimality guarantees, 10.2× without. Somewhat robust to moderate distribution shift.

### 11. Accelerating transmission-constrained unit commitment via a data-driven learning framework
*Z. Lin et al. Frontiers in Energy Research 10 (2023) 1012781*
Link: https://www.frontiersin.org/journals/energy-research/articles/10.3389/fenrg.2022.1012781/full
- **Setup:** multi-label, *one model per unit* mapping inputs to that unit's 24-h on/off vector. Compares DNN, decision tree, random forest and 1-NN on **real 2018–2020 data with a temporal split** (train until June 2020, test Jul–Dec 2020).
- **Results:** with a realistic temporal split, precision/recall for the "on" class was only about 0.74–0.84, far below the ~97% seen with random splits. DNN, DT and RF performed similarly; k-NN had the weakest classification scores but gave the largest speedup because its schedules were always feasible.
- **Lesson:** expect a large gap between random-split CV and a true temporal split.

### 12. Unit Commitment Predictor With a Performance Guarantee: A Support Vector Machine Classifier
*F. Pourahmadi, J. Kazempour (2023)*
Link: https://arxiv.org/abs/2310.08601
- One classifier per unit per hour. A regularized kernel (RBF) SVM beat linear SVM. On a small system, k-NN and kernel SVM were about equal.
- Regularization tuned by CV gives distributionally robust, out-of-sample guarantees. **Regularize strongly when data are limited** (we have about 2,900 cases).

### 13. Spatio-Temporal Deep Learning-Assisted Reduced Security-Constrained Unit Commitment
*A. V. Ramesh, X. Li. IEEE Trans. Power Systems 39(2) (2024). arXiv 2306.01570*
Link: https://arxiv.org/abs/2306.01570 (code and data on Figshare)
- GNN (spatial) + LSTM (temporal) predicts the commitment schedule and reduces the SCUC.
- Relevant if we model the watercourse topology as a graph of reservoirs and plants. With only 14 units and fixed topology, a GNN is probably overkill compared with plant/unit embeddings. **[my judgement]**

### 14. Fast Solution of Unit Commitment Using Machine Learning Approaches
*S. Schmitt, I. Harjunkoski, M. Giuntoli, J. Poland, X. Feng. IEEE ENERGYCON 2022. DOI 10.1109/ENERGYCON53164.2022.9830191*
Link: (IEEE Xplore; abstract only)
- Random forest vs. GNN vs. a rule-based predictor for binary UC. The authors say each has pros and cons, and that generalization is the key issue. I could not access the full text.

### 15. A Multi-Stage Warm-Start Deep Learning Framework for Unit Commitment
*M. E. Za'ter, A. Van Boven, B.-M. Hodge, K. Baker (arXiv 2604.21891, Apr 2026)*
Link: https://arxiv.org/abs/2604.21891
- A transformer predicts a 72-h commitment schedule. Deterministic post-processing enforces **minimum up/down times**, then a confidence-based fix plus MILP warm start follows.
- Search snippets say it placed 4th in an EPRI UC competition on a system that includes hydro and pumped storage. **[uncertain: not stated on the abstract page]**

### 16. Solving Mixed Integer Programs Using Neural Networks (Neural Diving)
*V. Nair et al. (DeepMind), arXiv 2012.13349 (2020)*
Link: https://arxiv.org/abs/2012.13349
- A generic "learn the MILP binaries" method: a GNN over the variable–constraint graph predicts each binary's value. Only *confident* variables are fixed, and the solver handles the rest.
- Main takeaway for us: **calibrated per-variable probabilities are the natural output**, which is exactly what ROC-AUC rewards.

### 17. Use of Machine Learning Models to Warmstart Column Generation for Unit Commitment
*N. Sugishita, A. Grothey, K. McKinnon. INFORMS J. Computing 36(4) (2024). arXiv 2110.06872*
Link: https://arxiv.org/abs/2110.06872
- Predicts duals rather than binaries. Mainly confirms that small feed-forward NNs trained on past daily instances generalize well enough to be useful. Low direct relevance.

### 18. Supporting reviews and surveys
- Y. Yang, L. Wu, "Machine learning approaches to the unit commitment problem: Current trends, emerging challenges, and new strategies", *The Electricity Journal* 34(1), 2021. https://researchwith.stevens.edu/en/publications/machine-learning-approaches-to-the-unit-commitment-problem-curren/ (ML as a complement to MILP, not a replacement).
- Y. Villeneuve, S. Séguin, A. Chehri, "AI-based scheduling models, optimization and prediction for hydropower generation", GERAD G-2022-52 / *Energies* 16(8) 3335 (2023). https://gerad.ca/en/papers/G-2022-52 (survey; Séguin is at UQAC, an InterOpt partner).
- F. Mohammadi et al., "Machine Learning Assisted Stochastic Unit Commitment: A Feasibility Study", EasyChair preprint 4533 (2020). https://easychair.org/publications/preprint/NFd5 (ML unlikely to work as a stand-alone solver; useful as an accelerator).
- P. H. Guerra Núñez, MSc thesis, UTFSM Chile (2023): multi-agent DRL with **TCN** policies predicts on/off over a **168-h** horizon for a real hydrothermal system. https://cris.usm.cl/handle/123456789/2303 (another vote for temporal convolutions at a 168-h horizon).

### 19. Context for the price regime shift (pitfall)
- The NMBU MSc thesis (2023) on water values in NO2/NO5 during the 2021 crisis found most producers *raised* water values as gas prices rose. https://nmbu.brage.unit.no/nmbu-xmlui/handle/11250/3078976
- Hafslund's 2022 annual report gives NO2 average prices of about 75 øre/kWh (2021) and about 213 øre/kWh (2022). https://aarsrapport2022.hafslund.no/en/hafslund-eco-vannkraft/
- Takeaway: absolute price levels in 2021–22 are far outside 2015–2020. The 2023–24 test period has lower but still elevated and volatile prices. **Features based on absolute EUR/MWh will shift; relative features (price/WV, price minus weekly mean, rank within week) should transfer better.**

---

## C. Patterns across the literature

| Theme | Evidence | Relevance |
|---|---|---|
| Simple baselines are strong | k-NN (Pineda & Morales; Xavier; Lin), MLP ≈ CNN (Barjaktarevic) | Build k-NN + GBDT + MLP/CNN quickly and compare honestly |
| 1D temporal convolutions work best for 24–168 h | Barjaktarevic (CDIL-CNN, TCN, CNN best); Guerra (TCN) | Use a 1D CNN/TCN over the 168-h axis, with channels = price, inflows, broadcast WV/volumes, outputs = 14 units |
| Temporal persistence (x_t = x_{t+1}), min up/down | Xavier (affine hints), Za'ter (post-processing) | Smoothing or lag features of predictions; sequence models |
| Relaxed-LP solution as a feature | Barjaktarevic (No_MIP outputs as inputs; No_MIP alone gives F1 0.9735) | We can approximate a "relaxed" decision with a price-vs-water-value rule (see below) |
| Random split overstates accuracy | Pineda (LOO), Barjaktarevic (random 80/20) vs Lin (temporal split: ~0.8 precision) | Validate on a forward-chaining split (e.g., train ≤2020 → val 2021, ≤2021 → 2022) |
| Distribution shift is the main weakness | Xavier (OOD tests), Reite (price regime), Mohammadi | Normalize prices relative to water values / weekly mean; consider recency weighting |
| Classification metric ≠ operational value | Barjaktarevic | Not our concern (the metric is AUC), but worth noting in the report |

---

## D. Implications for our project (prioritized)

1. **Set up a validation scheme that mimics the test before any modelling.**
   - Use forward-chaining by year, e.g. folds validating on 2020, 2021 and 2022 with all earlier data as training. Add a **gap of ≥ 7 days** between train and validation, because adjacent cases share 6 of 7 days of inputs and labels.
   - Do not trust random K-fold. The SINTEF paper's ~97.7% F1 comes from a random split.
   - Report **micro ROC-AUC**, plus per-unit AUC for diagnosis.

2. **Feature engineering around "price vs. water value". This is where the economics is.**
   - Build per unit and hour: `price_t / WV_res(unit)` and `price_t − WV_res(unit)`, using the water value of the reservoir feeding that plant (from the YAML topology).
   - Also: `price_t − mean(price over horizon)`, the price rank and percentile within the week (and within the day), and the hour of horizon. Hour-of-day and day-of-week are probably encoded implicitly by the case start date.
   - Add season (day of year), initial volume relative to the seasonal min/median/max (this mirrors SINTEF's scarcity coefficient), and inflow aggregates (current and next 24 h, whole week) per station group.
   - Rationale: SINTEF's water values are (very likely) `scarcity × 168-h forward mean price`, so these ratios carry the core signal. They are also regime-robust across the 2021–22 price spike.

3. **A strong tabular baseline: GBDT on a long-format table.**
   - One row per (case, unit, hour). Columns: unit and plant ID as categorical, plus the engineered features above and neighbouring-hour features (price at t−3..t+3, rolling means).
   - LightGBM/XGBoost with early stopping on the temporal fold.
   - The SINTEF paper did not test GBDT. This fills an obvious gap and is well suited to about 6.9M rows (2,922 × 2,352). **[my judgement]**
   - Also train a per-unit variant, or add unit-interaction features, since units behave very differently.

4. **k-NN over historical cases (Pineda & Morales; Xavier).**
   - Use a distance on standardized summaries: relative price profile, WV/price ratios, volumes, season.
   - Predict the distance-weighted average of the neighbours' 14×168 label matrices. This is a cheap, calibrated probability and a good ensemble member.
   - Restrict or weight neighbours by price regime if that helps on validation.

5. **Sequence model: 1D CNN / TCN (dilated) with 14×168 sigmoid outputs, BCE loss.**
   - Inputs as channels over 168 h: price, price/WV per plant, inflows; broadcast scalars such as initial volume and season.
   - This mirrors the best architecture in the closest paper. Keep it small (≤ 300k parameters), use dropout and weight decay, and ensemble several seeds.
   - MLP is a cheap alternative with nearly the same performance.

6. **Exploit structure.**
   - **Temporal smoothness:** UC rarely flips hour to hour (Xavier's x_t = x_{t+1} hint). Light smoothing of predicted probabilities along the time axis, or lagged-probability stacking, may help AUC.
   - **Units at the same plant (Tokke G1–G4, Vinje G1–G3)** probably commit in a priority order (e.g., G2 on ⇒ G1 on). Check this in EDA; if it holds, enforce monotonic ordering of probabilities within the plant or use cumulative "number of units on" targets. **[hypothesis, verify in data]**
   - **Overlapping cases:** the hour (date d, hour h) appears in up to 7 test cases with similar inputs. Averaging a model's predictions for the same absolute hour across overlapping cases (excluding end-of-horizon hours, which behave differently) is a legitimate test-time ensemble that uses only test inputs. **[hypothesis; validate on CV]**
   - **End-of-horizon effects:** the last hours depend on end water values. Keep "hour index in horizon" as a feature.

7. **Unit-level base rates dominate micro-AUC.** Some units are almost always on or off. Micro-AUC pools all units, so getting each unit's base rate right (unit ID, seasonal availability) gives much of the score. Check this first and make sure every model includes unit-level offsets.

8. **Expect label noise from the label-selection procedure.** Labels are the best of All/Mixed/No_MIP schedules, and the No_MIP rule over-commits (recall 0.995, precision 0.953). Do not overfit to perfectly separating training labels. Prefer regularized and calibrated models, and consider label smoothing for NNs.

9. **Distribution shift (2021–22 crisis → 2023–24 test).**
   - Use relative and normalized price features.
   - Check how validation on 2022 behaves when training includes vs. excludes 2021–22.
   - Try sample weighting that emphasises recent years. SINTEF itself restricts price sampling to recent years for this reason.

10. **External data and leakage.** A public SINTEF dataset appears to contain the same cases, with labels and No_MIP outputs, possibly covering 2023–24. **Do not use any test-period labels.** Check the competition rules before using any external SINTEF data at all, including the synthetic dataset for augmentation. If augmentation is allowed, the synthetic dataset (Reite et al.) is the intended way to fight data scarcity.

---

## Sources not found or not verifiable
- I found no paper using ML for UC in SHOP besides #1 and #6. The InterOpt news article mentions 2–260× speedups from the general literature, but gives no InterOpt accuracy numbers.
- No source reports ROC-AUC for hydro UC prediction. The closest comparable numbers are micro-F1 ≈ 0.975–0.978 (random split, with relaxed-LP features). Our AUC on a temporal split without LP features will likely look different. **[uncertain]**
- The Milla & Pérez-Díaz review (ELECTRIMACS 2024; cited in #1 as reporting up to 85% time savings when all binaries are predicted) was found only as a citation, not read.
