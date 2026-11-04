# Matched improved LightGBM and XGBoost models

Both models use exactly the same 56 features, dropped columns, training rows (60%, seed 0), 2,200 trees, learning rate 0.03, 127-leaf limit and 80% row/feature subsampling. The existing library-specific leaf-size translation is preserved: LightGBM min_child_samples=200, XGBoost min_child_weight=50. The unit feature remains categorical in LightGBM and numeric in XGBoost, as in the original pair.

## Results

| Model | Version | A (2020) | B (2021) | C (2022) | D (2019) | Mean A–D | Gain | Holdout 2018 | Public LB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LightGBM | Previous | 0.9868 | 0.9941 | 0.9922 | 0.9919 | 0.9912 | +0.00000 | 0.9877 | 0.99414 |
| LightGBM | Improved | 0.9893 | 0.9953 | 0.9935 | 0.9941 | 0.9930 | +0.00179 | 0.9896 | Not submitted |
| XGBoost | Previous | 0.9860 | 0.9940 | 0.9922 | 0.9916 | 0.9910 | +0.00000 | 0.9876 | 0.99429 |
| XGBoost | Improved | 0.9883 | 0.9952 | 0.9935 | 0.9939 | 0.9927 | +0.00174 | 0.9896 | Not submitted |

The gain is relative to the previous model of the same library. All scores use the unchanged A–D year splits, seven-day embargo and official metric. The 2018 check is performed after model selection. No new Kaggle score has been measured.

## Notebooks

- `Short_notebook_1.ipynb`: exp 039, improved LightGBM.
- `Short_notebook_2.ipynb`: exp 040, matched improved XGBoost.
- Rebuild both: `.venv/bin/python tools/build_notebooks.py`.
- Rebuild the previous 029/030 pair: `.venv/bin/python tools/build_notebooks.py --legacy`.
- Standalone wiring was checked on three real cases: inline features and small-model predictions matched the harness. The full training/submission runs use the harness.

## Report update guide

The comparison and hyperparameter tables are exported to `report/tables/tab_updated_models.tex` and `report/tables/tab_updated_hyperparameters.tex`. Full-precision CSVs are in `experiments/paired_model_comparison.csv` and `experiments/paired_hyperparameters.csv`. These are additional tables; existing authored report text and historical analysis are retained.

Describe the new development steps as efficient-segment terminal-water valuation (037), horizon/cascade/start-cost context (038), then a larger fit (039), with XGBoost applying the same final pipeline (040). Earlier conclusions that more rows/trees did not help referred to the older feature set; 039 adds a small gain on the richer features.

Keep public scores 0.99414 and 0.99429 attached to experiments 029 and 030. Do not attach them to the improved notebooks until those files are actually scored on Kaggle. The earlier logistic/RF/MLP comparisons used the previous feature set and should be labelled accordingly.

For report edits: update the feature discussion (Sections 4–5), the matched settings and historical tuning interpretation (Section 6), final-model figures (Section 7), and final comparison (Section 8). Full CV predictions remain pred_A.npy through pred_D.npy. After a holdout run, scores.csv describes that latest run; use scores_cv.csv or scores_all.csv for the original four-year results, and importance_cv.csv for their mean feature importance.

`Report.ipynb` retains the previous analysis. For its final-model figures and tables, change the configuration/prediction references to exp 039/040 while keeping exp 029/030 in the historical development sequence. The new paired CSV can be read directly with pandas.

Re-export these tables after recording a new result: `.venv/bin/python tools/build_paired_results.py`.
