"""Export matched old/new model results for the report, without retraining.

Run after exp 040 validation/holdout: .venv/bin/python tools/build_paired_results.py
No new public leaderboard scores are inferred from local validation.
"""
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
PAIRS = [('LightGBM', 'Previous', 'exp_029_main_river'),
         ('LightGBM', 'Improved', 'exp_039_dispatch_capacity'),
         ('XGBoost', 'Previous', 'exp_030_xgb_main_river'),
         ('XGBoost', 'Improved', 'exp_040_xgb_dispatch_capacity')]
PUBLIC = {'exp_029_main_river': .99414, 'exp_030_xgb_main_river': .99429}


def markdown_table(frame):
    rows = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
            '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows += ['| ' + ' | '.join(map(str, row)) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join(rows)


def latex_table(frame):
    """Export booktabs tables without pandas' optional Jinja2 dependency."""
    escapes = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
               '–': '--', '—': '---'}
    def escape(value):
        return ''.join(escapes.get(c, c) for c in str(value))
    rows = [r'\begin{tabular}{' + 'l' * len(frame.columns) + '}', r'\toprule',
            ' & '.join(map(escape, frame.columns)) + r' \\', r'\midrule']
    rows += [' & '.join(map(escape, row)) + r' \\' for row in frame.itertuples(index=False, name=None)]
    rows += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(rows) + '\n'


def build():
    results = pd.read_csv(ROOT / 'experiments/results.csv')
    complete = results.dropna(subset=['A', 'B', 'C', 'D']).drop_duplicates('name', keep='last').set_index('name')
    holdout = results.dropna(subset=['holdout_2018']).drop_duplicates('name', keep='last').set_index('name')
    rows, baseline = [], {}
    for model, version, name in PAIRS:
        score = complete.loc[name]
        if version == 'Previous':
            baseline[model] = score['mean']
        rows.append({'model': model, 'version': version, 'experiment': name,
                     **{f: score[f] for f in ['A', 'B', 'C', 'D', 'mean']},
                     'gain': score['mean'] - baseline[model],
                     'holdout_2018': holdout.loc[name, 'holdout_2018'] if name in holdout.index else float('nan'),
                     'public_lb': PUBLIC.get(name, float('nan'))})
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / 'experiments/paired_model_comparison.csv', index=False)
    display = table.drop(columns='experiment').copy()
    for c in ['A', 'B', 'C', 'D', 'mean', 'holdout_2018']:
        display[c] = display[c].map(lambda v: '—' if pd.isna(v) else f'{v:.4f}')
    display['gain'] = display['gain'].map(lambda v: f'{v:+.5f}')
    display['public_lb'] = display['public_lb'].map(lambda v: 'Not submitted' if pd.isna(v) else f'{v:.5f}')
    display = display.rename(columns={'model': 'Model', 'version': 'Version', 'A': 'A (2020)', 'B': 'B (2021)', 'C': 'C (2022)', 'D': 'D (2019)', 'mean': 'Mean A–D', 'gain': 'Gain', 'holdout_2018': 'Holdout 2018', 'public_lb': 'Public LB'})
    (ROOT / 'report/tables/tab_updated_models.tex').write_text(latex_table(display))

    configs = [yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text()) for _, _, name in PAIRS if name.startswith(('exp_039', 'exp_040'))]
    a, b = configs
    assert a['features'] == b['features'] and a['drop_features'] == b['drop_features']
    for key in ['row_share', 'seed']:
        assert a['model'][key] == b['model'][key]
    for key in ['n_estimators', 'learning_rate', 'subsample', 'colsample_bytree']:
        assert a['model']['params'][key] == b['model']['params'][key]
    lg, xg = a['model'], b['model']
    assert lg['params']['num_leaves'] == xg['params']['max_leaves']
    settings = pd.DataFrame([
        ('Features', 'Same 56 input-only features', 'Same 56 input-only features'),
        ('Training row share', f"{lg['row_share']:.0%}", f"{xg['row_share']:.0%}"),
        ('Trees', lg['params']['n_estimators'], xg['params']['n_estimators']),
        ('Learning rate', lg['params']['learning_rate'], xg['params']['learning_rate']),
        ('Leaves', lg['params']['num_leaves'], xg['params']['max_leaves']),
        ('Tree growth', 'leaf-wise', f"{xg['params']['grow_policy']}, max_depth={xg['params']['max_depth']}"),
        ('Minimum leaf setting', 'min_child_samples=200', 'min_child_weight=50'),
        ('Rows per tree', lg['params']['subsample'], xg['params']['subsample']),
        ('Features per tree', lg['params']['colsample_bytree'], xg['params']['colsample_bytree']),
        ('Seed', lg['seed'], xg['seed']),
    ], columns=['Setting', 'LightGBM (039)', 'XGBoost (040)'])
    settings.to_csv(ROOT / 'experiments/paired_hyperparameters.csv', index=False)
    (ROOT / 'report/tables/tab_updated_hyperparameters.tex').write_text(latex_table(settings))
    lines = ['# Matched improved LightGBM and XGBoost models', '',
             'Both models use exactly the same 56 features, dropped columns, training rows (60%, seed 0), 2,200 trees, learning rate 0.03, 127-leaf limit and 80% row/feature subsampling. The existing library-specific leaf-size translation is preserved: LightGBM min_child_samples=200, XGBoost min_child_weight=50. The unit feature remains categorical in LightGBM and numeric in XGBoost, as in the original pair.', '',
             '## Results', '', markdown_table(display), '',
             'The gain is relative to the previous model of the same library. All scores use the unchanged A–D year splits, seven-day embargo and official metric. The 2018 check is performed after model selection. No new Kaggle score has been measured.', '',
             '## Notebooks', '',
             '- `Short_notebook_1.ipynb`: exp 039, improved LightGBM.',
             '- `Short_notebook_2.ipynb`: exp 040, matched improved XGBoost.',
             '- Rebuild both: `.venv/bin/python tools/build_notebooks.py`.',
             '- Rebuild the previous 029/030 pair: `.venv/bin/python tools/build_notebooks.py --legacy`.',
             '- Standalone wiring was checked on three real cases: inline features and small-model predictions matched the harness. The full training/submission runs use the harness.', '',
             '## Report update guide', '',
             'The comparison and hyperparameter tables are exported to `report/tables/tab_updated_models.tex` and `report/tables/tab_updated_hyperparameters.tex`. Full-precision CSVs are in `experiments/paired_model_comparison.csv` and `experiments/paired_hyperparameters.csv`. These are additional tables; existing authored report text and historical analysis are retained.', '',
             'Describe the new development steps as efficient-segment terminal-water valuation (037), horizon/cascade/start-cost context (038), then a larger fit (039), with XGBoost applying the same final pipeline (040). Earlier conclusions that more rows/trees did not help referred to the older feature set; 039 adds a small gain on the richer features.', '',
             'Keep public scores 0.99414 and 0.99429 attached to experiments 029 and 030. Do not attach them to the improved notebooks until those files are actually scored on Kaggle. The earlier logistic/RF/MLP comparisons used the previous feature set and should be labelled accordingly.', '',
             'For report edits: update the feature discussion (Sections 4–5), the matched settings and historical tuning interpretation (Section 6), final-model figures (Section 7), and final comparison (Section 8). Full CV predictions remain pred_A.npy through pred_D.npy. After a holdout run, scores.csv describes that latest run; use scores_cv.csv or scores_all.csv for the original four-year results, and importance_cv.csv for their mean feature importance.', '',
             '`Report.ipynb` retains the previous analysis. For its final-model figures and tables, change the configuration/prediction references to exp 039/040 while keeping exp 029/030 in the historical development sequence. The new paired CSV can be read directly with pandas.', '',
             'Re-export these tables after recording a new result: `.venv/bin/python tools/build_paired_results.py`.', '']
    (ROOT / 'docs/PAIRED_MODELS_2026-11-04.md').write_text('\n'.join(lines))
    print(table.to_string(index=False))
    return table


if __name__ == '__main__':
    build()
