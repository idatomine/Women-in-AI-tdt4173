"""Check standalone notebook wiring against cached harness features and predictions."""
import ast
import copy
from dataclasses import replace
import importlib
import sys
import types
import unittest
from unittest.mock import patch

import nbformat
import numpy as np
import yaml

from src import features, models
from src.data import ROOT, HOURS, load_data
from tools.build_notebooks import build_notebook, NOTEBOOKS, LEGACY_NOTEBOOKS, PUBLIC_LB


class NotebookPipelineTests(unittest.TestCase):
    def test_matched_configs_and_standalone_pipelines(self):
        configs = [yaml.safe_load((ROOT / p).read_text()) for p in NOTEBOOKS.values()]
        self.assertEqual(configs[0]['features'], configs[1]['features'])
        self.assertEqual(configs[0]['drop_features'], configs[1]['drop_features'])
        self.assertEqual(configs[0]['model']['params']['num_leaves'], configs[1]['model']['params']['max_leaves'])
        for key in ['row_share', 'seed']:
            self.assertEqual(configs[0]['model'][key], configs[1]['model'][key])
        for key in ['n_estimators', 'learning_rate', 'subsample', 'colsample_bytree']:
            self.assertEqual(configs[0]['model']['params'][key], configs[1]['model']['params'][key])

        full = load_data()
        idx = np.array([100, 1000, 2500])
        data = replace(full, run_no=full.run_no[idx], starts=full.starts.iloc[idx].reset_index(drop=True),
                       n_train=len(idx), Y=full.Y[idx])
        cached = {}

        def small_group(group, _):
            if group not in cached:
                if group == 'lp_dispatch_context':
                    module = importlib.import_module('src.features.lp_dispatch_context')
                    with patch.object(module, 'load_group', side_effect=small_group):
                        values = module.build(data)
                    a = np.empty((len(idx), len(data.units), HOURS, len(values)), dtype=np.float32)
                    for j, v in enumerate(values.values()):
                        a[..., j] = np.broadcast_to(v, a.shape[:-1])
                    cached[group] = a, list(values)
                else:
                    a, names = features.load_group(group, full)
                    cached[group] = np.asarray(a[idx]), names
            return cached[group]

        # Resolve cached groups before patching the loader used by build_matrix.
        for group in configs[0]['features']:
            small_group(group, data)
        with patch.object(features, 'load_group', side_effect=lambda group, _: cached[group]):
            expected, expected_names = features.build_matrix(configs[0]['features'], data, configs[0]['drop_features'])

        for config_path, cfg in zip(NOTEBOOKS.values(), configs):
            notebook = build_notebook(config_path)
            nbformat.validate(notebook)
            self.assertIn(f"public LB {PUBLIC_LB[cfg['name']]}", notebook.cells[0].source)
            module = types.ModuleType('standalone_notebook_check')
            sys.modules[module.__name__] = module
            ns = module.__dict__
            try:
                for cell in notebook.cells:
                    if cell.cell_type != 'code':
                        continue
                    tree = ast.parse(cell.source)
                    self.assertFalse(any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('src')
                                         for n in ast.walk(tree)))
                    source = cell.source
                    if source.startswith('MODEL ='):
                        continue  # train a small deterministic model below instead
                    if 'data = load_data()' in source:
                        source = source.split('data = load_data()', 1)[0]
                    exec(compile(source, '<standalone notebook>', 'exec'), ns)
                actual, names = ns['build_matrix'](data)
                self.assertEqual(names, expected_names)
                self.assertEqual(actual.shape[-1], 56)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
                self.assertEqual(ns['WORKERS'], 1)
                tiny = copy.deepcopy(cfg['model'])
                tiny['params']['n_estimators'] = 12
                train, val = np.array([0, 1]), np.array([2])
                native = models.fit(tiny, expected, data.Y, train, names)
                inline = ns['fit'](tiny, actual, data.Y, train, names)
                np.testing.assert_allclose(ns['predict'](inline, actual, val),
                                           models.predict(native, expected, val), rtol=0, atol=1e-12)
            finally:
                sys.modules.pop(module.__name__, None)

    def test_legacy_generation_still_parses(self):
        for path in LEGACY_NOTEBOOKS.values():
            notebook = build_notebook(path)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    ast.parse(cell.source)


if __name__ == '__main__':
    unittest.main()
