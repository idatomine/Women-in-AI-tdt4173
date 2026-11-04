"""Regression checks for the lower-memory feature and row selection paths."""
import unittest
from unittest.mock import patch

import numpy as np

from src import features, models


class MemoryLayoutTests(unittest.TestCase):
    def test_sampled_rows_labels_and_weights_match_original_order(self):
        X = np.arange(12 * 3 * 5 * 4, dtype=np.float32).reshape(12, 3, 5, 4)
        Y = (X[..., 0] % 3 == 0).astype(int)
        idx = np.array([0, 3, 6, 8, 11])
        cfg = {'type': 'lightgbm', 'row_share': .35, 'seed': 7, 'params': {}}
        keep = np.random.default_rng(7).random(Y[idx].size) < cfg['row_share']
        case_weights = np.arange(12, dtype=float) + 1
        expected_weight = np.broadcast_to(case_weights[idx, None, None], Y[idx].shape).ravel()[keep]
        with patch.object(models, 'case_weights', return_value=case_weights), patch.object(models, '_train') as train:
            models.fit(cfg, X, Y, idx, ['a', 'b', 'c', 'd'])
        args = train.call_args.args
        np.testing.assert_array_equal(args[1], X[idx].reshape(-1, 4)[keep])
        np.testing.assert_array_equal(args[2], Y[idx].ravel()[keep])
        np.testing.assert_array_equal(args[5], expected_weight)

    def test_matrix_values_and_column_order_match_concatenation(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(2, 3, 168, 4)).astype('float32')
        b = rng.normal(size=(2, 3, 168, 2)).astype('float32')
        groups = {'a': (a, ['a0', 'a1', 'a2', 'a3']), 'b': (b, ['b0', 'b1'])}
        data = type('Data', (), {'n_cases': 2, 'units': ['u1', 'u2', 'u3']})()
        with patch.object(features, 'load_group', side_effect=lambda group, _: groups[group]):
            actual, names = features.build_matrix(['a', 'b'], data, ['a1'])
        np.testing.assert_array_equal(actual, np.concatenate([a[..., [0, 2, 3]], b], axis=-1))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(names, ['a0', 'a2', 'a3', 'b0', 'b1'])


if __name__ == '__main__':
    unittest.main()
