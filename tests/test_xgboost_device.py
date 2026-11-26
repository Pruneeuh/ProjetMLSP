import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost.core import XGBoostError

import model_tuning


class DeviceSelectionTests(unittest.TestCase):
    def test_device_probe_uses_actual_booster_device(self):
        for device, expected in [('cuda:0', 'cuda'), ('cpu', 'cpu')]:
            with self.subTest(device=device), patch('model_tuning.XGBClassifier') as classifier:
                classifier.return_value.get_booster.return_value.save_config.return_value = json.dumps({
                    'learner': {'generic_param': {'device': device}}
                })
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(model_tuning.xgboost_device(), expected)
                self.assertIn('XGBoost GPU', output.getvalue())

    def test_device_probe_reports_cuda_failure(self):
        with patch('model_tuning.XGBClassifier') as classifier:
            classifier.return_value.fit.side_effect = XGBoostError('CUDA driver unavailable')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(model_tuning.xgboost_device(), 'cpu')
            self.assertIn('CUDA driver unavailable', output.getvalue())

    def test_device_probe_reports_unsupported_xgboost(self):
        with patch('model_tuning.XGBClassifier') as classifier:
            classifier.return_value.get_booster.return_value.save_config.return_value = json.dumps({
                'learner': {'generic_param': {}}
            })
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(model_tuning.xgboost_device(), 'cpu')
            self.assertIn('XGBoost 2.0+', output.getvalue())

    def test_gpu_search_is_serial_and_cpu_search_retains_parallelism(self):
        for device, expected in [('cuda', 1), ('cpu', 4)]:
            models = model_tuning.build_models(xgb_device=device)
            self.assertEqual(model_tuning.search_n_jobs('xgboost', models['xgboost'][0], 4), expected)
            self.assertEqual(model_tuning.search_n_jobs('svm', models['svm'][0], 4), 4)

    def test_gpu_searches_propagate_fit_errors(self):
        import shap_analysis

        y = np.tile([0, 1], 12)
        table = pd.DataFrame({'subject_id': np.arange(len(y)), 'label': y, 'feature': y})
        models = {'xgboost': model_tuning.build_models(xgb_device='cuda')['xgboost']}
        profile = {'n_iter': 1, 'n_iter_trees': 1, 'outer_folds': 2, 'inner_folds': 2, 'repeats': 1}
        for target in ['model_tuning', 'shap_analysis']:
            with self.subTest(target=target), patch(f'{target}.RandomizedSearchCV') as search:
                search.return_value.fit.side_effect = RuntimeError('GPU fit failed')
                with self.assertRaisesRegex(RuntimeError, 'GPU fit failed'):
                    if target == 'model_tuning':
                        model_tuning.evaluate_dataset(
                            table, ['feature'], 'task', 'test', profile=profile, n_jobs=4, models=models)
                    else:
                        shap_analysis.fit_final_model(
                            table, ['feature'], 'xgboost', models, profile=profile, n_jobs=4)
                self.assertEqual(search.call_args.kwargs['n_jobs'], 1)
                self.assertEqual(search.call_args.kwargs['error_score'], 'raise')

    def test_cpu_nested_training_produces_predictions(self):
        rng = np.random.default_rng(42)
        y = np.tile([0, 1], 24)
        table = pd.DataFrame({
            'subject_id': np.arange(len(y)), 'label': y,
            'feature': y + rng.normal(0, 0.1, len(y)),
        })
        pipeline, _ = model_tuning.build_models(xgb_device='cpu')['xgboost']
        models = {'xgboost': (pipeline, {'select__percentile': [100], 'model__n_estimators': [2]})}
        profile = {'n_iter': 1, 'n_iter_trees': 1, 'outer_folds': 2, 'inner_folds': 2, 'repeats': 1}
        metrics, _, predictions, tuning = model_tuning.evaluate_dataset(
            table, ['feature'], 'task', 'test', profile=profile, n_jobs=1, models=models)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(predictions['probability'].between(0, 1).all())
        self.assertEqual(len(tuning), 2)
        self.assertTrue(tuning['inner_roc_auc'].notna().all())


if __name__ == '__main__':
    unittest.main()
