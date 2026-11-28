import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from leakage_audit import audit_recordings, audit_window_features, exclude_notebook_recordings
from model_tuning import (BalancedXGBClassifier, bootstrap_ci, build_models, evaluate_dataset,
                          grouped_splits, permute_group_labels, search_space_for_splits,
                          validation_groups)


class LeakageEvaluationTests(unittest.TestCase):
    def test_notebook_excludes_only_selected_borisov_recording(self):
        table = pd.DataFrame({
            'dataset': ['borisov', 'borisov', 'borisov', 'borisov', 'sakib'],
            'subject_id': ['SZ_387-02w1', 'SZ_387-02w1', 'SZ_387-03w1', 'SZ_other', 'SZ_387-02w1'],
            'window': [0, 1, 0, 0, 0],
        })
        filtered = exclude_notebook_recordings(table)
        self.assertEqual(filtered[['dataset', 'subject_id']].values.tolist(),
                         [['borisov', 'SZ_387-03w1'], ['borisov', 'SZ_other'], ['sakib', 'SZ_387-02w1']])
        self.assertEqual(len(table), 5)

    def test_related_recordings_stay_in_same_group_and_predictions_are_separate(self):
        subjects = ['SZ_387-02w1', 'SZ_387-03w1'] + [f'subject{i}' for i in range(22)]
        labels = np.array([1, 1] + [i % 2 for i in range(22)])
        groups = validation_groups('borisov', subjects)
        self.assertEqual(groups[0], groups[1])
        X = pd.DataFrame({'feature': labels + np.arange(len(labels)) / 100})
        splits = grouped_splits(X, labels, groups, 3, 42)
        self.assertTrue(all(not ({0, 1} & set(train) and {0, 1} & set(test))
                            for train, test in splits))
        table = X.assign(subject_id=subjects, label=labels)
        pipeline, _ = build_models()['logistic_regression']
        models = {'logistic_regression': (pipeline, {'select__percentile': [100], 'model__C': [1]})}
        profile = {'n_iter': 1, 'n_iter_trees': 1, 'outer_folds': 3, 'inner_folds': 2, 'repeats': 1}
        _, _, predictions, _ = evaluate_dataset(table, ['feature'], 'task', 'borisov',
                                                 profile=profile, n_jobs=1, models=models)
        self.assertEqual(set(predictions.subject_id), set(subjects))

    def test_knn_candidates_respect_smallest_inner_training_fold(self):
        space = build_models()['knn'][1]
        splits = [(np.arange(12), np.arange(12, 16)), (np.arange(9), np.arange(9, 16))]
        safe = search_space_for_splits('knn', space, splits)
        self.assertEqual(safe['model__n_neighbors'], list(range(3, 10)))
        self.assertTrue(hasattr(space['model__n_neighbors'], 'dist'))
        with self.assertRaisesRegex(ValueError, 'No kNN candidates'):
            search_space_for_splits('knn', {'model__n_neighbors': [10, 11]}, splits)

    def test_knn_small_folds_complete_without_invalid_candidates(self):
        labels = np.tile([0, 1], 12)
        table = pd.DataFrame({'subject_id': np.arange(len(labels)), 'label': labels,
                              'feature': labels + np.arange(len(labels)) / 100})
        pipeline, space = build_models()['knn']
        space = {**space, 'select__percentile': [100]}
        profile = {'n_iter': 2, 'n_iter_trees': 1, 'outer_folds': 2, 'inner_folds': 2, 'repeats': 2}
        metrics, _, predictions, tuning = evaluate_dataset(
            table, ['feature'], 'task', 'test', profile=profile, n_jobs=1, models={'knn': (pipeline, space)})
        self.assertEqual(len(tuning), 4)
        self.assertTrue(tuning.inner_roc_auc.notna().all())
        self.assertEqual(len(predictions), len(labels))
        self.assertGreaterEqual(metrics.iloc[0].ba_sd_over_repeats, 0)

    def test_xgboost_weights_are_computed_from_each_fit(self):
        y = np.array([0, 0, 0, 1])
        with patch.object(XGBClassifier, 'fit', return_value=None) as fit:
            BalancedXGBClassifier().fit(np.arange(4).reshape(-1, 1), y)
        np.testing.assert_allclose(fit.call_args.kwargs['sample_weight'], [2 / 3, 2 / 3, 2 / 3, 2])

    def test_permutation_preserves_labels_within_groups(self):
        groups = np.array(['a', 'a', 'b', 'c', 'c', 'd'])
        labels = np.array([0, 0, 1, 1, 1, 0])
        result = permute_group_labels(labels, groups, np.random.default_rng(42))
        self.assertEqual(result[0], result[1])
        self.assertEqual(result[3], result[4])
        self.assertEqual(sorted(result[[0, 2, 3, 5]]), [0, 0, 1, 1])

    def test_duplicate_feature_windows_across_groups_are_rejected(self):
        table = pd.DataFrame({'subject_id': ['a', 'b'], 'label': [0, 0],
                              'window': [0, 0], 'feature': [1.0, 1.0]})
        with self.assertRaisesRegex(ValueError, 'different validation groups'):
            audit_window_features(table, ['feature'], 'test')
        table.loc[1, 'feature'] = 2
        audit_window_features(table, ['feature'], 'test')

    def test_duplicate_raw_segments_across_groups_are_rejected(self):
        table = pd.DataFrame({'subject_id': ['a', 'b'], 'label': [0, 0]})
        a = np.arange(20).reshape(1, 20)
        b = np.concatenate([a[:, :10], np.arange(20, 30).reshape(1, 10)], axis=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'signals.npz'
            np.savez(path, a=a, b=b)
            with self.assertRaisesRegex(ValueError, 'share raw data'):
                audit_recordings(path, table, 'test', fs=1, window_seconds=10)

    def test_raw_metadata_label_mismatch_is_rejected(self):
        table = pd.DataFrame({'subject_id': ['a'], 'label': [1]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'signals.npz'
            metadata = Path(tmp) / 'metadata.csv'
            np.savez(path, a=np.arange(20).reshape(1, 20))
            pd.DataFrame({'subject_id': ['a'], 'label': [0]}).to_csv(metadata, index=False)
            with self.assertRaisesRegex(ValueError, 'do not match raw metadata'):
                audit_recordings(path, table, 'test', fs=1, metadata_path=metadata)

    def test_repeat_bootstrap_uses_mean_repeat_metric(self):
        y = np.array([0, 0, 1, 1])
        preds = np.array([[0, 0, 1, 1], [1, 1, 0, 0]])
        low, high = bootstrap_ci(y, preds, n_boot=100, seed=1)
        self.assertEqual((low, high), (0.5, 0.5))


if __name__ == '__main__':
    unittest.main()
