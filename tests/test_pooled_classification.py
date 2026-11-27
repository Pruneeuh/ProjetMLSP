import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from eeg_config import CANAUX
from pooled_classification import (DATASET_BASE, ParticipantBalancedClassifier, evaluate_pooled,
                                   pooled_models, pooled_splits, prepare_pooled_features)


def example_data():
    rows = []
    columns = {}
    for dataset_index, (dataset, base) in enumerate(DATASET_BASE.items()):
        columns[dataset] = [f'ch{i:02d}_alpha_SampEn' for i in range(1, len(CANAUX[base]) + 1)]
        for participant in range(12):
            for window in range(2):
                row = {'dataset': dataset, 'subject_id': f'p{participant}', 'window': window,
                       'label': participant % 2}
                row.update({name: participant % 2 + i / 100 + window / 1000000
                            + participant / 10000 + dataset_index / 100000000
                            for i, name in enumerate(columns[dataset])})
                rows.append(row)
    return pd.DataFrame(rows), columns


class PooledClassificationTests(unittest.TestCase):
    def test_common_channel_alignment_and_dataset_qualified_participants(self):
        frame, columns = example_data()
        pooled, features, channels = prepare_pooled_features(frame, columns)
        self.assertEqual(len(channels), 10)
        self.assertEqual(len(features), 10)
        self.assertEqual(pooled.subject_key.nunique(), 36)
        for dataset_index, (dataset, base) in enumerate(DATASET_BASE.items()):
            idx = CANAUX[base].index('F7')
            actual = pooled.loc[pooled.dataset == dataset, 'F7_alpha_SampEn'].iloc[0]
            self.assertAlmostEqual(actual, idx / 100 + dataset_index / 100000000)
        self.assertEqual(set(pooled.class_label), {0, 1, 2, 3})

    def test_missing_common_feature_is_rejected(self):
        frame, columns = example_data()
        idx = CANAUX[DATASET_BASE['sakib']].index('F7') + 1
        columns['sakib'].remove(f'ch{idx:02d}_alpha_SampEn')
        with self.assertRaisesRegex(ValueError, 'feature sets do not match'):
            prepare_pooled_features(frame, columns)

    def test_exact_pooled_feature_duplicates_are_rejected(self):
        frame, columns = example_data()
        original = frame.index[(frame.dataset == 'mumtaz') & (frame.subject_id == 'p0') & (frame.window == 0)][0]
        duplicate = frame.index[(frame.dataset == 'mumtaz') & (frame.subject_id == 'p2') & (frame.window == 0)][0]
        frame.loc[duplicate, columns['mumtaz']] = frame.loc[original, columns['mumtaz']]
        with self.assertRaisesRegex(ValueError, 'different validation groups'):
            prepare_pooled_features(frame, columns)

    def test_split_keeps_groups_together_and_stratifies_each_dataset(self):
        frame, columns = example_data()
        pooled, _, _ = prepare_pooled_features(frame, columns)
        strata = pooled.dataset + '::' + pooled.binary_label.astype(str)
        for train, test in pooled_splits(pooled, strata, 3, 42):
            self.assertFalse(set(pooled.validation_group.iloc[train]) & set(pooled.validation_group.iloc[test]))
            self.assertEqual(set(strata.iloc[test]), set(strata))

    def test_class_weights_balance_participants_not_windows(self):
        y = np.array([0, 0, 0, 0, 1])
        weights = np.array([0.25, 0.25, 0.25, 0.25, 1.0])
        with patch.object(LogisticRegression, 'fit', autospec=True) as fit:
            fit.side_effect = lambda estimator, X, y, sample_weight: setattr(estimator, 'classes_', np.unique(y))
            ParticipantBalancedClassifier(LogisticRegression()).fit(
                np.arange(5).reshape(-1, 1), y, sample_weight=weights)
            applied = fit.call_args.kwargs['sample_weight']
        self.assertAlmostEqual(applied[y == 0].sum(), applied[y == 1].sum())

    def test_binary_multiclass_and_control_site_end_to_end(self):
        frame, columns = example_data()
        pooled, features, _ = prepare_pooled_features(frame, columns)
        pipeline, _ = pooled_models()['logistic_regression']
        models = {'logistic_regression': (pipeline, {'model__estimator__C': [1]})}
        for task, count in [('binary', 36), ('multiclass', 36), ('control_site', 18)]:
            with self.subTest(task=task):
                metrics, predictions, tuning = evaluate_pooled(
                    pooled, features, task, models, repeats=1, outer_folds=3, inner_folds=2,
                    n_iter=1, n_jobs=1)
                self.assertEqual(metrics.iloc[0].n_participants, count)
                self.assertEqual(len(predictions), count)
                self.assertEqual(len(tuning), 3)
                prob_columns = [c for c in predictions if c.startswith('prob_')]
                np.testing.assert_allclose(predictions[prob_columns].sum(axis=1), 1)


if __name__ == '__main__':
    unittest.main()
