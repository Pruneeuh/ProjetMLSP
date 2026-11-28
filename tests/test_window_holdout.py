import unittest
from types import SimpleNamespace
from unittest.mock import patch

import eeg_tuning
import numpy as np
import pandas as pd

from eeg_config import BASES
from eeg_fenetres import matrice_fenetres, moyenner_sujets, partition_sujets, plis_groupes
from tests.test_participant_tuning import multiclass


class WindowHoldoutTests(unittest.TestCase):
    def test_partition_keeps_linked_recordings_and_strata_together(self):
        rows = []
        for base in BASES:
            for label in (0, 1):
                for i in range(10):
                    rows.append({"base": base, "label_binaire": label, "classe": label,
                                 "groupe": f"{base}_{label}_{i}"})
        rows.append(rows[0].copy())
        meta = pd.DataFrame(rows)
        split = partition_sujets(meta, seed=7)
        self.assertEqual(split.iloc[0], split.iloc[-1])
        self.assertEqual(int(split.iloc[:-1].eq("test").sum()), 12)
        counts = meta.assign(partition=split).groupby(["base", "label_binaire", "partition"]).size()
        self.assertTrue(counts.ge(1).all())
        groups = meta["groupe"].to_numpy()
        strata = (meta["base"] + meta["label_binaire"].astype(str)).to_numpy()
        for tr, va in plis_groupes(strata, groups, 3, 42):
            self.assertFalse(set(groups[tr]) & set(groups[va]))

    def test_incomplete_window_is_rejected(self):
        from eeg_config import BANDS, MEASURES
        data = pd.DataFrame([{"subject_id": "s", "window": 0, "canal": "F7", "band": band,
                              "log_sd": 1.0, **{m: 1.0 for m in MEASURES}} for band in BANDS[:-1]])
        info = {"base": "depression", "data": data, "records": pd.DataFrame(index=["s"])}
        meta = pd.DataFrame({"partition": ["train"]}, index=pd.Index(["s"], name="subject_id"))
        with self.assertRaisesRegex(ValueError, "fenêtre incomplète"):
            matrice_fenetres(info, ["F7"], meta, etendues=False)

    def test_probabilities_are_averaged_by_subject(self):
        meta = pd.DataFrame({"subject_id": ["a", "a", "b"], "groupe": ["g", "g", "h"],
                             "base": ["depression"] * 3})
        participants, proba = moyenner_sujets(meta, [0, 0, 1], np.array([[.9, .1], [.3, .7], [.2, .8]]))
        self.assertEqual(list(participants.index), ["a", "b"])
        np.testing.assert_allclose(proba, [[.6, .4], [.2, .8]])

    def test_inner_tuning_holds_out_complete_groups_of_windows(self):
        groups = np.repeat(np.arange(32), 3)
        y = (groups % 2).astype(int)
        x = np.column_stack((groups, np.tile([0., .1, .2], 32)))
        calls = []

        def capture(x_tr, x_val, base_tr, base_val, reference, method):
            self.assertFalse(set(x_tr[:, 0]) & set(x_val[:, 0]))
            calls.append((len(x_tr), len(x_val)))
            return x_tr, x_val

        with patch.object(eeg_tuning.eh, "harmonise", side_effect=capture):
            eeg_tuning.tune_and_fit("regression_logistique", x, y, groups, y, 42,
                                    {"xgboost": "cpu", "lightgbm": "cpu"}, inner_folds=2,
                                    draws=1, bases=np.repeat("site", len(y)), reference=y == 0,
                                    harmonisation="zscore")
        self.assertEqual(len(calls), 2)

    def test_multiclass_holdout_tuning_and_harmonization_only_see_train(self):
        meta_rows, data = [], []
        for base_index, base in enumerate(BASES):
            for label in (0, 1):
                for subject in range(10):
                    identity = base_index * 100 + label * 10 + subject
                    for window in (0, 1):
                        meta_rows.append(dict(subject_id=f"{base}_{label}_{subject}", base=base,
                                              groupe=f"{base}_{label}_{subject}", label_binaire=label,
                                              classe=base_index + 1 if label else 0,
                                              partition="test" if subject < 2 else "train"))
                        data.append([identity + window / 10])
        meta, x = pd.DataFrame(meta_rows), np.array(data)
        train_values = set(x[meta.partition.eq("train"), 0])
        test_values = set(x[meta.partition.eq("test"), 0])

        class Model:
            classes_ = np.arange(4)

            def fit(self, inputs, y):
                self.assert_train = set(inputs[:, 0])
                return self

            def predict_proba(self, inputs):
                p = .1 + inputs[:, 0] / 1000
                return np.column_stack((1 - p, p / 3, p / 3, p / 3))

        def tune(name, xt, y, groups, strata, bases, ref, method, seed, args):
            self.assertEqual(set(xt[:, 0]), train_values)
            return Model(), {"best_params": {}}

        def harmonise(xt, xv, bt, bv, ref, method):
            self.assertEqual(set(xt[:, 0]), train_values)
            self.assertEqual(set(xv[:, 0]), test_values)
            return xt, xv

        args = SimpleNamespace(repetitions=1, graine=42, plis=3)
        specs = [dict(modele="test", kind="test", caracteristiques="entropies", harmonisation="zscore")]
        with patch.object(multiclass, "tuned_model", side_effect=tune), \
             patch.object(multiclass.eh, "harmonise", side_effect=harmonise):
            _, predictions, _, _ = multiclass.main_analysis({"entropies": x}, meta, args, specs, [], holdout=True)
        self.assertEqual(set(predictions.subject_id), set(meta.loc[meta.partition.eq("test"), "subject_id"]))
        self.assertEqual(len(predictions), meta.loc[meta.partition.eq("test"), "subject_id"].nunique())
        sid = "depression_0_0"
        actual = predictions.loc[predictions.subject_id.eq(sid), "proba_temoin"].iloc[0]
        self.assertAlmostEqual(actual, 1 - (.1 + .05 / 1000))


if __name__ == "__main__":
    unittest.main()
