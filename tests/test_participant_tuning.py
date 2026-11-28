import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

import eeg_tuning


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("interpretation", ROOT / "09_interpretabilite.py")
interpretation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(interpretation)
spec06 = importlib.util.spec_from_file_location("multiclass", ROOT / "06_classification_multiclasse.py")
multiclass = importlib.util.module_from_spec(spec06)
spec06.loader.exec_module(multiclass)
spec05 = importlib.util.spec_from_file_location("binary", ROOT / "05_classification_binaire.py")
binary = importlib.util.module_from_spec(spec05)
spec05.loader.exec_module(binary)
DEVICES = {"xgboost": "cpu", "lightgbm": "cpu"}


class ParticipantTuningTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.x = rng.normal(size=(48, 12))
        self.x[:, 0] = np.arange(48)
        self.groups = np.arange(48).astype(str)

    def test_all_model_searches_fit_binary_and_multiclass(self):
        models = ["regression_logistique", "mlp", "svm", "knn", "lda",
                  "foret_aleatoire", "xgboost", "lightgbm"]
        for n_classes in (2, 4):
            y = np.arange(48) % n_classes
            for name in models:
                with self.subTest(model=name, classes=n_classes):
                    model, info = eeg_tuning.tune_and_fit(
                        name, self.x, y, self.groups, y, 42, DEVICES,
                        inner_folds=2, draws=1)
                    model.fit(self.x, y)
                    proba = eeg_tuning.predict_proba(model, self.x[:3], n_classes)
                    self.assertEqual(proba.shape, (3, n_classes))
                    self.assertTrue(np.isfinite(proba).all())
                    self.assertTrue(info["best_params"])

    def test_inner_harmonization_only_sees_inner_training_rows(self):
        y = np.arange(48) % 2
        bases = np.where(np.arange(48) % 4 < 2, "a", "b")
        strata = (bases == "b").astype(int) * 2 + y
        calls = []

        def capture(x_tr, x_va, base_tr, base_va, reference, method):
            self.assertFalse(set(x_tr[:, 0]) & set(x_va[:, 0]))
            self.assertTrue(np.array_equal(reference, y[x_tr[:, 0].astype(int)] == 0))
            calls.append((len(x_tr), len(x_va)))
            return x_tr, x_va

        with patch.object(eeg_tuning.eh, "harmonise", side_effect=capture):
            eeg_tuning.tune_and_fit(
                "regression_logistique", self.x, y, self.groups, strata, 42, DEVICES,
                inner_folds=2, draws=1, bases=bases, reference=y == 0, harmonisation="zscore")
        self.assertEqual(len(calls), 2)

    def test_knn_search_restricts_neighbors_to_inner_training_size(self):
        y = np.arange(24) % 2
        model, _ = eeg_tuning.tune_and_fit(
            "knn", self.x[:24], y, self.groups[:24], y, 42, DEVICES, inner_folds=2, draws=8)
        folds = eeg_tuning.make_folds(y, self.groups[:24], 2, 42, must_have=y, min_par_strate=2)
        self.assertLessEqual(model.named_steps["model"].n_neighbors, min(len(tr) for tr, _ in folds))

    def test_gpu_probe_only_checks_requested_backends(self):
        with patch.object(eeg_tuning, "xgboost_device", return_value="cuda") as xgb, \
             patch.object(eeg_tuning, "lightgbm_device", return_value="cpu") as lgb:
            self.assertEqual(eeg_tuning.detect_devices(["xgboost"]), {"xgboost": "cuda", "lightgbm": "cpu"})
            xgb.assert_called_once()
            lgb.assert_not_called()

    def test_feature_search_handles_single_reference_variable(self):
        y = np.arange(48) % 2
        model, info = eeg_tuning.tune_and_fit(
            "regression_logistique", self.x[:, :1], y, self.groups, y, 42, DEVICES,
            inner_folds=2, draws=4)
        self.assertEqual(info["best_params"]["select__percentile"], 100)
        model.fit(self.x[:, :1], y)
        self.assertEqual(eeg_tuning.predict_proba(model, self.x[:2, :1], 2).shape, (2, 2))

    def test_fold_shap_supports_linear_tree_and_kernel_models(self):
        y = np.arange(48) % 2
        args = SimpleNamespace(shap_fond=4, shap_sujets=3, shap_echantillons=30)
        columns = [f"F7|alpha|measure{i}" for i in range(self.x.shape[1])]
        for name in ("regression_logistique", "foret_aleatoire", "xgboost", "lightgbm", "knn"):
            with self.subTest(model=name):
                model, _ = eeg_tuning.tune_and_fit(
                    name, self.x[:40], y[:40], self.groups[:40], y[:40], 42, DEVICES,
                    inner_folds=2, draws=1)
                model.fit(self.x[:40], y[:40])
                values = interpretation.expliquer_shap(model, name, self.x[:40], self.x[40:],
                                                       columns, 2, args, 42)
                self.assertEqual(set(values.variable), set(columns))
                self.assertTrue(np.isfinite(values.shap_abs_moyen).all())

    def test_multiclass_shap_preserves_class_and_feature_axes(self):
        y = np.arange(48) % 4
        args = SimpleNamespace(shap_fond=4, shap_sujets=3, shap_echantillons=30)
        columns = [f"F7|alpha|measure{i}" for i in range(self.x.shape[1])]
        for name in ("regression_logistique", "foret_aleatoire", "xgboost", "lightgbm", "knn"):
            with self.subTest(model=name):
                model, _ = eeg_tuning.tune_and_fit(
                    name, self.x[:40], y[:40], self.groups[:40], y[:40], 42, DEVICES,
                    inner_folds=2, draws=1)
                model.fit(self.x[:40], y[:40])
                values = interpretation.expliquer_shap(model, name, self.x[:40], self.x[40:],
                                                       columns, 4, args, 42)
                self.assertEqual(len(values), len(columns) * 4)
                self.assertEqual(set(values["classe"]), set(interpretation.CLASSES))
                self.assertTrue(np.isfinite(values.shap_abs_moyen).all())

    def test_window_shap_weights_participants_equally(self):
        y = np.arange(48) % 2
        model, _ = eeg_tuning.tune_and_fit(
            "regression_logistique", self.x[:40], y[:40], self.groups[:40], y[:40], 42, DEVICES,
            enabled=False)
        model.fit(self.x[:40], y[:40])
        args = SimpleNamespace(shap_fond=4, shap_sujets=2, shap_echantillons=30)
        columns = [f"F7|A4|measure{i}" for i in range(self.x.shape[1])]
        subjects = np.array(["short", "long", "long", "long"])
        values = np.vstack([np.zeros((1, len(columns))), np.full((3, len(columns)), 2.0)])
        with patch("shap.LinearExplainer") as explainer:
            explainer.return_value.shap_values.return_value = values
            result = interpretation.expliquer_shap(
                model, "regression_logistique", self.x[:40], self.x[40:44], columns, 2, args, 42,
                self.groups[:40], subjects)
        np.testing.assert_allclose(result.shap_abs_moyen, 1.0)

    def test_permutation_preserves_each_participant_trajectory(self):
        values = np.array([[10.0, 100], [20, 200], [21, 201], [22, 202]])
        subjects = np.array(["short", "long", "long", "long"])
        class Swap:
            def permutation(self, ids):
                return ids[::-1]
        result = interpretation.permuter_sujets(values, subjects, np.array([0]), Swap())
        np.testing.assert_array_equal(result[:, 0], [20, 10, 10, 10])
        np.testing.assert_array_equal(result[:, 1], values[:, 1])

    def test_multiclass_uses_nested_fold_models(self):
        rng = np.random.default_rng(42)
        bases = np.repeat(["depression", "schizophrenie", "anxiete"], 32)
        binary = np.tile(np.repeat([0, 1], 16), 3)
        y = np.where(binary, np.repeat([1, 2, 3], 32), 0)
        groups = np.array([f"subject{i}" for i in range(len(y))])
        x = rng.normal(size=(len(y), 12))
        meta = pd.DataFrame({"base": bases, "label_binaire": binary, "classe": y, "groupe": groups},
                            index=groups)
        args = SimpleNamespace(
            devices=DEVICES, profil_tuning="quick", plis_internes=2, tirages_tuning=1,
            sans_tuning=False, c=1.0, mlp_couches=[16], mlp_alpha=1.0,
            repetitions=1, graine=42, plis=3, modele="regression_logistique",
            tirages=1, shap=True, shap_fond=4, shap_sujets=3, shap_echantillons=30)
        specs = [dict(modele="regression_logistique", kind="regression_logistique",
                      caracteristiques="entropies", harmonisation="zscore")]
        tuning = []
        metrics, predictions, _, _ = multiclass.main_analysis({"entropies": x}, meta, args, specs, tuning)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(predictions), len(y))
        self.assertEqual(len(tuning), 3)

    def test_binary_nested_cv_and_multiclass_controls(self):
        rng = np.random.default_rng(7)
        bases = np.repeat(["depression", "schizophrenie", "anxiete"], 32)
        y_binary = np.tile(np.repeat([0, 1], 16), 3)
        y = np.where(y_binary, np.repeat([1, 2, 3], 32), 0)
        groups = np.array([f"subject{i}" for i in range(len(y))])
        x = rng.normal(size=(len(y), 8))
        meta = pd.DataFrame({"base": bases, "label_binaire": y_binary, "classe": y, "groupe": groups},
                            index=groups)
        args = SimpleNamespace(
            devices=DEVICES, profil_tuning="quick", plis_internes=2, tirages_tuning=1,
            sans_tuning=False, c=1.0, mlp_couches=[16], mlp_alpha=1.0, repetitions=1,
            graine=42, plis=3, harmonisations=["zscore"], lodo_calibration=0.5,
            permutations=1)
        folds = eeg_tuning.make_folds(y_binary[:32], groups[:32], 3, 42,
                                      must_have=y_binary[:32], min_par_strate=2)
        probabilities, fold_ids, choices = binary.cv_proba(
            "knn", x[:32], y_binary[:32], groups[:32], folds, 42, args)
        self.assertEqual(probabilities.shape, (32, 2))
        self.assertEqual(len(choices), 3)
        self.assertTrue((fold_ids > 0).all())
        inputs = {"entropies": x, "amplitude": x}
        site = multiclass.site_controls(inputs, meta, args, ["regression_logistique"])
        self.assertEqual(len(site), 8)
        specs = [dict(modele="regression_logistique", kind="regression_logistique",
                      caracteristiques="entropies", harmonisation="zscore")]
        lodo = multiclass.lodo(inputs, meta, args, specs)
        self.assertEqual(len(lodo), 6)
        metrics, _, _, _ = multiclass.main_analysis(inputs, meta, args, specs, [])
        permutations = multiclass.permutation_tests(inputs, meta, args, metrics, specs)
        self.assertEqual(set(permutations["permutation"]), {"globale", "intra_base"})
        self.assertTrue((permutations["n_permutations"] == 1).all())


if __name__ == "__main__":
    unittest.main()
