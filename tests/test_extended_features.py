import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import eeg_gpu_entropies
from eeg_config import BANDS, BASES, MEASURES, MESURES_ETENDUES, MULTIECHELLE, PUISSANCES
from eeg_donnees import agreger, agreger_multiechelle, charger_extraction


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


extract = load_script("03_extraction_entropies")
fusion = load_script("04_fusion_harmonisation")
binary = load_script("05_classification_binaire")
multiclass = load_script("06_classification_multiclasse")
synthesis = load_script("10_synthese_resultats")
loading = load_script("01_chargement_donnees")
pipeline = load_script("00_pipeline")


class ExtendedFeatureTests(unittest.TestCase):
    def test_pairwise_backend_matches_cpu_entropy_calculations(self):
        rng = np.random.default_rng(42)
        signals = [rng.normal(size=73), np.repeat([0.0, 0.15, 0.3], 24)]
        for x in signals:
            for m in (2, 3):
                with self.subTest(length=len(x), m=m):
                    b, a = eeg_gpu_entropies.compter_paires(x, m, 0.2, np, bloc=11)
                    actual = -np.log(a / b) if a and b else np.nan
                    np.testing.assert_allclose(actual, extract.sample_entropy(x, m=m), equal_nan=True)
                    actual_fuzzy = eeg_gpu_entropies.fuzzy_entropie(x, m, 0.2, 2, np, bloc=11)
                    np.testing.assert_allclose(actual_fuzzy, extract.fuzzy_entropy(x, m=m),
                                               rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_multiscale_gpu_pairwise_route_matches_cpu(self):
        x = np.random.default_rng(11).normal(size=128)
        expected = extract.entropies_multiechelle(x, scales=4)
        with patch.dict(sys.modules, {"cupy": np}):
            actual = extract.entropies_multiechelle(x, scales=4, calcul="cuda")
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_component_gpu_route_matches_cpu(self):
        x = np.random.default_rng(7).normal(size=86)
        params = dict(entropyhub=False, m=2, r=0.2, n_fuzzy=2, c_disp=6)
        sd, expected = extract.calculer_mesures(x, {**params, "calcul": "cpu"})
        with patch.dict(sys.modules, {"cupy": np}):
            gpu_sd, actual = extract.calculer_mesures(x, {**params, "calcul": "cuda"})
        self.assertEqual(sd, gpu_sd)
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-12, atol=1e-12, equal_nan=True)

    def test_cuda_backend_when_available(self):
        if importlib.util.find_spec("cupy") is None:
            self.skipTest("CuPy unavailable")
        backend, _ = eeg_gpu_entropies.detecter_cuda()
        if backend != "cuda":
            self.skipTest("CUDA GPU unavailable")
        import cupy as cp

        x = np.random.default_rng(13).normal(size=128)
        np.testing.assert_array_equal(eeg_gpu_entropies.compter_paires(x, 2, 0.2, cp),
                                      eeg_gpu_entropies.compter_paires(x, 2, 0.2, np))
        np.testing.assert_allclose(eeg_gpu_entropies.fuzzy_entropie(x, 2, 0.2, 2, cp),
                                   extract.fuzzy_entropy(x), rtol=1e-10, atol=1e-10)
        expected = extract.entropies_multiechelle(x, scales=4)
        actual = extract.entropies_multiechelle(x, scales=4, calcul="cuda")
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-10, equal_nan=True)
        params = dict(entropyhub=False, m=2, r=0.2, n_fuzzy=2, c_disp=6)
        _, expected = extract.calculer_mesures(x, {**params, "calcul": "cpu"})
        _, actual = extract.calculer_mesures(x, {**params, "calcul": "cuda"})
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=1e-10, atol=1e-10, equal_nan=True)

    def test_refined_multiscale_entropy_and_flat_signal(self):
        x = np.random.default_rng(42).normal(size=1280)
        values = extract.entropies_multiechelle(x, scales=3)
        z = (x - x.mean()) / x.std()
        self.assertAlmostEqual(values["RCMSE_01"], extract.sample_entropy(z, r=0.15))
        patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(z, 4), axis=1, kind="stable")
        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        self.assertAlmostEqual(values["RCMPE_01"], -np.sum(probabilities * np.log(probabilities)))
        self.assertEqual(len(values), 6)
        self.assertTrue(all(np.isnan(v) for v in extract.entropies_multiechelle(np.zeros(1280)).values()))

    @unittest.skipUnless(importlib.util.find_spec("EntropyHub"), "EntropyHub unavailable")
    def test_refined_sample_entropy_matches_entropyhub(self):
        import contextlib
        import io
        import EntropyHub as EH

        x = np.random.default_rng(42).normal(size=1279)
        z = (x - x.mean()) / x.std()
        with contextlib.redirect_stdout(io.StringIO()):
            expected, _ = EH.cMSEn(z, EH.MSobject("SampEn", m=2, r=.15), Scales=2, Refined=True)
        actual = extract.entropies_multiechelle(x, scales=2)
        np.testing.assert_allclose([actual["RCMSE_01"], actual["RCMSE_02"]], expected)

    def test_questionable_borisov_record_is_audited_and_excluded(self):
        meta = pd.DataFrame([
            dict(subject_id="387-02w1", statut="retenu", remarque=""),
            dict(subject_id="387-03w1", statut="retenu", remarque=""),
        ])
        result = loading.exclure_borisov_prudent(meta).set_index("subject_id")
        self.assertEqual(result.loc["387-02w1", "statut"], "exclu_doublon_potentiel")
        self.assertEqual(result.loc["387-03w1", "statut"], "retenu")

    def test_full_pipeline_enables_requested_analyses_without_permutation_tests(self):
        commands = []
        class FakeProcess:
            def __init__(self, cmd):
                step = Path(cmd[1]).name[:2]
                self.stdout = ([f"SORTIE : {Path(cmd[1]).parent / 'sorties' / '03_extraction' / (b + '_run')}\n"
                                for b in BASES] if step == "03" else
                               [f"SORTIE : {Path(cmd[1]).parent / 'sorties' / '04_fusion' / 'run'}\n"]
                               if step == "04" else [])
            def wait(self):
                return 0
        def launch(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess(cmd)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
             patch.object(pipeline, "signature", return_value="key"), \
             patch.object(pipeline, "CachePipeline") as cache, \
             patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
             patch.object(sys, "argv", ["00_pipeline.py"]), \
             self.assertRaises(SystemExit) as result:
            cache.return_value.trouver.return_value = None
            pipeline.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(len(commands), 9)
        by_step = {Path(cmd[1]).name[:2]: cmd for cmd in commands}
        self.assertNotIn("07", by_step)
        self.assertIn("--sans-profond", by_step["10"])
        self.assertEqual(by_step["02"][by_step["02"].index("--sortie") + 1],
                         by_step["03"][by_step["03"].index("--donnees") + 1])
        self.assertIn("--caracteristiques-etendues", by_step["04"])
        self.assertIn("--ablation", by_step["05"])
        self.assertIn("--shap", by_step["09"])
        for step in ("05", "06", "09"):
            self.assertIn("etendues_puissances", by_step[step])
            self.assertIn("--rapids", by_step[step])
        self.assertTrue(all("--permutations" not in cmd for cmd in commands))

    def test_pipeline_can_disable_rapids(self):
        commands = []

        class FakeProcess:
            stdout = []

            def wait(self):
                return 0

        def launch(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
             patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
             patch.object(sys, "argv", ["00_pipeline.py", "--etapes", "5", "6", "9", "--sans-rapids"]), \
             self.assertRaises(SystemExit) as result:
            pipeline.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(len(commands), 3)
        self.assertTrue(all("--rapids" not in command for command in commands))

    def test_pipeline_cache_records_entropy_backend(self):
        commands = []
        device = {"device": "test GPU", "cupy": "test", "driver": 1, "runtime": 1}

        class FakeProcess:
            def __init__(self):
                self.stdout = [f"SORTIE : {base}_run\n" for base in BASES]

            def wait(self):
                return 0

        def launch(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
             patch.object(pipeline, "entrees_etape", return_value=[Path(tmp)]), \
             patch.object(pipeline, "detecter_cuda", return_value=("cuda", device)), \
             patch.object(pipeline, "signature", return_value="key") as sign, \
             patch.object(pipeline, "CachePipeline") as cache, \
             patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
             patch.object(sys, "argv", ["00_pipeline.py", "--etapes", "3"]), \
             self.assertRaises(SystemExit) as result:
            cache.return_value.trouver.return_value = None
            pipeline.main()
        self.assertEqual(result.exception.code, 0)
        self.assertEqual(commands[0][commands[0].index("--calcul") + 1], "cuda")
        options = sign.call_args_list[0].args[2]
        self.assertEqual(options[options.index("--calcul") + 1], "cuda")
        self.assertEqual(json.loads(options[options.index("--gpu-info") + 1]), device)

    def test_explicit_network_step_and_failed_network_are_handled(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as tmp:
                commands = []
                class FakeProcess:
                    stdout = []
                    def __init__(self, code):
                        self.code = code
                    def wait(self):
                        return self.code
                def launch(cmd, **kwargs):
                    commands.append(cmd)
                    return FakeProcess(exit_code if Path(cmd[1]).name.startswith("07_") else 0)
                with patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
                     patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
                     patch.object(sys, "argv", ["00_pipeline.py", "--etapes", "7", "10", "--continuer"]), \
                     self.assertRaises(SystemExit) as result:
                    pipeline.main()
                self.assertEqual(result.exception.code, exit_code)
                self.assertEqual("--sans-profond" in commands[1], exit_code != 0)

    def test_pipeline_passes_verified_cached_outputs_to_downstream_stages(self):
        commands = []
        class FakeProcess:
            stdout = []
            def wait(self):
                return 0
        def launch(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {
                1: [root / "01_donnees"],
                2: [root / "qc"],
                3: [root / f"{base}_run" for base in BASES],
                4: [root / "fusion"],
            }
            with patch.object(pipeline, "SCRIPT_DIR", root), \
                 patch.object(pipeline, "signature", side_effect=lambda e, *a: str(e)), \
                 patch.object(pipeline, "CachePipeline") as cache, \
                 patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
                 patch.object(sys, "argv", ["00_pipeline.py", "--etapes", "1", "2", "3", "4", "5", "6"]), \
                 self.assertRaises(SystemExit) as result:
                cache.return_value.trouver.side_effect = lambda step, key: paths[step]
                pipeline.main()
            self.assertEqual(result.exception.code, 0)
            self.assertEqual([Path(c[1]).name[:2] for c in commands], ["05", "06"])
            for base, path in zip(BASES, paths[3]):
                self.assertEqual(commands[0][commands[0].index(f"--{base}") + 1], str(path))
            self.assertEqual(commands[1][commands[1].index("--fusion") + 1], str(paths[4][0]))

    def test_pipeline_recalculer_bypasses_cache(self):
        class FakeProcess:
            stdout = []
            def wait(self):
                return 0
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
             patch.object(pipeline, "signature", return_value="key"), \
             patch.object(pipeline, "CachePipeline") as cache, \
             patch.object(pipeline.subprocess, "Popen", return_value=FakeProcess()) as launch, \
             patch.object(sys, "argv", ["00_pipeline.py", "--etapes", "1", "--recalculer"]), \
             self.assertRaises(SystemExit) as result:
            pipeline.main()
        self.assertEqual(result.exception.code, 0)
        launch.assert_called_once()
        cache.return_value.trouver.assert_not_called()
        cache.return_value.enregistrer.assert_called_once()

    def test_extraction_defaults_enable_all_features(self):
        calls = []
        def run(base, args, params):
            calls.append(params)
            return Path("output")
        with patch.object(sys, "argv", ["03_extraction_entropies.py", "depression"]), \
             patch.object(extract, "dernier_dossier", return_value=Path("processed")), \
             patch.object(extract, "run_dataset", side_effect=run):
            extract.main()
        self.assertTrue(calls[0]["caracteristiques_etendues"])
        self.assertTrue(calls[0]["multiechelle"])

    def test_per_class_recall(self):
        y = np.array([0, 0, 1, 1])
        probabilities = np.array([[.9, .1], [.2, .8], [.1, .9], [.7, .3]])
        self.assertEqual(synthesis.rappel_classe(y, probabilities, 0), .5)
        self.assertEqual(synthesis.rappel_classe(y, probabilities, 1), .5)
        self.assertTrue(np.isnan(synthesis.rappel_classe(y, probabilities, 2)))

    def test_synthesis_writes_per_class_metrics_and_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out, interpretation = root / "binary", root / "summary", root / "interpretation"
            source.mkdir()
            out.mkdir()
            interpretation.mkdir()
            rows = []
            for i in range(10):
                label = i % 2
                p = .8 if label else .2
                rows.append(dict(tache="binaire_depression", harmonisation="aucune", modele="regression_logistique",
                                 repetition=1, subject_id=f"s{i}", groupe=f"s{i}", classe=label,
                                 proba_temoin=1-p, proba_trouble=p))
            pd.DataFrame(rows).to_csv(source / "predictions_hors_pli.csv", index=False)
            pd.DataFrame([dict(base="depression", sous_ensemble="mesure=ApEn", exactitude_equilibree=.8)]).to_csv(
                source / "ablation.csv", index=False)
            pd.DataFrame([dict(tache="binaire_depression", classe="trouble", variable="F7|A4|ApEn",
                               shap_abs_moyen=.3)]).to_csv(interpretation / "shap_valeurs.csv", index=False)
            with patch.object(sys, "argv", ["10", "--binaire", str(source), "--bootstrap", "20", "--sans-profond",
                                            "--interpretabilite", str(interpretation)]), \
                 patch.object(synthesis, "dernier_dossier", return_value=None) as latest, \
                 patch.object(synthesis, "nouveau_dossier", return_value=out):
                synthesis.main()
            self.assertNotIn("profond", [call.args[0] for call in latest.call_args_list])
            result = pd.read_csv(out / "metriques_par_classe.csv")
            self.assertEqual(set(result.classe), {"temoin", "trouble"})
            np.testing.assert_allclose(result.rappel, 1)
            self.assertTrue(result[["ic95_rappel_bas", "ic95_rappel_haut"]].notna().all().all())
            self.assertTrue((out / "ablation.csv").exists())
            self.assertEqual(pd.read_csv(out / "shap_principales.csv").iloc[0].variable, "F7|A4|ApEn")

    @unittest.skipUnless(importlib.util.find_spec("EntropyHub"), "EntropyHub is optional")
    def test_energy_and_notebook_entropy_definitions(self):
        from EntropyHub import ApEn, PermEn

        signal = np.sin(np.arange(1280) * 0.11) + 0.2 * np.cos(np.arange(1280) * 0.37)
        p = dict(fenetre=10, chevauchement=0, duree_max=None, notch=None,
                 ondelette="db4", caracteristiques_etendues=True)
        with patch.object(extract, "calculer_mesures",
                          return_value=(1, dict.fromkeys(MEASURES, 0.0))):
            _, rows, n = extract.process_subject(("depression", "s", "s", 1,
                                                  signal[None, :], 128, ["F7"], p))
        coefs = extract.dwt_bands(extract.bandpass(signal[None, :], 128)[0])
        energy = np.array([np.sum(c ** 2) for c in coefs])
        self.assertEqual(n, 1)
        np.testing.assert_allclose([r["LogPower"] for r in rows], np.log10(energy + 1e-12))
        np.testing.assert_allclose([r["RelPower"] for r in rows], energy / energy.sum())
        coef = coefs[0]
        self.assertAlmostEqual(rows[0]["ApEn"], float(np.ravel(ApEn(coef, m=2, r=.2 * coef.std())[0])[-1]))
        self.assertAlmostEqual(rows[0]["PermEn"], float(np.ravel(PermEn(coef, m=4, tau=1)[1])[-1]))
        hist = np.histogram(coef, bins=8)[0]
        probs = hist[hist > 0] / hist.sum()
        self.assertAlmostEqual(rows[0]["Shannon_amplitude"], -np.sum(probs * np.log2(probs)) / 3)

    def test_classic_extraction_reports_missing_extended_columns(self):
        info = {"base": "depression", "data": pd.DataFrame(columns=["canal", *MEASURES])}
        with self.assertRaisesRegex(ValueError, "relancer 03"):
            agreger(info, ["F7"], MEASURES + MESURES_ETENDUES)

    def test_flat_window_has_undefined_relative_power(self):
        p = dict(fenetre=10, chevauchement=0, duree_max=None, notch=None,
                 ondelette="db4", caracteristiques_etendues=True)
        _, rows, _ = extract.process_subject(("depression", "s", "s", 1,
                                              np.zeros((1, 1280)), 128, ["F7"], p))
        self.assertTrue(all(np.isnan(row["RelPower"]) for row in rows))
        self.assertTrue(all(row["LogPower"] == -12 for row in rows))

    @unittest.skipUnless(importlib.util.find_spec("EntropyHub"), "EntropyHub is optional")
    def test_extract_writes_extended_columns_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "extraction"
            out.mkdir()
            signal = np.sin(np.arange(1280) * 0.11) + 0.2 * np.cos(np.arange(1280) * 0.37)
            processed = extract.pretraiter(signal[None, :], 128, ["F7"])
            np.savez(root / "depression_signals.npz", subject=processed)
            pd.DataFrame([dict(subject_id="subject", group_validation="subject", label=1,
                               statut="retenu", fs=128, signal_pretraite_sha256=hashlib.sha256(
                                   np.ascontiguousarray(processed, dtype=np.float64).tobytes()).hexdigest())
                          ]).to_csv(root / "depression_metadata.csv", index=False)
            (root / "depression_canaux.json").write_text(json.dumps({"canaux": ["F7"]}), encoding="utf-8")
            (root / "controle_qualite.json").write_text(json.dumps({
                "bases": ["depression"], "pretraitement": {"fs_cible": 128, "bandpass": [0.5, 45.0], "notch": None}
            }), encoding="utf-8")
            args = Namespace(donnees=root, limite=None, workers=1, large=False)
            p = dict(fenetre=10, chevauchement=0, duree_max=None, notch=None, ondelette="db4",
                     caracteristiques_etendues=True, multiechelle=True, pretraite=True,
                     entropyhub=False, m=2, r=0.2, n_fuzzy=2, c_disp=6)
            with patch.object(extract, "nouveau_dossier", return_value=out):
                extract.run_dataset("depression", args, p)
            result = pd.read_csv(out / "features_windows.csv")
            self.assertEqual(len(result), 5)
            self.assertTrue(result[MESURES_ETENDUES + PUISSANCES].notna().all().all())
            self.assertTrue(result.loc[result.band == BANDS[0], MULTIECHELLE].notna().all().all())
            self.assertTrue(result.loc[result.band != BANDS[0], MULTIECHELLE].isna().all().all())
            subject = pd.read_csv(out / "features_sujets.csv")
            np.testing.assert_allclose(subject.loc[subject.band == BANDS[0], MULTIECHELLE],
                                       result.loc[result.band == BANDS[0], MULTIECHELLE])
            params = json.loads((out / "parametres_extraction.json").read_text(encoding="utf-8"))
            self.assertEqual(params["mesures"], MEASURES + MESURES_ETENDUES + PUISSANCES)

    def test_extended_extractions_fuse_and_feed_both_classifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs = {}
            for base_index, base in enumerate(BASES):
                run = root / base
                run.mkdir()
                runs[base] = run
                records, rows = [], []
                for label in (0, 1):
                    for i in range(6):
                        sid = f"{base}_{label}_{i}"
                        records.append(dict(subject_id=sid, group_validation=sid, label=label, statut="extrait"))
                        for channel in ["F7", "F3", "F4", "F8"]:
                            for band in BANDS:
                                row = dict(dataset=base, subject_id=sid, group_validation=sid, label=label,
                                           channel=channel, band=band, window=0, sd_before_standardization=1.0)
                                row.update({name: label + i / 10 + base_index / 100 + j / 1000
                                            for j, name in enumerate(MEASURES + MESURES_ETENDUES + PUISSANCES)})
                                row.update({name: label + i / 10 + j / 1000 if band == BANDS[0] else np.nan
                                            for j, name in enumerate(MULTIECHELLE)})
                                rows.append(row)
                pd.DataFrame(records).to_csv(run / "audit_enregistrements.csv", index=False)
                pd.DataFrame(rows).to_csv(run / "features_windows.csv", index=False)

            info = charger_extraction(runs["depression"], "depression")
            classic, _ = agreger(info, ["F7"])
            extended, _ = agreger(info, ["F7"], MEASURES + MESURES_ETENDUES + PUISSANCES)
            self.assertEqual((classic.shape[1], extended.shape[1]), (25, 50))
            self.assertEqual(agreger_multiechelle(info, ["F7"]).shape[1], 20)

            fused = root / "fused"
            fused.mkdir()
            extraction_args = [arg for base in BASES for arg in (f"--{base}", str(runs[base]))]
            with patch.object(sys, "argv", ["04", *extraction_args, "--sans-diagnostic"]), \
                 patch.object(fusion, "nouveau_dossier", return_value=fused):
                fusion.main()
            self.assertEqual(pd.read_csv(fused / "entropies_etendues_sujets.csv").shape[1], 61)
            self.assertEqual(pd.read_csv(fused / "puissances_sujets.csv").shape[1], 41)

            binary_folds, multiclass_folds = [], []
            for feature_set, count in (("classiques", 100), ("etendues", 240), ("etendues_puissances", 280)):
                bout = root / f"binary_{feature_set}"
                bout.mkdir()
                selection = [] if feature_set == "etendues_puissances" else ["--caracteristiques", feature_set]
                argv = ["05", "--bases", "depression", "--depression", str(runs["depression"]),
                        "--modeles", "regression_logistique", "--sans-reference-amplitude",
                        "--plis", "3", "--repetitions", "1", *selection, "--sans-tuning", "--cpu"]
                with patch.object(sys, "argv", argv), patch.object(binary, "nouveau_dossier", return_value=bout):
                    binary.main()
                result = pd.read_csv(bout / "metriques_par_repetition.csv")
                self.assertEqual(int(result.iloc[0]["n_variables"]), count)
                bpreds = synthesis.lire_predictions(bout, "binaire")
                self.assertEqual(synthesis.colonnes_proba(bpreds), ["proba_temoin", "proba_trouble"])
                binary_folds.append(pd.read_csv(bout / "predictions_hors_pli.csv")
                                    .set_index("subject_id")["pli"].sort_index())

                mout = root / f"multiclass_{feature_set}"
                mout.mkdir()
                argv = ["06", "--fusion", str(fused), "--modeles", "regression_logistique",
                        "--harmonisations", "zscore", "--plis", "3", "--repetitions", "1",
                        "--sans-permutations", "--sans-lodo", "--sans-tuning", "--cpu", *selection]
                with patch.object(sys, "argv", argv), patch.object(multiclass, "nouveau_dossier", return_value=mout):
                    multiclass.main()
                result = pd.read_csv(mout / "multiclasse_metriques.csv")
                selected = result[(result.modele == "regression_logistique") & (result.harmonisation == "aucune")]
                self.assertEqual(int(selected.iloc[0]["nombre_variables"]), count)
                self.assertTrue((mout / "controle_site_resume.csv").exists())
                predictions = pd.read_csv(mout / "predictions_hors_pli.csv")
                mpreds = synthesis.lire_predictions(mout, "multiclasse")
                self.assertEqual(len(synthesis.colonnes_proba(mpreds)), 4)
                multiclass_folds.append(predictions[(predictions.modele == "regression_logistique") &
                                                     (predictions.harmonisation == "aucune")]
                                        .set_index("subject_id")["pli"].sort_index())
                params = json.loads((mout / "controle_multiclasse.json").read_text(encoding="utf-8"))
                self.assertEqual(params["arguments"]["caracteristiques"], feature_set)
            for folds in binary_folds[1:]:
                pd.testing.assert_series_equal(binary_folds[0], folds)
            for folds in multiclass_folds[1:]:
                pd.testing.assert_series_equal(multiclass_folds[0], folds)

            for run in runs.values():
                path = run / "features_windows.csv"
                pd.read_csv(path).drop(columns=MESURES_ETENDUES + PUISSANCES + MULTIECHELLE).to_csv(path, index=False)
            legacy = root / "legacy"
            legacy.mkdir()
            with patch.object(sys, "argv", ["04", *extraction_args, "--sans-diagnostic",
                                            "--sans-caracteristiques-etendues"]), \
                 patch.object(fusion, "nouveau_dossier", return_value=legacy):
                fusion.main()
            self.assertTrue((legacy / "entropies_sujets.csv").exists())
            self.assertFalse((legacy / "puissances_sujets.csv").exists())


if __name__ == "__main__":
    unittest.main()
