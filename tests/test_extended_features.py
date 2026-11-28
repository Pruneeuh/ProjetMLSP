import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_config import BANDS, BASES, MEASURES, MESURES_ETENDUES, PUISSANCES
from eeg_donnees import agreger, charger_extraction


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


class ExtendedFeatureTests(unittest.TestCase):
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
            np.savez(root / "depression_signals.npz", subject=signal[None, :])
            pd.DataFrame([dict(subject_id="subject", group_validation="subject", label=1,
                               statut="retenu", fs=128)]).to_csv(root / "depression_metadata.csv", index=False)
            (root / "depression_canaux.json").write_text(json.dumps({"canaux": ["F7"]}), encoding="utf-8")
            args = Namespace(donnees=root, limite=None, workers=1, large=False)
            p = dict(fenetre=10, chevauchement=0, duree_max=None, notch=None, ondelette="db4",
                     caracteristiques_etendues=True, entropyhub=False, m=2, r=0.2, n_fuzzy=2, c_disp=6)
            with patch.object(extract, "nouveau_dossier", return_value=out):
                extract.run_dataset("depression", args, p)
            result = pd.read_csv(out / "features_windows.csv")
            self.assertEqual(len(result), 5)
            self.assertTrue(result[MESURES_ETENDUES + PUISSANCES].notna().all().all())
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
                                rows.append(row)
                pd.DataFrame(records).to_csv(run / "audit_enregistrements.csv", index=False)
                pd.DataFrame(rows).to_csv(run / "features_windows.csv", index=False)

            info = charger_extraction(runs["depression"], "depression")
            classic, _ = agreger(info, ["F7"])
            extended, _ = agreger(info, ["F7"], MEASURES + MESURES_ETENDUES + PUISSANCES)
            self.assertEqual((classic.shape[1], extended.shape[1]), (25, 50))

            fused = root / "fused"
            fused.mkdir()
            extraction_args = [arg for base in BASES for arg in (f"--{base}", str(runs[base]))]
            with patch.object(sys, "argv", ["04", *extraction_args, "--sans-diagnostic",
                                            "--caracteristiques-etendues"]), \
                 patch.object(fusion, "nouveau_dossier", return_value=fused):
                fusion.main()
            self.assertEqual(pd.read_csv(fused / "entropies_etendues_sujets.csv").shape[1], 61)
            self.assertEqual(pd.read_csv(fused / "puissances_sujets.csv").shape[1], 41)

            binary_folds, multiclass_folds = [], []
            for feature_set, count in (("classiques", 100), ("etendues", 160), ("etendues_puissances", 200)):
                bout = root / f"binary_{feature_set}"
                bout.mkdir()
                argv = ["05", "--bases", "depression", "--depression", str(runs["depression"]),
                        "--modeles", "regression_logistique", "--sans-reference-amplitude",
                        "--plis", "3", "--repetitions", "1", "--caracteristiques", feature_set,
                        "--sans-tuning", "--cpu"]
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
                        "--sans-permutations", "--sans-lodo", "--sans-tuning", "--cpu",
                        "--caracteristiques", feature_set]
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


if __name__ == "__main__":
    unittest.main()
