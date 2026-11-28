import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from eeg_config import MEASURES
from eeg_pretraitement import pretraiter

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


qc = load_script("02_controle_qualite")
extract = load_script("03_extraction_entropies")
pipeline = load_script("00_pipeline")


class PreprocessingTests(unittest.TestCase):
    def prepare_source(self, root, invalid=False):
        x = np.sin(np.arange(2560) * 2 * np.pi * 10 / 256)[None, :]
        if invalid:
            x[0, 3] = np.nan
        np.savez(root / "depression_signals.npz", a=x, b=x * 2)
        pd.DataFrame([
            dict(subject_id=sid, group_validation=sid, label=label, statut="retenu", fs=256,
                 n_samples=2560, duree_s=10, n_canaux_origine=19,
                 signal_sha256=hashlib.sha256(np.ascontiguousarray(
                     x if sid == "a" else x * 2, dtype=np.float64).tobytes()).hexdigest())
            for sid, label in (("a", 0), ("b", 1))
        ] + [dict(subject_id="excluded", group_validation="excluded", label=1, statut="exclu", fs=256,
                  n_samples=2560, duree_s=10, n_canaux_origine=19)]).to_csv(
                      root / "depression_metadata.csv", index=False)
        (root / "depression_canaux.json").write_text(json.dumps({
            "canaux": ["F7"], "ordre_suppose": False,
        }), encoding="utf-8")
        return x

    def test_preprocess_is_saved_once_and_used_for_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "qc"
            out.mkdir()
            x = self.prepare_source(root)
            with patch.object(sys, "argv", ["02", "depression", "--donnees", str(root)]), \
                 patch.object(qc, "nouveau_dossier", return_value=out):
                qc.main()
            self.assertTrue((out / "controle_qualite.json").is_file())
            with np.load(out / "depression_signals.npz") as archive:
                processed = archive["a"]
                self.assertEqual(set(archive.files), {"a", "b"})
            np.testing.assert_allclose(processed, pretraiter(x, 256, ["F7"]))
            self.assertEqual(processed.shape, (1, 1280))
            meta = pd.read_csv(out / "depression_metadata.csv").set_index("subject_id")
            self.assertEqual(meta.loc["a", "fs_source"], 256)
            self.assertEqual(meta.loc["a", "fs"], 128)
            self.assertEqual(meta.loc["excluded", "statut"], "exclu")
            self.assertTrue(meta.loc["a", "signal_pretraite_sha256"])
            rows = pd.read_csv(out / "qc_enregistrements.csv")
            self.assertTrue((rows["fs_pretraite"] == 128).all())

            captured = []
            def dwt(signal, *args):
                captured.append(signal.copy())
                return [signal] * 5
            params = dict(pretraite=True, fenetre=10, chevauchement=0, duree_max=None,
                          ondelette="db4", caracteristiques_etendues=False)
            with patch.object(extract, "dwt_bands", side_effect=dwt), \
                 patch.object(extract, "calculer_mesures", return_value=(1, dict.fromkeys(MEASURES, 0.0))):
                extract.process_subject(("depression", "a", "a", 0, processed, 128, ["F7"], params))
            np.testing.assert_array_equal(captured[0], processed[0])
            meta = pd.read_csv(out / "depression_metadata.csv")
            meta.loc[0, "signal_pretraite_sha256"] = "not-the-hash"
            meta.to_csv(out / "depression_metadata.csv", index=False)
            args = type("Args", (), {"donnees": out, "limite": None})()
            with self.assertRaisesRegex(ValueError, "empreinte prétraitée incohérente"):
                extract.run_dataset("depression", args, {})

    def test_nonfinite_input_stops_without_completion_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "qc"
            out.mkdir()
            self.prepare_source(root, invalid=True)
            with patch.object(sys, "argv", ["02", "depression", "--donnees", str(root)]), \
                 patch.object(qc, "nouveau_dossier", return_value=out), \
                 self.assertRaises(SystemExit) as result:
                qc.main()
            self.assertEqual(result.exception.code, 1)
            self.assertFalse((out / "controle_qualite.json").exists())

    def test_extraction_rejects_missing_preprocessing_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = type("Args", (), {"donnees": Path(tmp)})()
            with self.assertRaisesRegex(ValueError, "Prétraitement 02 incomplet"):
                extract.run_dataset("depression", args, {})

    def test_hash_mismatch_stops_qc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "qc"
            out.mkdir()
            self.prepare_source(root)
            metadata = pd.read_csv(root / "depression_metadata.csv")
            metadata.loc[0, "signal_sha256"] = "not-the-hash"
            metadata.to_csv(root / "depression_metadata.csv", index=False)
            with patch.object(sys, "argv", ["02", "depression", "--donnees", str(root)]), \
                 patch.object(qc, "nouveau_dossier", return_value=out), \
                 self.assertRaises(SystemExit) as result:
                qc.main()
            self.assertEqual(result.exception.code, 1)
            self.assertFalse((out / "controle_qualite.json").exists())

    def test_brief_large_peak_is_reported_without_excluding_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "qc"
            out.mkdir()
            x = self.prepare_source(root)
            x[0, 1280] += 1000
            np.savez(root / "depression_signals.npz", a=x, b=x * 2)
            meta = pd.read_csv(root / "depression_metadata.csv")
            for idx, signal in ((0, x), (1, x * 2)):
                meta.loc[idx, "signal_sha256"] = hashlib.sha256(
                    np.ascontiguousarray(signal, dtype=np.float64).tobytes()).hexdigest()
            meta.to_csv(root / "depression_metadata.csv", index=False)
            with patch.object(sys, "argv", ["02", "depression", "--donnees", str(root)]), \
                 patch.object(qc, "nouveau_dossier", return_value=out):
                qc.main()
            records = pd.read_csv(out / "qc_enregistrements.csv").set_index("subject_id")
            self.assertLess(records.loc["a", "part_saturee_max"], 0.01)
            self.assertEqual(records.loc["a", "canaux_pic_eleve"], 1)
            alerts = pd.read_csv(out / "qc_alertes.csv")
            self.assertTrue(alerts.loc[alerts.subject_id == "a", "alerte"].str.contains("pic d'amplitude").any())
            output = pd.read_csv(out / "depression_metadata.csv").set_index("subject_id")
            self.assertEqual(output.loc["a", "statut"], "retenu")

    def test_filter_preserves_alpha_and_suppresses_mains(self):
        fs = 256
        t = np.arange(2560) / fs
        original = (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 60 * t))[None, :]
        output = pretraiter(original, fs, ["F7"])[0]
        t_out = np.arange(len(output)) / 128
        middle = slice(128, -128)
        amplitude = lambda f: 2 * abs(np.mean(output[middle] * np.exp(-2j * np.pi * f * t_out[middle])))
        self.assertGreater(amplitude(10), 0.9)
        self.assertLess(amplitude(60), 0.05)

    def test_pipeline_stops_on_qc_failure_even_with_continue(self):
        commands = []
        class FakeProcess:
            stdout = []
            def __init__(self, code):
                self.code = code
            def wait(self):
                return self.code
        def launch(cmd, **kwargs):
            commands.append(cmd)
            return FakeProcess(1 if Path(cmd[1]).name.startswith("02_") else 0)
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(pipeline, "SCRIPT_DIR", Path(tmp)), \
             patch.object(pipeline, "signature", return_value="key"), \
             patch.object(pipeline, "CachePipeline") as cache, \
             patch.object(pipeline.subprocess, "Popen", side_effect=launch), \
             patch.object(sys, "argv", ["00", "--etapes", "1", "2", "3", "--continuer"]), \
             self.assertRaises(SystemExit) as result:
            cache.return_value.trouver.return_value = None
            pipeline.main()
        self.assertEqual(result.exception.code, 1)
        self.assertEqual(len(commands), 2)


if __name__ == "__main__":
    unittest.main()
