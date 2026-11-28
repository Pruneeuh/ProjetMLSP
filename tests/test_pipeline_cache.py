import json
import tempfile
import unittest
from pathlib import Path

from eeg_cache_pipeline import CachePipeline, signature, verifier_sorties
from eeg_config import BASES


ROOT = Path(__file__).resolve().parents[1]


class PipelineCacheTests(unittest.TestCase):
    def test_reuse_requires_matching_inputs_and_intact_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "recording.edf").write_bytes(b"input")
            output = root / "output"
            output.mkdir()
            for base in BASES:
                for suffix in ("signals.npz", "metadata.csv", "canaux.json"):
                    (output / f"{base}_{suffix}").write_bytes(b"output")
            for name in ("inventaire_chargement.json", "inventaire_chargement.csv"):
                (output / name).write_bytes(b"inventory")
            (output / "inventaire_chargement.json").write_text(
                json.dumps({"resume": {base: {"retenus": 1} for base in BASES}}), encoding="utf-8")
            cache = CachePipeline(root / "cache.json")
            key = signature(1, [source], [], ROOT)
            cache.enregistrer(1, key, [output])
            self.assertEqual(CachePipeline(cache.path).trouver(1, key), [output])

            (source / "recording.edf").write_bytes(b"changed")
            changed_key = signature(1, [source], [], ROOT)
            self.assertNotEqual(key, changed_key)
            self.assertIsNone(cache.trouver(1, changed_key))
            self.assertNotEqual(changed_key, signature(1, [source], ["--different"], ROOT))
            (output / "depression_signals.npz").write_bytes(b"changed output")
            self.assertIsNone(cache.trouver(1, key))
            (output / "depression_signals.npz").unlink()
            self.assertIsNone(cache.trouver(1, key))

    def test_incomplete_inventory_cannot_be_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for base in BASES:
                for suffix in ("signals.npz", "metadata.csv", "canaux.json"):
                    (root / f"{base}_{suffix}").write_bytes(b"old")
            (root / "inventaire_chargement.csv").write_bytes(b"inventory")
            (root / "inventaire_chargement.json").write_text(
                json.dumps({"resume": {"depression": {"retenus": 1}}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventaire"):
                CachePipeline(root / "cache.json").enregistrer(1, "key", [root])

    def test_partial_outputs_are_never_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = CachePipeline(root / "cache.json")
            with self.assertRaises(ValueError):
                cache.enregistrer(2, "key", [root])
            self.assertFalse(cache.path.exists())

    def test_participant_only_fusion_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("entropies_sujets.csv", "amplitude_sujets.csv", "meta_sujets.csv",
                         "entropies_etendues_sujets.csv", "puissances_sujets.csv",
                         "entropies_multiechelle_sujets.csv", "fusion.json"):
                (root / name).write_text("legacy", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fenetres_communes.csv"):
                verifier_sorties(4, [root])


if __name__ == "__main__":
    unittest.main()
