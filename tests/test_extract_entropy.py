import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import extract_entropy as extraction


class CombinedExtractionTests(unittest.TestCase):
    def test_dataset_run_writes_complete_single_file(self):
        signal = np.random.default_rng(42).normal(size=(1, 256))
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(extraction, "OUTPUT_DIR", Path(tmp)), \
             patch.object(extraction, "load_saved_dataset",
                          return_value=([signal], np.array([1]), ["subject"])):
            extraction.run_dataset("sakib", 2, 0, "db4", 1)
            table = extraction.load_window_features("sakib")
            self.assertEqual(len(table), 1)
            self.assertEqual(len(table.columns), 3 + 5 * 9)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["sakib_entropy_features.csv"])

    def test_all_features_share_one_decomposition(self):
        coefs = [np.sin(np.arange(64) * (i + 1) / 10) for i in range(5)]
        task = ("subject", 1, np.zeros((1, 128)), 128, 1, 0, "db4")
        with patch.object(extraction, "bandpass", side_effect=lambda sig, fs: sig), \
             patch.object(extraction, "dwt_bands", return_value=coefs) as dwt:
            row, = extraction.process_subject(task)
        dwt.assert_called_once()
        self.assertEqual(len(row), 3 + 5 * 9)
        energies = np.array([np.sum(coef ** 2) for coef in coefs])
        for band, coef, energy in zip(extraction.BANDS, coefs, energies):
            prefix = f"ch01_{band}"
            for name, fn in extraction.ENTROPIES.items():
                np.testing.assert_allclose(row[f"{prefix}_{name}"], fn(coef), equal_nan=True)
            self.assertAlmostEqual(row[f"{prefix}_LogPower"], np.log10(energy + 1e-12))
            self.assertAlmostEqual(row[f"{prefix}_RelPower"], energy / energies.sum())

    def test_zero_energy_relative_power_is_missing(self):
        task = ("subject", 0, np.zeros((1, 128)), 128, 1, 0, "db4")
        with patch.object(extraction, "bandpass", side_effect=lambda sig, fs: sig), \
             patch.object(extraction, "dwt_bands", return_value=[np.zeros(64)] * 5), \
             patch.object(extraction, "ENTROPIES", {}):
            row, = extraction.process_subject(task)
        for band in extraction.BANDS:
            self.assertTrue(np.isnan(row[f"ch01_{band}_RelPower"]))
            self.assertEqual(row[f"ch01_{band}_LogPower"], -12)


class CachedFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.output_patch = patch.object(extraction, "OUTPUT_DIR", self.root)
        self.output_patch.start()
        self.addCleanup(self.output_patch.stop)
        self.base = self.root / "sakib_entropy_features.csv"
        self.extra = self.root / "sakib_extra_features.csv"
        self.keys = {"subject_id": ["s1", "s2"], "label": [0, 1], "window": [0, 0]}
        self.original = pd.DataFrame({
            **self.keys,
            **{f"ch01_{band}_{name}": [1.0, 2.0] for band in extraction.BANDS
               for name in ("SampEn", "DispEn", "FuzzyEn", "Shannon")},
        })
        self.additional = pd.DataFrame({
            **self.keys,
            **{f"ch01_{band}_{name}": [3.0, 4.0] for band in extraction.BANDS
               for name in ("ApEn", "PermEn", "LZC", "LogPower", "RelPower")},
        })

    def test_legacy_pair_is_joined_and_combined_cache_ignores_extra(self):
        self.original.to_csv(self.base, index=False)
        self.additional.to_csv(self.extra, index=False)
        combined = extraction.load_window_features("sakib")
        self.assertEqual(len(combined.columns), 3 + 5 * 9)
        self.assertEqual(combined["ch01_delta_ApEn"].tolist(), [3.0, 4.0])
        combined.to_csv(self.base, index=False)
        self.extra.write_text("invalid", encoding="utf-8")
        pd.testing.assert_frame_equal(extraction.load_window_features("sakib"), combined)

    def test_incomplete_or_mismatched_caches_fail(self):
        self.original.to_csv(self.base, index=False)
        with self.assertRaisesRegex(ValueError, "incomplete legacy cache"):
            extraction.load_window_features("sakib")
        self.additional.assign(label=[1, 1]).to_csv(self.extra, index=False)
        with self.assertRaisesRegex(ValueError, "windows or labels do not match"):
            extraction.load_window_features("sakib")
        self.additional.drop(columns="ch01_alpha_PermEn").to_csv(self.extra, index=False)
        with self.assertRaisesRegex(ValueError, "missing feature columns"):
            extraction.load_window_features("sakib")

    def test_duplicate_windows_are_rejected(self):
        combined = self.original.merge(self.additional, on=list(self.keys))
        pd.concat([combined, combined.iloc[[0]]]).to_csv(self.base, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate subject/window"):
            extraction.load_window_features("sakib")


if __name__ == "__main__":
    unittest.main()
