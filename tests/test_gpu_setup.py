import contextlib
import io
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lightgbm.basic import LightGBMError

import gpu_setup
import eeg_gpu_entropies
import model_tuning


class GPUSetupTests(unittest.TestCase):
    def test_entropy_gpu_unavailable_is_reported(self):
        with patch('eeg_gpu_entropies.importlib.util.find_spec', return_value=None), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(eeg_gpu_entropies.detecter_cuda(), ('cpu', None))
        self.assertIn('CPU', output.getvalue())

    def test_entropy_gpu_probe_records_device_and_versions(self):
        import numpy as np

        runtime = SimpleNamespace(getDeviceCount=lambda: 1, getDeviceProperties=lambda _: {'name': b'test GPU'},
                                  driverGetVersion=lambda: 123, runtimeGetVersion=lambda: 456,
                                  CUDARuntimeError=RuntimeError)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime), float64=np.float64,
                               asarray=np.asarray, exp=np.exp, __version__='1.0')
        with patch('eeg_gpu_entropies.importlib.util.find_spec', return_value=object()), \
             patch.dict(sys.modules, {'cupy': cupy}), contextlib.redirect_stdout(io.StringIO()):
            backend, info = eeg_gpu_entropies.detecter_cuda()
        self.assertEqual(backend, 'cuda')
        self.assertEqual(info, {'device': 'test GPU', 'cupy': '1.0', 'driver': 123, 'runtime': 456})

    def test_pipeline_scripts_enable_rapids_before_loading_models(self):
        root = Path(__file__).resolve().parents[1]
        bootstrap = """
import runpy, sys, types
module = types.ModuleType("gpu_setup")
def enable():
    if "sklearn" in sys.modules:
        raise RuntimeError("sklearn was imported before RAPIDS")
    print("RAPIDS startup checked")
    return False
module.enable_cuml_accel = enable
sys.modules["gpu_setup"] = module
script = sys.argv[1]
sys.argv = [script, "--rapids", "--help"]
runpy.run_path(script, run_name="__main__")
"""
        for name in ('05_classification_binaire.py', '06_classification_multiclasse.py', '09_interpretabilite.py'):
            with self.subTest(script=name):
                result = subprocess.run([sys.executable, '-c', bootstrap, str(root / name)], cwd=root,
                                        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                                        capture_output=True, text=True, encoding='utf-8')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('RAPIDS startup checked', result.stdout)

    def test_missing_rapids_is_reported(self):
        with patch('gpu_setup.importlib.util.find_spec', return_value=None):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertFalse(gpu_setup.enable_cuml_accel())
            self.assertIn('not installed', output.getvalue())

    def test_rapids_enabled_before_sklearn(self):
        runtime = SimpleNamespace(getDeviceCount=Mock(return_value=1), CUDARuntimeError=RuntimeError)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
        install = Mock()
        cuml = SimpleNamespace(accel=SimpleNamespace(install=install))
        with patch('gpu_setup.importlib.util.find_spec', return_value=object()), \
             patch('gpu_setup.sys', SimpleNamespace(modules={})), \
             patch.dict(sys.modules, {'cupy': cupy, 'cuml': cuml}), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(gpu_setup.enable_cuml_accel())
        install.assert_called_once_with(log_level='info')

    def test_rapids_falls_back_when_no_cuda_device_is_visible(self):
        runtime = SimpleNamespace(getDeviceCount=Mock(return_value=0), CUDARuntimeError=RuntimeError)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
        with patch('gpu_setup.importlib.util.find_spec', return_value=object()), \
             patch.dict(sys.modules, {'cupy': cupy}), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(gpu_setup.enable_cuml_accel())
        self.assertIn('No CUDA device', output.getvalue())

    def test_rapids_requires_kernel_restart_after_sklearn_import(self):
        runtime = SimpleNamespace(getDeviceCount=Mock(return_value=1), CUDARuntimeError=RuntimeError)
        cupy = SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))
        with patch('gpu_setup.importlib.util.find_spec', return_value=object()), \
             patch('gpu_setup.sys', SimpleNamespace(modules={'sklearn': object()})), \
             patch.dict(sys.modules, {'cupy': cupy}):
            with self.assertRaisesRegex(RuntimeError, 'Restart the notebook kernel'):
                gpu_setup.enable_cuml_accel()

    def test_lightgbm_tries_cuda_then_opencl(self):
        with patch('model_tuning.LGBMClassifier') as classifier:
            classifier.return_value.fit.side_effect = [LightGBMError('no CUDA'), None]
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(model_tuning.lightgbm_device(), 'gpu')
            self.assertIn('no CUDA', output.getvalue())
            self.assertEqual([call.kwargs['device_type'] for call in classifier.call_args_list], ['cuda', 'gpu'])

    def test_lightgbm_reports_cpu_fallback(self):
        with patch('model_tuning.LGBMClassifier') as classifier:
            classifier.return_value.fit.side_effect = LightGBMError('no GPU')
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(model_tuning.lightgbm_device(), 'cpu')
            self.assertIn('using CPU', output.getvalue())

    def test_gpu_lightgbm_search_is_serial_and_forest_is_unweighted(self):
        models = model_tuning.build_models(lgbm_device='cuda')
        self.assertEqual(model_tuning.search_n_jobs('lightgbm', models['lightgbm'][0], 4), 1)
        forest, space = models['random_forest']
        self.assertIsNone(forest.get_params()['model__class_weight'])
        self.assertNotIn('model__class_weight', space)

    def test_cpu_lightgbm_trains(self):
        import numpy as np
        import pandas as pd

        y = np.tile([0, 1], 24)
        table = pd.DataFrame({'subject_id': np.arange(len(y)), 'label': y, 'feature': y})
        pipeline, _ = model_tuning.build_models(lgbm_device='cpu')['lightgbm']
        models = {'lightgbm': (pipeline, {'select__percentile': [100], 'model__n_estimators': [2],
                                          'model__min_child_samples': [2]})}
        profile = {'n_iter': 1, 'n_iter_trees': 1, 'outer_folds': 2, 'inner_folds': 2, 'repeats': 1}
        metrics, _, predictions, tuning = model_tuning.evaluate_dataset(
            table, ['feature'], 'task', 'test', profile=profile, n_jobs=1, models=models)
        self.assertEqual(len(metrics), 1)
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(tuning['inner_roc_auc'].notna().all())


if __name__ == '__main__':
    unittest.main()
