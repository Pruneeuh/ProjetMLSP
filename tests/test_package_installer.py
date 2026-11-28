import ast
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "00_package_installs.py"
spec = importlib.util.spec_from_file_location("package_installer", path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class PackageInstallerTests(unittest.TestCase):
    def test_isolated_install_uses_wheelhouse_and_checks_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "new-environment"
            with patch.dict(os.environ, {"PYTHONPATH": "old-stack", "PYTHONHOME": "old-home"}), \
                 patch.object(installer.subprocess, "run") as run:
                self.assertEqual(installer.install_environment(target), target)
            self.assertEqual(run.call_count, 4)
            create, install, check, smoke = run.call_args_list
            self.assertEqual(create.args[0][1:3], ["-m", "venv"])
            self.assertIn("--no-index", install.args[0])
            self.assertIn("--only-binary=:all:", install.args[0])
            self.assertNotIn("--user", install.args[0])
            self.assertEqual(check.args[0][-2:], ["pip", "check"])
            self.assertEqual(smoke.args[0][-2], "-c")
            ast.parse(smoke.args[0][-1])
            for call in run.call_args_list:
                self.assertTrue(call.kwargs["check"])
                self.assertNotIn("PYTHONPATH", call.kwargs["env"])
                self.assertNotIn("PYTHONHOME", call.kwargs["env"])
                self.assertEqual(call.kwargs["env"]["PYTHONNOUSERSITE"], "1")

    def test_existing_directory_is_never_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(installer.subprocess, "run") as run:
                with self.assertRaises(FileExistsError):
                    installer.install_environment(Path(tmp))
            run.assert_not_called()

    def test_pip_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(installer.subprocess, "run",
                              side_effect=[None, subprocess.CalledProcessError(1, "pip")]) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    installer.install_environment(Path(tmp) / "env")
            self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
