import ast
import importlib.util
import io
import json
import os
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


path = Path(__file__).resolve().parents[1] / "00_package_installs.py"
spec = importlib.util.spec_from_file_location("package_installer", path)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

notebook_path = path.with_name("00_notebook_install.py")
notebook_spec = importlib.util.spec_from_file_location("notebook_installer", notebook_path)
notebook_installer = importlib.util.module_from_spec(notebook_spec)
notebook_spec.loader.exec_module(notebook_installer)


class PackageInstallerTests(unittest.TestCase):
    def test_notebook_can_run_pipeline_after_kernel_restart_without_reinstalling(self):
        notebook = json.loads(path.with_name("00_pipeline_notebook.ipynb").read_text(encoding="utf-8"))
        first = "".join(notebook["cells"][1]["source"])
        pipeline = "".join(notebook["cells"][5]["source"])
        commands = []

        class FakeProcess:
            stdout = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def wait(self):
                return 0

        def launch(command, **kwargs):
            commands.append(command)
            return FakeProcess()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("requirement.txt", "00_pipeline.py", "00_notebook_install.py"):
                (root / name).touch()
            with patch.object(Path, "cwd", return_value=root), \
                 patch.object(subprocess, "Popen", side_effect=launch):
                scope = {}
                exec(first, scope)
                exec(pipeline, scope)
        self.assertEqual(len(commands), 1)
        self.assertEqual(Path(commands[0][2]).name, "00_pipeline.py")

    def test_notebook_installs_converted_wheel_and_checks_current_environment(self):
        py = f"cp{notebook_installer.sys.version_info.major}{notebook_installer.sys.version_info.minor}"
        release = {"urls": [{"filename": f"cupy_cuda12x-14.2.0-{py}-{py}-manylinux2014_x86_64.whl",
                             "url": "https://example.test/cupy.whl", "yanked": False}]}
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[:2] == ["bash", "-e"]:
                wheel = Path(kwargs["cwd"]) / f"cupy_cuda12x-14.2.0+computecanada-{py}-{py}-linux_x86_64.whl"
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr("cupy/__init__.py", "test")
                    archive.writestr("cupy/_core/core.so", "test")

        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "requirement.txt"
            requirements.write_text("cupy-cuda12x==14.2.0\n", encoding="utf-8")
            with patch.object(notebook_installer.sys, "platform", "linux"), \
                 patch.object(notebook_installer.sys, "prefix", "system"), \
                 patch.object(notebook_installer.sys, "base_prefix", "system"), \
                 patch.object(notebook_installer.platform, "machine", return_value="x86_64"), \
                 patch.object(notebook_installer.shutil, "which", return_value="available"), \
                 patch.object(notebook_installer.urllib.request, "urlopen",
                              return_value=io.BytesIO(json.dumps(release).encode())), \
                 patch.object(notebook_installer.subprocess, "run", side_effect=run):
                notebook_installer.install(requirements)
        convert = next(command for command in calls if command[:2] == ["bash", "-e"])
        self.assertIn("https://example.test/cupy.whl", convert)
        install_calls = [c for c in calls if "install" in c and "pip" in c]
        self.assertEqual(len(install_calls), 2)
        self.assertTrue(all("--user" in command for command in install_calls))
        self.assertIn("--no-deps", install_calls[0])
        self.assertIn("--find-links", install_calls[1])
        self.assertIn("-r", install_calls[1])
        self.assertIn(["pip", "check"], [c[-2:] for c in calls])
        self.assertTrue(any("import cupy" in c[-1] for c in calls))

    def test_notebook_rejects_empty_converted_wheel(self):
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "empty.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("cupy-14.2.0.dist-info/METADATA", "")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                notebook_installer.check_wheel(wheel)

    def test_notebook_does_not_install_if_conversion_fails(self):
        calls = []

        def run(command, **kwargs):
            calls.append(command)
            if command[:2] == ["bash", "-e"]:
                raise subprocess.CalledProcessError(1, command)

        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "requirement.txt"
            requirements.write_text("cupy-cuda12x==14.2.0\n", encoding="utf-8")
            with patch.object(notebook_installer, "cupy_wheel_url", return_value="https://example.test/cupy.whl"), \
                 patch.object(notebook_installer.shutil, "which", return_value="available"), \
                 patch.object(notebook_installer.subprocess, "run", side_effect=run), \
                 self.assertRaises(subprocess.CalledProcessError):
                notebook_installer.install(requirements)
        self.assertFalse(any("pip" in command for command in calls))

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
