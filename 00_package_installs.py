"""Create an isolated Valeria Python environment using the configured wheelhouse.

Load the same Valeria modules each time the environment is used. This installer
does not modify an existing environment or register a Jupyter kernel.

    module load StdEnv/2023 python/3.12.4 scipy-stack/2024a
    python 00_package_installs.py
    python 00_package_installs.py --env ~/venvs/projetmlsp-autre
"""
import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


# Versions observed in the working Valeria notebook; local +computecanada
# wheel versions also satisfy these public version pins.
PACKAGES = [
    "numpy==2.2.2",
    "scipy==1.18.1",
    "matplotlib==3.11.1",
    "pandas==3.0.5",
    "scikit-learn==1.7.1",
    "xgboost==3.2.0",
    "lightgbm==4.5.0",
    "shap==0.51.0",
    "PyWavelets==1.7.0",
    "mne==1.12.1",
    "EntropyHub==2.0",
    "joblib",
    "torch",
    "jinja2",
    "ipykernel>=6",
    "ipywidgets",
]

SMOKE_TEST = """
import torch
import numpy as np
import scipy.spatial
import pandas
import matplotlib.pyplot
import pywt
import mne
import EntropyHub
import joblib
import jinja2
import ipykernel
import ipywidgets
import sklearn
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import shap
from importlib.metadata import version

x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
y = np.array([0, 0, 1, 1])
torch.from_numpy(x)
LGBMClassifier(n_estimators=2, min_child_samples=1, n_jobs=1, verbosity=-1).fit(x, y)
model = XGBClassifier(n_estimators=2, max_depth=1, n_jobs=1, tree_method="hist",
                      device="cpu").fit(x, y)
values = np.asarray(shap.TreeExplainer(model).shap_values(x, check_additivity=False))
if values.shape != x.shape or not np.isfinite(values).all():
    raise RuntimeError(f"Unexpected XGBoost SHAP values: {values.shape}")
for name in ("numpy", "scipy", "matplotlib", "pandas", "scikit-learn", "xgboost",
             "lightgbm", "shap", "PyWavelets", "mne", "EntropyHub", "torch", "ipykernel"):
    print(f"{name}: {version(name)}", flush=True)
print("CPU import, model training and XGBoost SHAP checks passed.", flush=True)
"""


def install_environment(path):
    """Create a new environment; fail without touching an existing path."""
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing path: {path}")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["OMP_NUM_THREADS"] = "1"

    subprocess.run([sys.executable, "-m", "venv", str(path)], env=env, check=True)
    python = path / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--no-index",
                    "--only-binary=:all:", *PACKAGES], env=env, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    subprocess.run([str(python), "-c", SMOKE_TEST], env=env, check=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", type=Path, default=Path.home() / "venvs" / "projetmlsp-valeria",
                        help="New environment path; an existing path is never overwritten.")
    args = parser.parse_args()
    if sys.platform != "linux" or sys.version_info[:2] != (3, 12):
        parser.exit(1, "Use Valeria Linux with the python/3.12.4 module loaded.\n")
    try:
        path = install_environment(args.env)
    except (OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Installation failed: {exc}\n"
                    "No fallback to PyPI was attempted. Check the wheelhouse and loaded modules. "
                    "If an incomplete environment was created, choose a new --env path.\n")

    quoted = shlex.quote(str(path))
    print(f"\nEnvironment ready: {path}")
    print("For each terminal session, reload the same Valeria modules, then run:")
    print(f"  source {quoted}/bin/activate")
    print("  unset PYTHONPATH PYTHONHOME")
    print("  export PYTHONNOUSERSITE=1")
    print("To add a Jupyter kernel, choose a unique name and run:")
    print('  python -m ipykernel install --user --name projetmlsp-valeria '
          '--display-name "ProjetMLSP (Valeria)" --env PYTHONPATH "" '
          '--env PYTHONHOME "" --env PYTHONNOUSERSITE 1')
    print("The installer did not test GPU availability. Restart Jupyter and select the new kernel.")


if __name__ == "__main__":
    main()
