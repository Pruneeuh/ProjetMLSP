"""Install notebook requirements using an Alliance-converted CuPy wheel."""

import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


BUILDER_URL = "https://github.com/ComputeCanada/wheels_builder.git"
BUILDER_COMMIT = "c58455e3f8a775f718319dd4bdffb65086236d6b"


def cupy_version(requirements):
    matches = re.findall(r"(?im)^\s*cupy-cuda12x\s*==\s*([0-9][0-9.]*)\s*$",
                         Path(requirements).read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("requirement.txt must contain exactly one pinned cupy-cuda12x version.")
    return matches[0]


def cupy_wheel_url(version):
    if sys.platform != "linux" or platform.machine() != "x86_64":
        raise ValueError("CuPy conversion is supported only on Linux x86_64.")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    suffix = f"-{py}-{py}-manylinux2014_x86_64.whl"
    with urllib.request.urlopen(f"https://pypi.org/pypi/cupy-cuda12x/{version}/json", timeout=30) as response:
        release = json.load(response)
    wheels = [item["url"] for item in release["urls"]
              if item["filename"].endswith(suffix) and not item.get("yanked")]
    if len(wheels) != 1:
        raise ValueError(f"No unique CuPy {version} wheel found for {py} on PyPI.")
    return wheels[0]


def check_wheel(path):
    with zipfile.ZipFile(path) as wheel:
        files = wheel.infolist()
        if (not any(item.filename == "cupy/__init__.py" and item.file_size for item in files)
                or not any(item.filename.startswith("cupy/_core/") and item.filename.endswith(".so")
                           and item.file_size for item in files)):
            raise ValueError(f"Converted CuPy wheel is incomplete: {path}")


def install(requirements):
    requirements = Path(requirements).resolve()
    version = cupy_version(requirements)
    url = cupy_wheel_url(version)
    missing = [name for name in ("git", "bash", "wget", "setrpaths") if shutil.which(name) is None]
    if missing:
        raise RuntimeError("Alliance wheel conversion tools not found: " + ", ".join(missing)
                           + ". Load the Valeria software environment or contact support.")
    user = ["--user"] if sys.prefix == sys.base_prefix else []
    with tempfile.TemporaryDirectory(prefix="projetmlsp-wheel-") as temp:
        root = Path(temp)
        builder, wheels = root / "wheels_builder", root / "wheels"
        wheels.mkdir()
        subprocess.run(["git", "clone", "--no-checkout", BUILDER_URL, str(builder)], check=True)
        subprocess.run(["git", "-C", str(builder), "checkout", "--detach", BUILDER_COMMIT], check=True)
        subprocess.run(["bash", "-e", str(builder / "unmanylinuxize.sh"),
                        "--package", "cupy-cuda12x", "--version", version, "--url", url],
                       cwd=wheels, check=True)
        converted = list(wheels.glob(f"cupy_cuda12x-{version}+computecanada-*-linux_x86_64.whl"))
        if len(converted) != 1:
            raise ValueError("Alliance conversion did not produce exactly one compatible CuPy wheel.")
        check_wheel(converted[0])
        pip = [sys.executable, "-m", "pip", "--isolated", "install", *user]
        subprocess.run([*pip, "--no-deps", "--force-reinstall", str(converted[0])], check=True)
        subprocess.run([*pip, "--only-binary=:all:", "--index-url", "https://pypi.org/simple",
                        "--find-links", str(wheels), "-r", str(requirements)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    subprocess.run([sys.executable, "-c", "import cupy; print('CuPy import OK:', cupy.__version__)"], check=True)
    print("Installation complete. Restart the notebook kernel before running the pipeline.", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python 00_notebook_install.py requirement.txt")
    install(sys.argv[1])
