"""Download the three public EEG datasets used in the project.

Layout produced (matches load_eeg.py):
    data_eeg/
    ├── mumtaz/     depression  - Mumtaz (2016), Figshare 4244171 (EDF files)
    ├── borisov/    schizophrenia - norm.zip + sch.zip (Moscow State University)
    └── sakib/      anxiety - Sakib et al. (2023), Mendeley rh3fy75zdv v2 (.mat files)

Usage:
    python download_data.py                      # everything
    python download_data.py anxiety borisov      # subset
    python download_data.py --out D:/eeg
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "Mozilla/5.0 (ProjetMLSP data downloader)"}
FIGSHARE_ARTICLE = "https://api.figshare.com/v2/articles/4244171"
BORISOV_URLS = {
    "norm.zip": "http://brain.bio.msu.ru/eeg_data/schizophrenia/norm.zip",
    "sch.zip": "http://brain.bio.msu.ru/eeg_data/schizophrenia/sch.zip",
}
SAKIB_ZIP = "https://data.mendeley.com/public-api/zip/rh3fy75zdv/download/2"


def fetch(url, dest, expected_size=None):
    dest = Path(dest)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
        print(f"  skip {dest.name} (already downloaded)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    print(f"  ok   {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_mumtaz(out):
    folder = out / "mumtaz"
    with urllib.request.urlopen(urllib.request.Request(FIGSHARE_ARTICLE, headers=UA), timeout=60) as r:
        files = json.load(r)["files"]
    print(f"[mumtaz] {len(files)} EDF files from Figshare")
    for f in files:
        try:
            fetch(f["download_url"], folder / f["name"], f.get("size"))
        except Exception as e:
            print(f"  WARN {f['name']}: {e}", file=sys.stderr)


def download_borisov(out):
    folder = out / "borisov"
    print("[borisov] Moscow State University (norm.zip, sch.zip)")
    for name, url in BORISOV_URLS.items():
        fetch(url, folder / name)


def download_sakib(out):
    folder = out / "sakib"
    print("[sakib] Mendeley Data (.mat files)")
    if any(folder.glob("*.mat")):
        print("  skip (.mat files already present)")
        return
    zp = folder / "_download.zip"
    fetch(SAKIB_ZIP, zp)
    with zipfile.ZipFile(zp) as z:
        for m in z.namelist():
            if m.lower().endswith(".mat"):
                with z.open(m) as src, open(folder / Path(m).name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    zp.unlink()
    print(f"  extracted {len(list(folder.glob('*.mat')))} .mat files")


DATASETS = {"mumtaz": download_mumtaz, "borisov": download_borisov, "sakib": download_sakib}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help=f"any of {list(DATASETS)} (default: all)")
    p.add_argument("--out", default="data_eeg", help="output directory (default: data_eeg)")
    args = p.parse_args()
    bad = [d for d in args.datasets if d not in DATASETS]
    if bad:
        p.error(f"unknown dataset(s): {bad}")
    for name in args.datasets or DATASETS:
        try:
            DATASETS[name](Path(args.out))
        except Exception as e:
            print(f"  FAILED {name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
