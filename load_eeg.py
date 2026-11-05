"""Load the raw EEG datasets downloaded by download_data.py and save them as
<name>_signals.npz + <name>_metadata.csv in outputs/ (label: 0 = control, 1 = patient).

Dependencies: pip install numpy scipy pandas mne

    python download_data.py
    python load_eeg.py
Later steps: from load_eeg import load_saved_dataset
"""
import re
import zipfile
from pathlib import Path
from typing import List, Tuple

import mne
import numpy as np
import pandas as pd
import scipy.io

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data_eeg"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

MUMTAZ_CONDITION = "EC"  # EC (eyes closed), EO (eyes open) or TASK
MUMTAZ_CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                   "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

Dataset = Tuple[List[np.ndarray], np.ndarray, List[str]]


def load_mumtaz(path: Path, condition: str = MUMTAZ_CONDITION) -> Dataset:
    """Mumtaz (depression), EDF files 'H S1 EC.edf' / 'MDD S1 EC.edf'. 256 Hz, 19 channels."""
    signals, labels, ids = [], [], []
    pattern = re.compile(r"^(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$", re.I)
    for f in sorted(path.glob("*.edf")):
        m = pattern.match(re.sub(r"\s+", " ", f.name))
        if not m or m.group(3).upper() != condition.upper():
            continue
        group, num = m.group(1).upper(), m.group(2)
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
            names = {}
            for ch in raw.ch_names:  # e.g. "EEG Fp1-LE" -> "Fp1"
                core = re.sub(r"^EEG\s*", "", ch).split("-")[0].strip()
                names[core.lower()] = ch
            picks = [names[c.lower()] for c in MUMTAZ_CHANNELS if c.lower() in names]
            if len(picks) != len(MUMTAZ_CHANNELS):
                print(f"  ! {f.name}: {len(picks)}/19 channels found, ignored")
                continue
            data = raw.copy().pick(picks).get_data()
        except Exception as e:
            print(f"  ! Error reading {f.name}: {e}")
            continue
        ids.append(f"{group}_S{num}_{condition.upper()}")
        labels.append(0 if group == "H" else 1)
        signals.append(data)
    return signals, np.array(labels, dtype=int), ids


def load_borisov(zip_dir: Path) -> Dataset:
    """Borisov (schizophrenia): norm.zip / sch.zip of .eea text files,
    16 channels x 7680 samples (128 Hz, 60 s) stacked vertically."""
    n_ch, n_samp = 16, 7680
    signals, labels, ids = [], [], []
    for zname, label, prefix in (("norm.zip", 0, "HC"), ("sch.zip", 1, "SZ")):
        zp = zip_dir / zname
        if not zp.exists():
            print(f"  ! Missing file: {zp}")
            continue
        with zipfile.ZipFile(zp) as zf:
            names = sorted(n for n in zf.namelist()
                           if n.lower().endswith((".eea", ".txt")) and not n.endswith("/"))
            for n in names:
                with zf.open(n) as fh:
                    data = np.loadtxt(fh, dtype=np.float64)
                if data.size != n_ch * n_samp:
                    print(f"  ! Unexpected size for {n}: {data.size}, ignored")
                    continue
                ids.append(f"{prefix}_{Path(n).stem}")
                labels.append(label)
                signals.append(data.reshape(n_ch, n_samp))
    return signals, np.array(labels, dtype=int), ids


def load_sakib(path: Path) -> Dataset:
    """Sakib (anxiety): .mat files, ASub* = anxiety (1), ACSub* = control (0). 14 channels, 128 Hz."""
    signals, labels, ids = [], [], []
    for f in sorted(path.glob("*.mat")):
        sid = f.stem
        if sid.startswith("ACSub"):
            label = 0
        elif sid.startswith("ASub"):
            label = 1
        else:
            print(f"  ! Unrecognized name, ignored: {sid}")
            continue
        mat = scipy.io.loadmat(f)
        arrays = [v for k, v in mat.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray)
                  and v.ndim == 2 and min(v.shape) > 1 and np.issubdtype(v.dtype, np.number)]
        if not arrays:
            print(f"  ! No EEG array in {sid}")
            continue
        arr = max(arrays, key=lambda a: a.size)
        data = arr.T if arr.shape[0] > arr.shape[1] else arr  # -> (channels, samples)
        ids.append(sid)
        labels.append(label)
        signals.append(np.asarray(data, dtype=np.float64))
    return signals, np.array(labels, dtype=int), ids


def save_dataset(name: str, signals, labels, ids, output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not signals:
        print(f"  (nothing to save for '{name}')")
        return
    npz = output_dir / f"{name}_signals.npz"
    np.savez_compressed(npz, **dict(zip(ids, signals)))
    pd.DataFrame({
        "subject_id": ids, "label": labels,
        "n_channels": [s.shape[0] for s in signals],
        "n_samples": [s.shape[1] for s in signals],
    }).to_csv(output_dir / f"{name}_metadata.csv", index=False)
    print(f"  -> {npz.name}: {len(signals)} recordings")


def load_saved_dataset(name: str, output_dir: Path = OUTPUT_DIR) -> Dataset:
    """Reload (signals, labels, subject_ids) saved by this script."""
    meta = pd.read_csv(output_dir / f"{name}_metadata.csv")
    with np.load(output_dir / f"{name}_signals.npz") as npz:
        signals = [npz[s] for s in meta["subject_id"]]
    return signals, meta["label"].to_numpy(dtype=int), meta["subject_id"].tolist()


def summarize(name: str, signals, labels) -> None:
    if not signals:
        print(f"  {name}: no data")
        return
    counts = dict(zip(*[x.tolist() for x in np.unique(labels, return_counts=True)]))
    print(f"  {name}: {len(signals)} recordings, labels {counts}, first shape {signals[0].shape}")


if __name__ == "__main__":
    jobs = (("mumtaz", load_mumtaz, DATA_ROOT / "mumtaz"),
            ("borisov", load_borisov, DATA_ROOT / "borisov"),
            ("sakib", load_sakib, DATA_ROOT / "sakib"))
    for i, (name, fn, p) in enumerate(jobs, 1):
        print(f"[{i}/3] {name}")
        sig, lab, ids = fn(p)
        summarize(name, sig, lab)
        save_dataset(name, sig, lab, ids)
