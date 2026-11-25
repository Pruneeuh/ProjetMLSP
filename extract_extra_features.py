"""Étape 2b : caractéristiques supplémentaires sur les mêmes fenêtres et bandes DWT.

Ajoute à extract_entropy.py (SampEn, DispEn, FuzzyEn, Shannon) :
  ApEn, PermEn  : entropies de Li et al. (2024) [2]
  LZC           : complexité de Lempel-Ziv (cinquième mesure de la proposition)
  LogPower      : log de l'énergie de la sous-bande
  RelPower      : part de l'énergie de la bande dans l'énergie totale de la fenêtre

Même fenêtrage, filtrage et DWT que extract_entropy.py (mêmes valeurs par défaut), donc les lignes
se joignent sur (subject_id, window).

Sortie : outputs/<name>_extra_features.csv, colonnes ch{XX}_{bande}_{mesure}

Usage :
    python extract_extra_features.py                  # les 3 jeux
    python extract_extra_features.py sakib --workers 4
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import EntropyHub as EH

from extract_entropy import BANDS, FS, bandpass, dwt_bands, make_windows
from load_eeg import OUTPUT_DIR, load_saved_dataset


def _last(v):
    tab = np.ravel(v)
    if tab.size == 0:
        return np.nan
    v = float(tab[-1])
    return v if np.isfinite(v) else np.nan


def approximate_entropy(x, m=2, r=0.2):
    sd = x.std()
    if sd == 0:
        return np.nan
    try:
        return _last(EH.ApEn(x, m=m, r=r * sd)[0])
    except Exception:
        return np.nan


def permutation_entropy(x, m=4, tau=1):
    """Entropie de permutation normalisée (Pnorm d'EntropyHub)."""
    try:
        return _last(EH.PermEn(x, m=m, tau=tau)[1])
    except Exception:
        return np.nan


def lempel_ziv(x):
    """Complexité de Lempel-Ziv (LZ76) du signal binarisé à la médiane, normalisée par n / log2(n)."""
    s = (x > np.median(x)).astype(np.int8).tolist()
    n = len(s)
    if n < 2:
        return np.nan
    i, k, l = 0, 1, 1
    k_max, c = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            k_max = max(k, k_max)
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return c / (n / np.log2(n))


ENTROPIES = {
    "ApEn": approximate_entropy,
    "PermEn": permutation_entropy,
    "LZC": lempel_ziv,
}


def process_subject(task):
    sid, label, sig, fs, win_sec, overlap, wavelet = task
    sig = bandpass(sig, fs)
    win, starts = make_windows(sig.shape[1], fs, win_sec, overlap)
    rows = []
    for w, s in enumerate(starts):
        seg = sig[:, s:s + win]
        row = {"subject_id": sid, "label": label, "window": w}
        for ch in range(seg.shape[0]):
            coefs = dwt_bands(seg[ch], fs, wavelet)
            energy = np.array([np.sum(c ** 2) for c in coefs])
            total = energy.sum()
            for band, coef, e in zip(BANDS, coefs, energy):
                prefix = f"ch{ch + 1:02d}_{band}"
                for ename, fn in ENTROPIES.items():
                    v = fn(coef)
                    row[f"{prefix}_{ename}"] = v if np.isfinite(v) else np.nan
                row[f"{prefix}_LogPower"] = np.log10(e + 1e-12)
                row[f"{prefix}_RelPower"] = e / total if total > 0 else np.nan
        rows.append(row)
    return rows


def run_dataset(name, win_sec, overlap, wavelet, workers):
    signals, labels, ids = load_saved_dataset(name)
    fs = FS[name]
    print(f"[{name}] {len(signals)} enregistrements, fs={fs} Hz, fenêtre={win_sec}s, chevauchement={overlap}")
    tasks = [(sid, int(lab), sig, fs, win_sec, overlap, wavelet)
             for sid, lab, sig in zip(ids, labels, signals)]
    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, rows in enumerate(ex.map(process_subject, tasks), 1):
            all_rows.extend(rows)
            print(f"[{name}] {i}/{len(tasks)} sujets traités", flush=True)
    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / f"{name}_extra_features.csv"
    df.to_csv(out, index=False)
    feat_cols = [c for c in df.columns if c.startswith("ch")]
    print(f"  -> {out.name}: {len(df)} fenêtres x {len(feat_cols)} features")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help=f"parmi {list(FS)} (défaut : tous)")
    p.add_argument("--window", type=float, default=10.0)
    p.add_argument("--overlap", type=float, default=0.0)
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    bad = [d for d in args.datasets if d not in FS]
    if bad:
        p.error(f"jeu(x) inconnu(s) : {bad}")
    for name in args.datasets or FS:
        run_dataset(name, args.window, args.overlap, args.wavelet, args.workers)


if __name__ == "__main__":
    main()
