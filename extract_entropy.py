"""Étape 2 : prétraitement, fenêtrage, DWT (5 bandes) et calcul des 5 entropies.

Entrée  : outputs/<name>_signals.npz + <name>_metadata.csv  (produits par load_eeg.py)
Sortie  : outputs/<name>_entropy_features.csv
          une ligne par fenêtre : subject_id, label, window, puis
          ch{XX}_{bande}_{entropie}  (canaux x 5 bandes x 4 entropies)

Entropies : SampEn, DispEn, FuzzyEn, Shannon
Bandes    : delta, theta, alpha, beta, gamma (via ondelettes discrètes, db4)

Usage :
    python extract_entropy.py                       # les 3 jeux
    python extract_entropy.py sakib borisov         # un sous-ensemble
    python extract_entropy.py --window 10 --overlap 0.5 --workers 4

Dépendances : pip install numpy scipy pandas pywavelets EntropyHub
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, sosfiltfilt
import EntropyHub as EH

from load_eeg import OUTPUT_DIR, load_saved_dataset

# Fréquences d'échantillonnage (voir Tableau 1 de la proposition)
FS = {"mumtaz": 256, "borisov": 128, "sakib": 128}
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BANDPASS = (0.5, 45.0)  # Hz


# --------------------------------------------------------------------------
# Prétraitement
# --------------------------------------------------------------------------
def bandpass(sig: np.ndarray, fs: float, low=BANDPASS[0], high=BANDPASS[1]) -> np.ndarray:
    """Filtre Butterworth passe-bande, phase nulle. sig : (canaux, échantillons)."""
    high = min(high, 0.95 * fs / 2)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1)


def make_windows(n_samples: int, fs: float, win_sec: float, overlap: float):
    """Indices de début des fenêtres de durée fixe."""

    if not 0 <= overlap < 1:
        raise ValueError(f"overlap doit être dans [0, 1) : {overlap}")
    if win_sec <= 0:
        raise ValueError(f"win_sec doit être > 0 : {win_sec}")
    win = int(round(win_sec * fs))
    step = max(1, int(round(win * (1 - overlap))))
    return win, list(range(0, n_samples - win + 1, step))


# --------------------------------------------------------------------------
# DWT : 5 bandes
# --------------------------------------------------------------------------
def dwt_bands(x: np.ndarray, fs: float, wavelet: str = "db4"):
    """Décompose x en 5 sous-bandes [delta, theta, alpha, beta, gamma].

    Niveau L = log2(fs/8) pour que l'approximation cA_L couvre 0-4 Hz :
      fs=256 -> L=5 : cA5 0-4 | cD5 4-8 | cD4 8-16 | cD3 16-32 | cD2 32-64
      fs=128 -> L=4 : cA4 0-4 | cD4 4-8 | cD3 8-16 | cD2 16-32 | cD1 32-64
    pywt.wavedec renvoie [cA_L, cD_L, ..., cD_1] : les 5 premiers éléments
    correspondent donc directement à delta..gamma, quel que soit fs.
    (gamma est tronquée par le filtre à 45 Hz / Nyquist ; alpha ~ 8-16 Hz.)
    """
    level = int(round(np.log2(fs / 8)))
    return pywt.wavedec(x, wavelet, level=level)[:5]


# --------------------------------------------------------------------------
# Calcul des entropies
# --------------------------------------------------------------------------


def _last(v): 
    """Retourne la dernière valeur du tableau renvoyé par EntropyHub, ou NaN si non finie."""
    tab = np.ravel(v)
    if tab.size == 0: 
        return np.nan
    v = float(tab[-1])
    return v if np.isfinite(v) else np.nan 

def sample_entropy(x, m=2, r=0.2):
    """Calcul SampEn en utilisant EntropyHub """

    sd = x.std()
    if sd == 0:
        return np.nan
    try : 
        SamEn,_,_ = EH.SampEn(x, m=m, r=r*sd )
    except Exception : 
        return np.nan
    return _last(SamEn)

def fuzzy_entropy(x, m=2, r=0.2, nexp=2):
    """Calcul FuzzyEn en utilisant EntropyHub """
    sd = x.std()
    if sd == 0:
        return np.nan
    
    try:
        FuzzyEn,_,_ = EH.FuzzEn(x, m=m, r=(r*sd,nexp))
    except Exception:
        return np.nan
    return _last(FuzzyEn)
   

def dispersion_entropy(x, m=2, c=6, delay=1):
    """DispEn (dispersion entropy) normalisée en utilisant EntropyHub """
    try : 
        DispEn,_ = EH.DispEn(x, m=m, c=c, tau=delay, Norm=True)
    except Exception:
        return np.nan
    return _last(DispEn)
    

def shannon_entropy(x, bins=32):
    """Entropie de Shannon de l'histogramme des amplitudes"""
    try : 
        h, _ = np.histogram(x, bins=bins)
        p = h / h.sum()
        p = p[p > 0]
        return -np.sum(p*np.log2(p))
    except Exception:
        return np.nan



ENTROPIES = {
    "SampEn": sample_entropy,
    "DispEn": dispersion_entropy,
    "FuzzyEn": fuzzy_entropy,
    "Shannon": shannon_entropy,
}


# --------------------------------------------------------------------------
# Traitement d'un participant
# --------------------------------------------------------------------------
def process_subject(task):
    sid, label, sig, fs, win_sec, overlap, wavelet = task
    sig = bandpass(sig, fs)
    win, starts = make_windows(sig.shape[1], fs, win_sec, overlap)
    rows = []
    for w, s in enumerate(starts):
        seg = sig[:, s:s + win]
        row = {"subject_id": sid, "label": label, "window": w}
        for ch in range(seg.shape[0]):
            for band, coef in zip(BANDS, dwt_bands(seg[ch], fs, wavelet)):
                for ename, fn in ENTROPIES.items():
                    v = fn(coef)
                    row[f"ch{ch + 1:02d}_{band}_{ename}"] = v if np.isfinite(v) else np.nan
        rows.append(row)
    return rows


def run_dataset(name, win_sec, overlap, wavelet, workers):
    signals, labels, ids = load_saved_dataset(name)
    fs = FS[name]
    print(f"[{name}] {len(signals)} enregistrements, fs={fs} Hz, "
          f"fenêtre={win_sec}s, chevauchement={overlap}")
    tasks = [(sid, int(lab), sig, fs, win_sec, overlap, wavelet)
             for sid, lab, sig in zip(ids, labels, signals)]

    all_rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        '''for i, rows in enumerate(ex.map(process_subject, tasks), 1):
            all_rows.extend(rows)
            print(f"  {i}/{len(tasks)} {tasks[i - 1][0]}: {len(rows)} fenêtres")'''
        for i, rows in enumerate(ex.map(process_subject, tasks), 1):
            all_rows.extend(rows)
            print(f"[{name}] {i}/{len(tasks)} sujets traités")

    df = pd.DataFrame(all_rows)
    out = OUTPUT_DIR / f"{name}_entropy_features.csv"
    df.to_csv(out, index=False)

    feat_cols = [c for c in df.columns if c.startswith("ch")]
    n_nan = int(df[feat_cols].isna().sum().sum())
    print(f"  -> {out.name}: {len(df)} fenêtres x {len(feat_cols)} features, "
          f"{n_nan} NaN, fenêtres par classe : {df['label'].value_counts().to_dict()}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help=f"parmi {list(FS)} (défaut : tous)")
    p.add_argument("--window", type=float, default=10.0, help="durée de fenêtre en secondes")
    p.add_argument("--overlap", type=float, default=0.0, help="chevauchement entre 0 et <1")
    p.add_argument("--wavelet", default="db4")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    bad = [d for d in args.datasets if d not in FS]
    if bad:
        p.error(f"jeu(x) inconnu(s) : {bad}")
    print("workers =", args.workers)
    for name in args.datasets or FS:
        run_dataset(name, args.window, args.overlap, args.wavelet, args.workers)


if __name__ == "__main__":
    main()