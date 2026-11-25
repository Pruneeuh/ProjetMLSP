"""Génère de petites données EEG SYNTHÉTIQUES au format des 3 bases brutes, pour tester
tout le pipeline sans les vraies données (les résultats obtenus n'ont aucune valeur).

Sortie (par défaut data_eeg_test/ à côté des scripts) :
    mumtaz/H S1 EC.edf ...       EDF 256 Hz, 20 ou 22 canaux « EEG Fp1-LE » (+ 1 doublon)
    borisov/norm.zip, sch.zip    fichiers .eea texte 16 × 7680 (128 Hz, 60 s) ; contient 387-02w1/387-03w1
    sakib/ACSub1.mat, ASub1.mat  matrices (échantillons, 14) à 128 Hz

Chaque base a un « effet de site » (gain, pente spectrale) et chaque trouble modifie
la puissance alpha et la part de bruit lent, pour que les analyses aient quelque chose à trouver.

Usage :
    python outils/generer_donnees_test.py --sortie data_eeg_test --duree 60
    python 01_chargement_donnees.py --donnees data_eeg_test
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io

SCRIPT_DIR = Path(__file__).resolve().parent.parent


def signal(rng, n_ch, n, fs, gain, alpha, pente):
    """Bruit en 1/f^pente + rythme alpha (10 Hz) partagé en partie entre canaux. µV."""
    f = np.fft.rfftfreq(n, 1 / fs)
    spec = (rng.normal(size=(n_ch, len(f))) + 1j * rng.normal(size=(n_ch, len(f)))) / np.maximum(f, 0.5) ** (pente / 2)
    x = np.fft.irfft(spec, n=n, axis=1)
    x = x / x.std(axis=1, keepdims=True)
    t = np.arange(n) / fs
    commun = np.sin(2 * np.pi * (10 + 0.3 * rng.normal()) * t + rng.uniform(0, 6.28))
    env = 1 + 0.5 * np.sin(2 * np.pi * 0.1 * t + rng.uniform(0, 6.28))
    poids = rng.uniform(0.3, 1.0, size=(n_ch, 1))
    gain = gain * rng.lognormal(0, 0.35)              # variabilité entre sujets
    alpha = alpha * rng.lognormal(0, 0.35)
    return gain * (x + alpha * poids * env * commun)


def write_edf(path, data_uv, fs, labels):
    """Écrit un EDF minimal (16 bits, enregistrements de 1 s)."""
    ns, n = data_uv.shape
    spr = int(fs)
    nrec = n // spr
    data_uv = data_uv[:, :nrec * spr]
    pmin, pmax = -3000.0, 3000.0
    dmin, dmax = -32768, 32767
    dig = np.clip(np.round((data_uv - pmin) / (pmax - pmin) * (dmax - dmin) + dmin), dmin, dmax).astype("<i2")
    f = lambda s, w: str(s)[:w].ljust(w).encode("ascii")
    h = f("0", 8) + f("X X X X", 80) + f("Startdate 01-JAN-2020 X X X", 80) + f("01.01.20", 8) + f("00.00.00", 8)
    h += f(256 + 256 * ns, 8) + f("", 44) + f(nrec, 8) + f(1, 8) + f(ns, 4)
    for key, w in [("label", 16), ("trans", 80), ("dim", 8), ("pmin", 8), ("pmax", 8), ("dmin", 8),
                   ("dmax", 8), ("pre", 80), ("spr", 8), ("res", 32)]:
        for lab in labels:
            v = {"label": lab, "trans": "AgAgCl", "dim": "uV", "pmin": pmin, "pmax": pmax, "dmin": dmin,
                 "dmax": dmax, "pre": "", "spr": spr, "res": ""}[key]
            h += f(v, w)
    with open(path, "wb") as fh:
        fh.write(h)
        for r in range(nrec):
            fh.write(dig[:, r * spr:(r + 1) * spr].tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sortie", type=Path, default=SCRIPT_DIR / "data_eeg_test")
    p.add_argument("--duree", type=float, default=60.0, help="Durée Mumtaz et Sakib (s) ; Borisov = 60 s.")
    p.add_argument("--n", type=int, nargs=3, default=[10, 12, 10], metavar=("DEP", "SCZ", "ANX"),
                   help="Sujets par groupe (témoins = patients) pour chaque base.")
    p.add_argument("--graine", type=int, default=0)
    args = p.parse_args()
    rng = np.random.default_rng(args.graine)
    out = args.sortie
    for d in ["mumtaz", "borisov", "sakib"]:
        (out / d).mkdir(parents=True, exist_ok=True)

    # Mumtaz : 256 Hz, µV dans l'EDF ; 20 ou 22 canaux selon le fichier.
    noms = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz",
            "P4", "T6", "O1", "O2"]
    fs = 256
    premier = None
    for grp, lab in [("H", 0), ("MDD", 1)]:
        for s in range(1, args.n[0] + 1):
            extra = ["EEG A2-A1"] + (["EEG 23A-23R", "EEG 24A-24R"] if rng.random() < 0.5 else [])
            labels = [f"EEG {c}-LE" for c in noms] + extra
            x = signal(rng, len(labels), int(args.duree * fs), fs, gain=12, alpha=1.5 - 0.4 * lab, pente=1.2 + 0.1 * lab)
            write_edf(out / "mumtaz" / f"{grp} S{s} EC.edf", x, fs, labels)
            if premier is None:
                premier = (x, labels)
    write_edf(out / "mumtaz" / f"MDD S{args.n[0] + 1} EC.edf", premier[0], fs, premier[1])   # doublon volontaire

    # Borisov : 16 × 7680 empilés verticalement, texte.
    for zname, lab, pref in [("norm.zip", 0, "norm"), ("sch.zip", 1, "sch")]:
        with zipfile.ZipFile(out / "borisov" / zname, "w") as zf:
            ids = [f"{100 + i}w1" for i in range(args.n[1])]
            if lab == 1:
                ids[:2] = ["387-02w1", "387-03w1"]
            for sid in ids:
                x = signal(rng, 16, 7680, 128, gain=25, alpha=1.2 + 0.3 * lab, pente=1.5 - 0.1 * lab)
                buf = io.StringIO()
                np.savetxt(buf, x.reshape(-1), fmt="%.3f")
                zf.writestr(f"{pref}/{sid}.eea", buf.getvalue())

    # Sakib : (échantillons, 14), offset DC Emotiv ~4200 µV.
    for pref, lab in [("ACSub", 0), ("ASub", 1)]:
        for s in range(1, args.n[2] + 1):
            x = signal(rng, 14, int(args.duree * 128), 128, gain=8, alpha=1.0 + 0.2 * lab, pente=1.0 + 0.05 * lab)
            scipy.io.savemat(out / "sakib" / f"{pref}{s}.mat", {"data": (x + 4200).T})
    print(f"Données synthétiques écrites dans {out}")


if __name__ == "__main__":
    main()
