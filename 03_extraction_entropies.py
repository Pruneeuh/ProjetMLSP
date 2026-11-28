"""Étape 3 : prétraitement, fenêtrage et extraction des caractéristiques EEG.

Entrée  : sorties/02_controle_qualite/<date>/<base>_signals.npz et métadonnées (02)
Sortie  : sorties/03_extraction/<base>_<date>/
    features_windows.csv       format LONG, une ligne par fenêtre × canal × composante :
                               dataset, subject_id, group_validation, label, window, channel, band,
                               sd_before_standardization, SampEn, DispEn, FuzzyEn, Shannon_spectrale, LZC
                               (même format que l'ancien extraction_corrigee.py : les étapes 04 à 10
                               acceptent les deux)
    audit_enregistrements.csv  un fichier source par ligne (statut extrait / doublon_exclu_de_analyse /
                               ..., nombre de fenêtres et de lignes, empreinte du signal)
    features_large.csv         (option --large) une ligne par fenêtre, colonnes canal_bande_mesure
    features_sujets.csv        moyennes des fenêtres par participant × canal × composante
    parametres_extraction.json

ApEn, PermEn, Shannon_amplitude, LogPower et RelPower sont toujours calculés.
RCMSE et RCMPE sont calculées sur chaque canal large bande aux échelles 1 à 10 ;
elles figurent uniquement sur la ligne A4 pour ne pas les répéter par bande DWT.
LogPower est le log10 de l'énergie DWT en µV², pas une puissance moyenne.

Chaîne de traitement, pour chaque enregistrement :
  1. filtre Butterworth passe-bande 0,5-45 Hz et rééchantillonnage à 128 Hz effectués par 02 ;
  3. troncature optionnelle (--duree-max, ex. 60 s pour égaliser avec Borisov) ;
  4. fenêtres de --fenetre s (défaut 10 s), chevauchement --chevauchement ;
  5. DWT db4 niveau 4 → A4 (0-4 Hz), D4 (4-8), D3 (8-16), D2 (16-32), D1 (32-64, tronquée à 45) ;
  6. pour chaque composante : écart-type (gardé comme référence d'amplitude), standardisation,
     puis 5 mesures :
       SampEn            entropie d'échantillon (m=2, r=0,2·sd, distance de Tchebychev)
       DispEn            entropie de dispersion normalisée (m=2, c=6, fonction de répartition normale)
       FuzzyEn           entropie floue (m=2, r=0,2, n=2, retrait de la moyenne de chaque motif)
       Shannon_spectrale entropie de Shannon de la DSP de Welch normalisée (0 à 1)
       LZC               complexité de Lempel-Ziv (1976) normalisée, binarisation à la médiane
     Implémentation numpy/scipy par défaut ; --entropyhub utilise EntropyHub pour SampEn,
     FuzzyEn et DispEn (comme extract_entropy.py).

Usage :
    python 03_extraction_entropies.py                         # les 3 bases
    python 03_extraction_entropies.py anxiete --workers 4
    python 03_extraction_entropies.py --fenetre 10 --chevauchement 0.5 --duree-max 60
    python 03_extraction_entropies.py --entropyhub --caracteristiques-etendues --large

Dépendances : numpy scipy pandas pywavelets EntropyHub
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
from scipy.signal import welch
from scipy.special import ndtr
from scipy.spatial import cKDTree

from eeg_config import AMPLITUDE, BANDPASS, BANDS, BASES, FS_CIBLE, MEASURES, MESURES_ETENDUES, MULTIECHELLE, PUISSANCES
from eeg_donnees import canaux_sauvegardes, charger_signaux
from eeg_pretraitement import bandpass, notch, pretraiter, reechantillonner
from eeg_utils import arguments_dict, dernier_dossier, ecrire_json, log, nouveau_dossier, versions


# --------------------------------------------------------------------------
# Prétraitement
# --------------------------------------------------------------------------
def make_windows(n_samples, fs, win_sec, overlap):
    """Taille et indices de début des fenêtres de durée fixe."""
    if not 0 <= overlap < 1:
        raise ValueError(f"le chevauchement doit être dans [0, 1) : {overlap}")
    if win_sec <= 0:
        raise ValueError(f"la fenêtre doit être > 0 : {win_sec}")
    win = int(round(win_sec * fs))
    step = max(1, int(round(win * (1 - overlap))))
    return win, list(range(0, n_samples - win + 1, step))


def dwt_bands(x, fs=FS_CIBLE, wavelet="db4"):
    """5 composantes [A4, D4, D3, D2, D1] : niveau L = log2(fs/8) → A_L couvre 0-4 Hz."""
    level = int(round(np.log2(fs / 8)))
    return pywt.wavedec(x, wavelet, level=level)[:5]


# --------------------------------------------------------------------------
# Entropies (signal déjà standardisé : moyenne 0, écart-type 1)
# --------------------------------------------------------------------------
def _cheb_matrices(x, m):
    """Distances de Tchebychev entre motifs de longueur m et m+1 (mêmes N-m motifs)."""
    t = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    d = np.abs(t[:, 0][:, None] - t[:, 0][None, :])
    for k in range(1, m):
        np.maximum(d, np.abs(t[:, k][:, None] - t[:, k][None, :]), out=d)
    d1 = np.maximum(d, np.abs(t[:, m][:, None] - t[:, m][None, :]))
    return d, d1


def sample_entropy(x, m=2, r=0.2):
    """SampEn = -ln(A/B), A et B = paires de motifs semblables (i<j) de longueur m+1 et m."""
    if len(x) <= m + 2:
        return np.nan
    d, d1 = _cheb_matrices(x, m)
    iu = np.triu_indices(len(d), k=1)
    b = np.count_nonzero(d[iu] <= r)
    a = np.count_nonzero(d1[iu] <= r)
    return float(-np.log(a / b)) if a > 0 and b > 0 else np.nan


def fuzzy_entropy(x, m=2, r=0.2, n=2):
    """FuzzyEn (Chen et al. 2007) : appartenance exp(-d^n / r), motifs centrés."""
    if len(x) <= m + 2:
        return np.nan
    t = np.lib.stride_tricks.sliding_window_view(x, m + 1)
    phis = []
    for dim in (m, m + 1):
        u = t[:, :dim] - t[:, :dim].mean(axis=1, keepdims=True)
        d = np.abs(u[:, 0][:, None] - u[:, 0][None, :])
        for k in range(1, dim):
            np.maximum(d, np.abs(u[:, k][:, None] - u[:, k][None, :]), out=d)
        sim = np.exp(-(d ** n) / r)
        np.fill_diagonal(sim, 0.0)
        phis.append(sim.sum() / (len(u) * (len(u) - 1)))
    return float(np.log(phis[0]) - np.log(phis[1])) if phis[0] > 0 and phis[1] > 0 else np.nan


def dispersion_entropy(x, m=2, c=6, delay=1):
    """DispEn normalisée (Rostaghi et Azami 2016) : NCDF → c classes → motifs de longueur m."""
    if len(x) < (m - 1) * delay + 2:
        return np.nan
    z = np.clip(np.round(c * ndtr(x) + 0.5), 1, c).astype(np.int64)
    idx = np.arange(len(z) - (m - 1) * delay)
    code = np.zeros(len(idx), dtype=np.int64)
    for k in range(m):
        code = code * c + (z[idx + k * delay] - 1)
    _, counts = np.unique(code, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(c ** m))


def spectral_shannon_entropy(x, nperseg=128):
    """Entropie de Shannon de la DSP de Welch normalisée, divisée par log(nombre de fréquences)."""
    if len(x) < 8:
        return np.nan
    _, pxx = welch(x, nperseg=min(nperseg, len(x)))
    s = pxx.sum()
    if s <= 0 or len(pxx) < 2:
        return np.nan
    p = pxx / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(pxx)))


def lempel_ziv_complexity(x):
    """LZ76 (algorithme de Kaspar et Schuster), binarisation à la médiane, normalisée c·log2(n)/n."""
    s = (x > np.median(x)).astype(np.uint8).tolist()
    n = len(s)
    if n < 2:
        return np.nan
    c, l, i, k, kmax = 1, 1, 0, 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            kmax = max(k, kmax)
            i += 1
            if i == l:
                c += 1
                l += kmax
                if l + 1 > n:
                    break
                i, k, kmax = 0, 1, 1
            else:
                k = 1
    return float(c * np.log2(n) / n)


def _last(v):
    tab = np.ravel(v)
    if tab.size == 0:
        return np.nan
    v = float(tab[-1])
    return v if np.isfinite(v) else np.nan


def entropies_entropyhub(x, p):
    import EntropyHub as EH
    out = {}
    try:
        out["SampEn"] = _last(EH.SampEn(x, m=p["m"], r=p["r"])[0])
    except Exception:
        out["SampEn"] = np.nan
    try:
        out["FuzzyEn"] = _last(EH.FuzzEn(x, m=p["m"], r=(p["r"], p["n_fuzzy"]))[0])
    except Exception:
        out["FuzzyEn"] = np.nan
    try:
        out["DispEn"] = _last(EH.DispEn(x, m=p["m"], c=p["c_disp"], tau=1, Norm=True)[0])
    except Exception:
        out["DispEn"] = np.nan
    return out


def calculer_mesures(coef, p):
    """Les 5 mesures d'une composante. Renvoie (sd avant standardisation, dict)."""
    coef = np.asarray(coef, dtype=np.float64)
    sd = float(coef.std())
    if not np.isfinite(sd) or sd == 0:
        return sd, {m: np.nan for m in MEASURES}
    x = (coef - coef.mean()) / sd
    if p["entropyhub"]:
        vals = entropies_entropyhub(x, p)
    else:
        vals = {"SampEn": sample_entropy(x, p["m"], p["r"]),
                "FuzzyEn": fuzzy_entropy(x, p["m"], p["r"], p["n_fuzzy"]),
                "DispEn": dispersion_entropy(x, p["m"], p["c_disp"])}
    vals["Shannon_spectrale"] = spectral_shannon_entropy(x)
    vals["LZC"] = lempel_ziv_complexity(x)
    return sd, {m: (vals[m] if np.isfinite(vals[m]) else np.nan) for m in MEASURES}


def mesures_etendues(coef):
    """Mesures du notebook sur les coefficients DWT non standardisés."""
    coef = np.asarray(coef, dtype=np.float64)
    sd = float(coef.std())
    if not np.isfinite(sd) or sd == 0:
        return {name: np.nan for name in MESURES_ETENDUES}
    from EntropyHub import ApEn, PermEn

    hist, _ = np.histogram(coef, bins=8)
    probabilities = hist[hist > 0] / hist.sum()
    values = {
        "ApEn": _last(ApEn(coef, m=2, r=0.2 * sd)[0]),
        "PermEn": _last(PermEn(coef, m=4, tau=1)[1]),
        "Shannon_amplitude": float(-np.sum(probabilities * np.log2(probabilities)) / np.log2(8)),
    }
    return values


def entropies_multiechelle(signal, scales=10):
    """RCMSE et RCMPE sur le signal large bande, selon Xu et al. (2025)."""
    signal = np.asarray(signal, dtype=np.float64)
    if not np.isfinite(signal).all() or signal.std() == 0:
        return {f"{name}_{scale:02d}": np.nan
                for name in ("RCMSE", "RCMPE") for scale in range(1, scales + 1)}
    x = (signal - signal.mean()) / signal.std()
    result = {}
    for scale in range(1, scales + 1):
        counts_m = counts_next = 0
        pattern_probabilities = {}
        for offset in range(scale):
            n = (len(x) - offset) // scale
            coarse = x[offset:offset + n * scale].reshape(n, scale).mean(axis=1)
            if len(coarse) < 5:
                raise ValueError(f"Série trop courte pour l'échelle {scale}")
            templates = np.lib.stride_tricks.sliding_window_view(coarse, 3)
            for dim in (2, 3):
                tree = cKDTree(templates[:, :dim])
                pairs = (tree.count_neighbors(tree, 0.15, p=np.inf) - len(templates)) / 2
                if dim == 2:
                    counts_m += pairs
                else:
                    counts_next += pairs
            patterns = np.argsort(np.lib.stride_tricks.sliding_window_view(coarse, 4),
                                  axis=1, kind="stable")
            unique, counts = np.unique(patterns, axis=0, return_counts=True)
            for pattern, count in zip(unique, counts):
                key = tuple(pattern)
                pattern_probabilities[key] = pattern_probabilities.get(key, 0.0) + count / len(patterns) / scale
        result[f"RCMSE_{scale:02d}"] = (
            float(-np.log(counts_next / counts_m)) if counts_m > 0 and counts_next > 0 else np.nan)
        probs = np.array(list(pattern_probabilities.values()))
        result[f"RCMPE_{scale:02d}"] = float(-np.sum(probs * np.log(probs)))
    return result


# --------------------------------------------------------------------------
# Traitement d'un enregistrement (exécuté dans un processus fils)
# --------------------------------------------------------------------------
def process_subject(task):
    base, sid, group, label, sig, fs, canaux, p = task
    if p.get("pretraite"):
        if fs != FS_CIBLE or sig is None or sig.ndim != 2 or sig.shape[0] != len(canaux) or not np.isfinite(sig).all():
            raise ValueError(f"{base}/{sid} : signal prétraité invalide ou fréquence incorrecte")
    else:
        sig = pretraiter(sig, fs, canaux, p["notch"])
    if p["duree_max"]:
        sig = sig[:, :int(round(p["duree_max"] * FS_CIBLE))]
    win, starts = make_windows(sig.shape[1], FS_CIBLE, p["fenetre"], p["chevauchement"])
    rows = []
    for w, s in enumerate(starts):
        seg = sig[:, s:s + win]
        for ch in range(seg.shape[0]):
            coefs = dwt_bands(seg[ch], FS_CIBLE, p["ondelette"])
            multiscale = entropies_multiechelle(seg[ch]) if p.get("multiechelle") else {}
            if p["caracteristiques_etendues"]:
                energies = np.array([np.sum(coef ** 2) for coef in coefs])
                if not np.isfinite(energies).all():
                    raise ValueError(f"{base}/{sid} : énergie DWT non finie")
                total = energies.sum()
                if not np.isfinite(total):
                    raise ValueError(f"{base}/{sid} : énergie DWT totale non finie")
            for j, (band, coef) in enumerate(zip(BANDS, coefs)):
                sd, vals = calculer_mesures(coef, p)
                if p["caracteristiques_etendues"]:
                    vals.update(mesures_etendues(coef))
                    vals["LogPower"] = float(np.log10(energies[j] + 1e-12))
                    vals["RelPower"] = float(energies[j] / total) if total > 0 else np.nan
                if j == 0:
                    vals.update(multiscale)
                rows.append(dict(dataset=base, subject_id=sid, group_validation=group, label=label, window=w,
                                 channel=canaux[ch], band=band, **{AMPLITUDE: sd}, **vals))
    return sid, rows, len(starts)


def run_dataset(base, args, p):
    manifest = Path(args.donnees) / "controle_qualite.json"
    if not manifest.is_file():
        raise ValueError(f"Prétraitement 02 incomplet ou absent : {manifest}")
    info = json.loads(manifest.read_text(encoding="utf-8"))
    if (base not in info.get("bases", []) or info.get("pretraitement", {}).get("fs_cible") != FS_CIBLE
            or info.get("pretraitement", {}).get("bandpass") != list(BANDPASS)):
        raise ValueError(f"{base} : prétraitement 02 absent ou incompatible dans {manifest}")
    p = {**p, "notch": info["pretraitement"].get("notch")}
    signaux, meta_all = charger_signaux(base, args.donnees, statut=None)
    keep = meta_all["statut"].eq("retenu")
    meta = meta_all.loc[keep].reset_index(drop=True)
    signaux = signaux_retenus = [s for s, k in zip(signaux, keep) if k]
    if "signal_pretraite_sha256" not in meta:
        raise ValueError(f"{base} : empreintes prétraitées absentes ; relancer 02.")
    for (_, row), sig in zip(meta.iterrows(), signaux):
        if sig is None:
            raise ValueError(f"{base}/{row['subject_id']} : signal prétraité absent.")
        fingerprint = hashlib.sha256(np.ascontiguousarray(sig, dtype=np.float64).tobytes()).hexdigest()
        if fingerprint != row["signal_pretraite_sha256"]:
            raise ValueError(f"{base}/{row['subject_id']} : empreinte prétraitée incohérente ; relancer 02.")
    canaux = canaux_sauvegardes(base, args.donnees)
    if args.limite:
        meta, signaux_retenus = meta.iloc[:args.limite], signaux[:args.limite]
    log(f"[{base}] {len(meta)} enregistrements, {len(canaux)} canaux, fenêtre {p['fenetre']} s, "
        f"chevauchement {p['chevauchement']}, {args.workers} processus")
    tasks = [(base, r["subject_id"], r["group_validation"], int(r["label"]), sig, float(r["fs"]), canaux, p)
             for (_, r), sig in zip(meta.iterrows(), signaux_retenus)]
    out = nouveau_dossier("extraction", prefixe=base)
    t0 = time.time()
    all_rows, n_win = [], {}
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, (sid, rows, nw) in enumerate(ex.map(process_subject, tasks), 1):
                all_rows.extend(rows)
                n_win[sid] = (nw, len(rows))
                log(f"[{base}] {i}/{len(tasks)} sujets traités ({time.time() - t0:.0f} s)")
    else:
        for i, t in enumerate(tasks, 1):
            sid, rows, nw = process_subject(t)
            all_rows.extend(rows)
            n_win[sid] = (nw, len(rows))
            log(f"[{base}] {i}/{len(tasks)} sujets traités ({time.time() - t0:.0f} s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "features_windows.csv", index=False)
    if not df.empty:
        feature_cols = MEASURES + MESURES_ETENDUES + PUISSANCES + MULTIECHELLE + [AMPLITUDE]
        feature_cols = [col for col in feature_cols if col in df]
        subjects = df.groupby(["dataset", "subject_id", "group_validation", "label", "channel", "band"],
                              as_index=False)[feature_cols].mean()
        subjects.to_csv(out / "features_sujets.csv", index=False)
    audit = meta_all.copy()
    audit["statut"] = audit["statut"].replace({"retenu": "extrait"})
    if args.limite:
        audit.loc[audit["statut"].eq("extrait") & ~audit["subject_id"].isin(n_win), "statut"] = "non_traite_limite"
    audit["nombre_fenetres"] = audit["subject_id"].map(lambda s: n_win.get(s, (0, 0))[0])
    audit["nombre_lignes"] = audit["subject_id"].map(lambda s: n_win.get(s, (0, 0))[1])
    sans_fenetre = audit["statut"].eq("extrait") & audit["nombre_fenetres"].eq(0)
    if sans_fenetre.any():
        log(f"  ! {int(sans_fenetre.sum())} enregistrement(s) trop court(s) pour une fenêtre : exclus")
        audit.loc[sans_fenetre, "statut"] = "trop_court"
    audit.to_csv(out / "audit_enregistrements.csv", index=False)
    measures = MEASURES + (MESURES_ETENDUES + PUISSANCES if p["caracteristiques_etendues"] else [])
    if args.large:
        large = df.pivot_table(index=["subject_id", "label", "window"], columns=["channel", "band"],
                               values=measures + [AMPLITUDE], aggfunc="first")
        large.columns = [f"{c}_{b}_{m}" for m, c, b in large.columns]
        large.reset_index().to_csv(out / "features_large.csv", index=False)
    n_nan = df[measures].isna().sum().to_dict()
    if p.get("multiechelle") and len(df):
        n_nan.update(df.loc[df["band"].eq(BANDS[0]), MULTIECHELLE].isna().sum().to_dict())
    ecrire_json(out / "parametres_extraction.json", {
        "base": base, "canaux": canaux, "fs_cible": FS_CIBLE, "bandes": BANDS, "mesures": measures,
        "mesures_multiechelle": MULTIECHELLE if p.get("multiechelle") else [],
        "puissances": "LogPower = log10(énergie DWT en µV² + 1e-12), RelPower = énergie / somme des 5 énergies",
        "parametres": p, "pretraitement_source": str(Path(args.donnees).resolve()),
        "enregistrements": int(audit["statut"].eq("extrait").sum()),
        "fenetres": int(df.groupby("subject_id")["window"].nunique().sum()) if len(df) else 0,
        "valeurs_manquantes": n_nan, "duree_s": round(time.time() - t0, 1),
        "arguments": arguments_dict(args), "versions": versions(("numpy", "scipy", "pandas", "PyWavelets", "EntropyHub"))})
    log(f"  -> {out.name}/features_windows.csv : {len(df)} lignes, "
        f"{df['subject_id'].nunique() if len(df) else 0} sujets, NaN par mesure {n_nan}")
    log(f"     fenêtres par classe : {df.groupby('label')['subject_id'].nunique().to_dict() if len(df) else {}} sujets")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="*", help=f"parmi {BASES} (défaut : toutes)")
    p.add_argument("--donnees", type=Path, default=None, help="Dossier de prétraitement 02 (défaut : le plus récent complet).")
    p.add_argument("--fenetre", type=float, default=10.0, help="Durée des fenêtres en secondes.")
    p.add_argument("--chevauchement", type=float, default=0.0, help="Chevauchement entre 0 et <1.")
    p.add_argument("--duree-max", type=float, default=None, help="Ne garder que les N premières secondes.")
    p.add_argument("--notch", type=float, default=None, help="Obsolète : configurer le coupe-bande dans 02.")
    p.add_argument("--ondelette", default="db4")
    p.add_argument("--m", type=int, default=2, help="Dimension de plongement (SampEn, FuzzyEn, DispEn).")
    p.add_argument("--r", type=float, default=0.2, help="Tolérance (fraction de l'écart-type).")
    p.add_argument("--n-fuzzy", type=float, default=2.0, help="Exposant de l'appartenance floue.")
    p.add_argument("--c-disp", type=int, default=6, help="Nombre de classes de DispEn.")
    p.add_argument("--entropyhub", action="store_true", help="Utiliser EntropyHub pour SampEn, FuzzyEn, DispEn.")
    p.add_argument("--caracteristiques-etendues", action="store_true",
                   help="Option conservée pour compatibilité : les caractéristiques sont toujours calculées.")
    p.add_argument("--large", action="store_true", help="Écrire aussi features_large.csv (une ligne par fenêtre).")
    p.add_argument("--limite", type=int, default=None, help="N premiers enregistrements seulement (essai).")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = p.parse_args()
    bad = [b for b in args.bases if b not in BASES]
    if bad:
        p.error(f"base(s) inconnue(s) : {bad}")
    if args.notch is not None:
        p.error("Le coupe-bande se règle en 02 ; relancer 02 avec --notch.")
    if args.donnees is None:
        args.donnees = dernier_dossier("qc", contient="controle_qualite.json")
    if args.donnees is None:
        p.error("Aucun prétraitement 02 complet ; lancer 02_controle_qualite.py.")
    try:
        import EntropyHub  # noqa: F401
    except ModuleNotFoundError:
        p.error("EntropyHub est requis pour ApEn et PermEn (pip install EntropyHub).")
    params = dict(fenetre=args.fenetre, chevauchement=args.chevauchement, duree_max=args.duree_max,
                  notch=args.notch, ondelette=args.ondelette, m=args.m, r=args.r, n_fuzzy=args.n_fuzzy,
                  c_disp=args.c_disp, entropyhub=args.entropyhub,
                  caracteristiques_etendues=True, multiechelle=True, pretraite=True, bandpass=list(BANDPASS))
    log(f"workers = {args.workers}")
    sorties = []
    for base in args.bases or BASES:
        try:
            sorties.append(run_dataset(base, args, params))
        except ValueError as exc:
            p.exit(1, f"[{base}] arrêté : {exc}\n")
    for s in sorties:
        log(f"SORTIE : {s}")


if __name__ == "__main__":
    main()
