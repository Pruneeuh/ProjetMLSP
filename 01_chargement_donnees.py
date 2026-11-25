"""Étape 1 : chargement des 3 bases EEG brutes, contrôle des doublons et sauvegarde.

Entrée  : data_eeg/mumtaz/*.edf          (dépression, Mumtaz 2016, 256 Hz, 19 électrodes gardées)
          data_eeg/borisov/norm.zip, sch.zip (schizophrénie, Borisov 2005, .eea, 16 × 7680, 128 Hz)
          data_eeg/sakib/*.mat           (anxiété, Sakib 2023, Emotiv EPOC+, 14 électrodes, 128 Hz)
Sortie  : sorties/01_donnees/<base>_signals.npz    signaux (canaux, échantillons) en µV
          sorties/01_donnees/<base>_metadata.csv   un fichier source par ligne :
              subject_id, fichier, label (0 témoin, 1 trouble), group_validation, statut,
              n_channels, n_samples, fs, duree_s, n_canaux_origine, unite, signal_sha256
          sorties/01_donnees/<base>_canaux.json    noms 10-10 des électrodes (ordre des lignes)
          sorties/01_donnees/inventaire_chargement.csv / .json

Statuts : retenu | doublon_exclu_de_analyse | illisible | canaux_manquants | taille_inattendue
Seuls les « retenus » sont dans le .npz.

Points d'attention :
  - Mumtaz : mne renvoie des volts → convertis en µV (comme les deux autres bases), sinon
    l'amplitude seule séparerait les bases. Le nombre de canaux du fichier d'origine
    (20 ou 22) est gardé : il est confondu avec le groupe (voir 02_controle_qualite.py).
  - Borisov et Sakib : l'ordre des électrodes est SUPPOSÉ (voir eeg_config.CANAUX).
  - Doublons : empreinte SHA-1 du signal ; le second fichier identique est exclu.
  - Groupes de validation : 387-02w1 et 387-03w1 (schizophrénie) dans le même groupe.

Usage :
    python 01_chargement_donnees.py                       # les 3 bases
    python 01_chargement_donnees.py depression anxiete    # un sous-ensemble
    python 01_chargement_donnees.py --donnees /chemin/data_eeg --condition EO

Dépendances : pip install numpy scipy pandas mne
"""
import argparse
import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io

from eeg_config import (BASES, CANAUX, DATA_ROOT, FS_ORIGINE, GROUPES_PRUDENTS, MUMTAZ_CANAUX_EDF,
                        MUMTAZ_CONDITION, ORDRE_SUPPOSE, SORTIES, SOUS_DOSSIER_BRUT, SOURCES, DOSSIERS)
from eeg_utils import ecrire_json, log, versions


def ligne(fichier, sid, label, statut, data=None, fs=None, n_orig=None, unite="µV", remarque=""):
    return dict(subject_id=sid, fichier=fichier, label=label, statut=statut,
                n_channels=None if data is None else data.shape[0],
                n_samples=None if data is None else data.shape[1],
                fs=fs, duree_s=None if data is None else round(data.shape[1] / fs, 2),
                n_canaux_origine=n_orig, unite=unite, remarque=remarque,
                signal_sha256=None if data is None else hashlib.sha256(
                    np.ascontiguousarray(data, dtype=np.float64).tobytes()).hexdigest())


# --------------------------------------------------------------------------
# Lecteurs
# --------------------------------------------------------------------------
def load_mumtaz(path: Path, condition: str = MUMTAZ_CONDITION):
    """Mumtaz (dépression), fichiers « H S1 EC.edf » / « MDD S1 EC.edf ». 256 Hz."""
    import mne
    rows, signals = [], {}
    pattern = re.compile(r"^(H|MDD)\s+S(\d+)\s+(EC|EO|TASK)\.edf$", re.I)
    fichiers = sorted(path.glob("*.edf"))
    if not fichiers:
        log(f"  ! Aucun fichier .edf dans {path}")
    for f in fichiers:
        m = pattern.match(re.sub(r"\s+", " ", f.name))
        if not m or m.group(3).upper() != condition.upper():
            continue
        group, num = m.group(1).upper(), m.group(2)
        sid = f"{group}_S{num}_{condition.upper()}"
        label = 0 if group == "H" else 1
        try:
            raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        except Exception as e:
            log(f"  ! Erreur de lecture {f.name} : {e}")
            rows.append(ligne(f.name, sid, label, "illisible", remarque=str(e)[:200]))
            continue
        names = {}
        for ch in raw.ch_names:  # ex. « EEG Fp1-LE » → « Fp1 »
            core = re.sub(r"^EEG\s*", "", ch).split("-")[0].strip()
            names[core.lower()] = ch
        picks = [names[c.lower()] for c in MUMTAZ_CANAUX_EDF if c.lower() in names]
        fs = float(raw.info["sfreq"])
        if len(picks) != len(MUMTAZ_CANAUX_EDF):
            log(f"  ! {f.name} : {len(picks)}/19 canaux trouvés, ignoré")
            rows.append(ligne(f.name, sid, label, "canaux_manquants", fs=fs, n_orig=len(raw.ch_names)))
            continue
        data = raw.copy().pick(picks).get_data() * 1e6   # volts → µV
        if abs(fs - FS_ORIGINE["depression"]) > 1e-6:
            log(f"  ! {f.name} : fs = {fs} Hz (256 attendu)")
        rows.append(ligne(f.name, sid, label, "retenu", data, fs, len(raw.ch_names)))
        signals[sid] = data
    return rows, signals


def load_borisov(zip_dir: Path):
    """Borisov (schizophrénie) : norm.zip / sch.zip de fichiers .eea texte,
    16 canaux × 7680 échantillons (128 Hz, 60 s) empilés verticalement."""
    n_ch, n_samp, fs = 16, 7680, FS_ORIGINE["schizophrenie"]
    rows, signals = [], {}
    for zname, label, prefix in (("norm.zip", 0, "HC"), ("sch.zip", 1, "SZ")):
        zp = zip_dir / zname
        if not zp.exists():
            log(f"  ! Fichier manquant : {zp}")
            continue
        with zipfile.ZipFile(zp) as zf:
            names = sorted(n for n in zf.namelist()
                           if n.lower().endswith((".eea", ".txt")) and not n.endswith("/"))
            for n in names:
                sid = f"{prefix}_{Path(n).stem}"
                try:
                    with zf.open(n) as fh:
                        data = np.loadtxt(fh, dtype=np.float64)
                except Exception as e:
                    rows.append(ligne(n, sid, label, "illisible", remarque=str(e)[:200]))
                    continue
                if data.size != n_ch * n_samp:
                    log(f"  ! Taille inattendue pour {n} : {data.size}, ignoré")
                    rows.append(ligne(n, sid, label, "taille_inattendue", remarque=f"{data.size} valeurs"))
                    continue
                data = data.reshape(n_ch, n_samp)
                rows.append(ligne(n, sid, label, "retenu", data, fs, n_ch))
                signals[sid] = data
    return rows, signals


def load_sakib(path: Path):
    """Sakib (anxiété) : .mat, ASub* = anxiété (1), ACSub* = témoin (0). 14 canaux, 128 Hz."""
    fs = FS_ORIGINE["anxiete"]
    rows, signals = [], {}
    fichiers = sorted(path.glob("*.mat"))
    if not fichiers:
        log(f"  ! Aucun fichier .mat dans {path}")
    for f in fichiers:
        sid = f.stem
        if sid.startswith("ACSub"):       # tester ACSub avant ASub
            label = 0
        elif sid.startswith("ASub"):
            label = 1
        else:
            log(f"  ! Nom non reconnu, ignoré : {sid}")
            continue
        try:
            mat = scipy.io.loadmat(f)
        except Exception as e:
            rows.append(ligne(f.name, sid, label, "illisible", remarque=str(e)[:200]))
            continue
        arrays = [v for k, v in mat.items()
                  if not k.startswith("__") and isinstance(v, np.ndarray)
                  and v.ndim == 2 and min(v.shape) > 1 and np.issubdtype(v.dtype, np.number)]
        if not arrays:
            log(f"  ! Aucun tableau EEG dans {sid}")
            rows.append(ligne(f.name, sid, label, "illisible", remarque="aucun tableau numérique 2D"))
            continue
        arr = max(arrays, key=lambda a: a.size)
        data = np.asarray(arr.T if arr.shape[0] > arr.shape[1] else arr, dtype=np.float64)  # (canaux, échantillons)
        if data.shape[0] != len(CANAUX["anxiete"]):
            log(f"  ! {sid} : {data.shape[0]} canaux (14 attendus), ignoré")
            rows.append(ligne(f.name, sid, label, "canaux_manquants", data, fs, data.shape[0]))
            continue
        rows.append(ligne(f.name, sid, label, "retenu", data, fs, data.shape[0]))
        signals[sid] = data
    return rows, signals


LECTEURS = {"depression": load_mumtaz, "schizophrenie": load_borisov, "anxiete": load_sakib}


# --------------------------------------------------------------------------
# Contrôles et sauvegarde
# --------------------------------------------------------------------------
def marquer_doublons(meta):
    """Le second fichier d'empreinte identique passe au statut doublon_exclu_de_analyse."""
    vus = {}
    for i, r in meta.iterrows():
        if r["statut"] != "retenu":
            continue
        h = r["signal_sha256"]
        if h in vus:
            log(f"  ! Doublon : {r['subject_id']} identique à {vus[h]}, exclu")
            meta.at[i, "statut"] = "doublon_exclu_de_analyse"
            meta.at[i, "remarque"] = f"identique à {vus[h]}"
            if meta.loc[meta["subject_id"] == vus[h], "label"].iloc[0] != r["label"]:
                log(f"  !! ATTENTION : doublon d'étiquette différente ({vus[h]} / {r['subject_id']})")
        else:
            vus[h] = r["subject_id"]
    return meta


def groupes_validation(base, ids):
    prudent = GROUPES_PRUDENTS.get(base, {})
    out = []
    for sid in ids:
        g = sid
        for motif, groupe in prudent.items():
            if motif in sid:
                g = groupe
        out.append(g)
    return out


def save_dataset(base, meta, signals, out_dir):
    keep = meta.loc[meta["statut"].eq("retenu"), "subject_id"].tolist()
    np.savez_compressed(out_dir / f"{base}_signals.npz", **{s: signals[s] for s in keep})
    meta.to_csv(out_dir / f"{base}_metadata.csv", index=False)
    ecrire_json(out_dir / f"{base}_canaux.json", {
        "base": base, "source": SOURCES[base], "canaux": CANAUX[base],
        "ordre_suppose": ORDRE_SUPPOSE[base], "fs": FS_ORIGINE[base], "unite": "µV"})
    log(f"  -> {base}_signals.npz : {len(keep)} enregistrements retenus")


def summarize(base, meta):
    r = meta.loc[meta["statut"].eq("retenu")]
    if r.empty:
        log(f"  {base} : aucune donnée")
        return {}
    counts = r["label"].value_counts().sort_index().to_dict()
    statuts = meta["statut"].value_counts().to_dict()
    log(f"  {base} : {len(r)} retenus (témoins {counts.get(0, 0)}, trouble {counts.get(1, 0)}), "
        f"{r['group_validation'].nunique()} groupes ; statuts {statuts}")
    log(f"    durée {r['duree_s'].min():.0f}-{r['duree_s'].max():.0f} s, canaux {sorted(int(v) for v in r['n_channels'].unique())}, "
        f"canaux d'origine {sorted(int(v) for v in r['n_canaux_origine'].dropna().unique())}")
    return dict(retenus=len(r), temoins=int(counts.get(0, 0)), trouble=int(counts.get(1, 0)),
                groupes=int(r["group_validation"].nunique()), statuts=statuts,
                duree_s=[float(r["duree_s"].min()), float(r["duree_s"].max())])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="*", help=f"parmi {BASES} (défaut : toutes)")
    p.add_argument("--donnees", type=Path, default=DATA_ROOT, help="Dossier des données brutes (défaut data_eeg/).")
    p.add_argument("--sortie", type=Path, default=SORTIES / DOSSIERS["donnees"])
    p.add_argument("--condition", default=MUMTAZ_CONDITION, choices=["EC", "EO", "TASK"],
                   help="Condition Mumtaz (défaut EC, yeux fermés).")
    args = p.parse_args()
    bad = [b for b in args.bases if b not in BASES]
    if bad:
        p.error(f"base(s) inconnue(s) : {bad}")
    args.sortie.mkdir(parents=True, exist_ok=True)
    inventaire, resume = [], {}
    for i, base in enumerate(args.bases or BASES, 1):
        src = args.donnees / SOUS_DOSSIER_BRUT[base]
        log(f"[{i}/{len(args.bases or BASES)}] {base} ({SOURCES[base]}) depuis {src}")
        if base == "depression":
            rows, signals = LECTEURS[base](src, args.condition)
        else:
            rows, signals = LECTEURS[base](src)
        if not rows:
            log(f"  (rien à sauvegarder pour {base})")
            continue
        meta = pd.DataFrame(rows)
        meta = marquer_doublons(meta)
        meta.insert(3, "group_validation", groupes_validation(base, meta["subject_id"]))
        meta.insert(0, "dataset", base)
        resume[base] = summarize(base, meta)
        save_dataset(base, meta, signals, args.sortie)
        inventaire.append(meta)
    if inventaire:
        pd.concat(inventaire, ignore_index=True).to_csv(args.sortie / "inventaire_chargement.csv", index=False)
        ecrire_json(args.sortie / "inventaire_chargement.json",
                    {"resume": resume, "arguments": vars(args), "versions": versions(("numpy", "scipy", "pandas", "mne"))})
    log(f"\nSORTIE : {args.sortie}")


if __name__ == "__main__":
    main()
