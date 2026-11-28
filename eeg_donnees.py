"""Lecture des données intermédiaires du pipeline (module partagé, non exécutable).

  - charger_signaux(base)            : signaux bruts sauvegardés par 01_chargement_donnees.py
  - charger_extraction(dossier, base): entropies par fenêtre (03_extraction_entropies.py ou
                                       ancien extraction_corrigee.py : même format)
  - agreger(info, canaux)            : une ligne par sujet (fenêtres agrégées)
  - construire_fusion(...)           : 3 bases sur les canaux communs, classes 0..3
  - charger_fusion(dossier)          : matrices produites par 04_fusion_harmonisation.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_config import (ALIASES, AMPLITUDE, BANDS, BASES, CANAUX, CLASS_OF_BASE, DOSSIERS, MEASURES,
                        MESURES_ETENDUES, ORDRE_PREFERE, PUISSANCES, SORTIES)
from eeg_utils import dernier_dossier, require


# --------------------------------------------------------------------------
# Étape 01 : signaux bruts
# --------------------------------------------------------------------------
def dossier_donnees(racine=None):
    return Path(racine or SORTIES) / DOSSIERS["donnees"]


def charger_signaux(base, dossier=None, statut="retenu"):
    """Renvoie (signaux, metadata) ; signaux = liste de tableaux (canaux, échantillons) dans
    l'ordre de metadata (lignes au statut demandé)."""
    dossier = Path(dossier or dossier_donnees())
    meta_path, npz_path = dossier / f"{base}_metadata.csv", dossier / f"{base}_signals.npz"
    require(meta_path.exists() and npz_path.exists(),
            f"{base} : {meta_path.name} ou {npz_path.name} introuvable dans {dossier} "
            "(lancer 01_chargement_donnees.py).")
    meta = pd.read_csv(meta_path)
    if statut is not None:
        meta = meta.loc[meta["statut"].eq(statut)].reset_index(drop=True)
    with np.load(npz_path) as npz:
        present = set(npz.files)
        signaux = [npz[s] if s in present else None for s in meta["subject_id"]]   # None : non sauvegardé
    return signaux, meta


def canaux_sauvegardes(base, dossier=None):
    path = Path(dossier or dossier_donnees()) / f"{base}_canaux.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["canaux"]
    return list(CANAUX[base])


# --------------------------------------------------------------------------
# Étape 03 : entropies par fenêtre
# --------------------------------------------------------------------------
def dossier_extraction_defaut(base, racine=None):
    """Extraction la plus récente de la base dans sorties/03_extraction/."""
    return dernier_dossier("extraction", prefixe=f"{base}_", contient="features_windows.csv", racine=racine)


def channel_map(base, channels, orders=None):
    """Nom dans l'extraction → nom 10-10. Les colonnes numérotées col01... (ancien
    extraction_corrigee.py) sont converties avec l'ordre supposé de la base."""
    orders = orders or CANAUX
    mapping = {}
    for ch in channels:
        name = str(ch)
        if name.startswith("col"):
            require(base in orders, f"{base} : canaux numérotés sans ordre connu (--carte-canaux).")
            idx = int(name[3:]) - 1
            require(0 <= idx < len(orders[base]), f"{base} : {name} hors de l'ordre fourni.")
            name = orders[base][idx]
        mapping[ch] = ALIASES.get(name, name)
    require(len(set(mapping.values())) == len(mapping), f"{base} : noms de canaux en double après correspondance.")
    return mapping


def charger_extraction(run, expected, orders=None, aggregation="moyenne"):
    """Lit features_windows.csv + audit_enregistrements.csv et vérifie leur cohérence."""
    run = Path(run)
    require(run.is_dir(), f"Dossier d'extraction introuvable : {run}")
    audit = pd.read_csv(run / "audit_enregistrements.csv")
    data = pd.read_csv(run / "features_windows.csv")
    need = {"subject_id", "group_validation", "label", "channel", "band", "window", "dataset", AMPLITUDE, *MEASURES}
    require(need <= set(data), f"{expected} : colonnes manquantes {sorted(need - set(data))}.")
    require({"subject_id", "group_validation", "label", "statut"} <= set(audit),
            f"{expected} : colonnes manquantes dans l'audit.")
    found = set(data["dataset"])
    require(found == {expected}, f"Le dossier {run.name} contient {found}, {expected} attendu.")
    require(not audit["subject_id"].duplicated().any(), f"{expected} : identifiant répété dans l'audit.")
    records = audit.loc[audit["statut"].eq("extrait")].set_index("subject_id").sort_index()
    require(set(records["label"]) == {0, 1}, f"{expected} : étiquettes hors 0/1.")
    require(set(data["subject_id"]) == set(records.index), f"{expected} : enregistrements ≠ audit.")
    require(set(data["band"]) == set(BANDS), f"{expected} : composantes inattendues {sorted(set(data['band']))}.")
    require(not data.duplicated(["subject_id", "window", "channel", "band"]).any(),
            f"{expected} : lignes fenêtre/canal/composante répétées.")
    for col in ["label", "group_validation"]:
        require(data[col].eq(data["subject_id"].map(records[col])).all(), f"{expected} : désaccord {col}.")
    sets = data.groupby("subject_id")["channel"].agg(frozenset)
    require(sets.nunique() == 1, f"{expected} : canaux différents selon les enregistrements.")
    original = sorted(sets.iloc[0], key=str)
    mapping = channel_map(expected, original, orders)
    data["canal"] = data["channel"].map(mapping)
    for col in MEASURES + [AMPLITUDE] + [c for c in MESURES_ETENDUES + PUISSANCES if c in data]:
        data[col] = pd.to_numeric(data[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    data["log_sd"] = np.log(data[AMPLITUDE].where(data[AMPLITUDE] > 0))
    canaux = sorted(set(mapping.values()), key=lambda c: ORDRE_PREFERE.index(c) if c in ORDRE_PREFERE else 99)
    return dict(base=expected, run=run, audit=audit, records=records, data=data, mapping=mapping,
                channels=canaux, aggregation=aggregation)


def agreger(info, canaux, mesures=None):
    """Agrège les fenêtres par sujet. Renvoie (entropies, amplitude) : DataFrames indexés par
    subject_id, colonnes « canal|bande|mesure » (ordre canal > bande > mesure) et
    « log_sd|canal|bande »."""
    mesures = MEASURES if mesures is None else mesures
    missing = set(mesures) - set(info["data"].columns)
    require(not missing, f"{info['base']} : caractéristiques absentes {sorted(missing)} ; "
                         "relancer 03 avec --caracteristiques-etendues.")
    data = info["data"].loc[info["data"]["canal"].isin(canaux)]
    keys = ["subject_id", "canal", "band"]
    how = "mean" if info["aggregation"] == "moyenne" else "median"
    ent = data.groupby(keys)[mesures].agg(how).unstack(["canal", "band"])
    ent = ent.reorder_levels([1, 2, 0], axis=1)
    cols = [(c, b, m) for c in canaux for b in BANDS for m in mesures]
    ent = ent.reindex(columns=pd.MultiIndex.from_tuples(cols)).reindex(info["records"].index)
    ent.columns = ["|".join(c) for c in ent.columns]
    amp = data.groupby(keys)["log_sd"].agg(how).unstack(["canal", "band"])
    amp = amp.reindex(columns=pd.MultiIndex.from_tuples([(c, b) for c in canaux for b in BANDS]))
    amp = amp.reindex(info["records"].index)
    amp.columns = ["log_sd|" + "|".join(c) for c in amp.columns]
    return ent, amp


def lire_carte_canaux(path):
    orders = dict(CANAUX)
    if path:
        custom = json.loads(Path(path).read_text(encoding="utf-8"))
        require(set(custom) <= set(BASES), "Clés de --carte-canaux : depression, schizophrenie, anxiete.")
        orders.update(custom)
    return orders


def ajouter_arguments_extraction(p):
    for b in BASES:
        p.add_argument(f"--{b}", type=Path, default=None,
                       help=f"Dossier d'extraction {b} (défaut : le plus récent de sorties/03_extraction).")
    p.add_argument("--carte-canaux", type=Path,
                   help='JSON {"schizophrenie": [16 noms], "anxiete": [14 noms]} remplaçant l\'ordre supposé.')
    p.add_argument("--agregation", choices=["moyenne", "mediane"], default="moyenne",
                   help="Agrégation des fenêtres par sujet (défaut : moyenne).")


def dossiers_extraction(args):
    """Dossiers d'extraction choisis (arguments --depression ... ou dernière extraction)."""
    runs = {}
    for b in BASES:
        p = getattr(args, b, None)
        p = Path(p).expanduser().resolve() if p else dossier_extraction_defaut(b)
        require(p is not None, f"Aucune extraction trouvée pour {b} dans sorties/03_extraction "
                               f"(lancer 03_extraction_entropies.py ou passer --{b} <dossier>).")
        runs[b] = p
    return runs


def construire_fusion(runs, orders=None, aggregation="moyenne"):
    """Fusion des 3 bases sur les canaux communs. Renvoie infos, common, ent, amp, meta.
    meta : base, label_binaire, classe (0 témoin, 1 dépression, 2 schizophrénie, 3 anxiété),
    groupe (base::group_validation), n_fenetres."""
    infos = {b: charger_extraction(runs[b], b, orders, aggregation) for b in BASES}
    common = [c for c in infos["depression"]["channels"] if all(c in infos[b]["channels"] for b in BASES)]
    common = sorted(common, key=lambda c: ORDRE_PREFERE.index(c) if c in ORDRE_PREFERE else 99)
    require(len(common) >= 4, f"Seulement {len(common)} canaux communs : {common}.")
    ent_parts, amp_parts, meta_parts = [], [], []
    for b in BASES:
        ent, amp = agreger(infos[b], common)
        rec = infos[b]["records"]
        nwin = infos[b]["data"].groupby("subject_id")["window"].nunique().reindex(rec.index)
        meta = pd.DataFrame({
            "base": b,
            "label_binaire": rec["label"].astype(int),
            "classe": np.where(rec["label"].eq(1), CLASS_OF_BASE[b], 0),
            "groupe": b + "::" + rec["group_validation"].astype(str),
            "n_fenetres": nwin.to_numpy(),
        }, index=rec.index)
        ent_parts.append(ent)
        amp_parts.append(amp)
        meta_parts.append(meta)
    ent, amp, meta = pd.concat(ent_parts), pd.concat(amp_parts), pd.concat(meta_parts)
    meta.index.name = ent.index.name = amp.index.name = "subject_id"
    require(ent.index.is_unique, "Identifiants de sujets en double entre bases.")
    require(not ent.isna().all(axis=1).any(), "Un enregistrement n'a aucune caractéristique exploitable.")
    require(meta.groupby("groupe")["classe"].nunique().max() == 1, "Groupe à étiquettes mixtes.")
    return infos, common, ent, amp, meta


# --------------------------------------------------------------------------
# Étape 04 : fusion
# --------------------------------------------------------------------------
def dossier_fusion_defaut(racine=None):
    return dernier_dossier("fusion", contient="entropies_sujets.csv", racine=racine)


def charger_fusion(dossier=None):
    """Renvoie (common, ent, amp, meta, infos_json) depuis un dossier de 04_fusion."""
    dossier = Path(dossier) if dossier else dossier_fusion_defaut()
    require(dossier is not None and (dossier / "entropies_sujets.csv").exists(),
            "Aucune fusion trouvée (lancer 04_fusion_harmonisation.py ou passer --fusion <dossier>).")
    meta = pd.read_csv(dossier / "meta_sujets.csv", index_col="subject_id")
    ent = pd.read_csv(dossier / "entropies_sujets.csv", index_col="subject_id").reindex(meta.index)
    amp = pd.read_csv(dossier / "amplitude_sujets.csv", index_col="subject_id").reindex(meta.index)
    info = json.loads((dossier / "fusion.json").read_text(encoding="utf-8"))
    return info["canaux_communs"], ent, amp, meta, info, dossier


def caracteristiques_fusion(common, ent, meta, dossier, choix):
    """Ajoute les colonnes étendues demandées, avec les mêmes contrôles pour 06 et 09."""
    if choix == "classiques":
        return ent
    files = [("entropies_etendues_sujets.csv", MESURES_ETENDUES)]
    if choix == "etendues_puissances":
        files.append(("puissances_sujets.csv", PUISSANCES))
    for name, measures in files:
        path = dossier / name
        require(path.is_file(), f"{path} absent ; relancer 04 avec --caracteristiques-etendues.")
        extra = pd.read_csv(path, index_col="subject_id")
        require(extra.index.is_unique and set(extra.index) == set(meta.index),
                f"{path} : sujets incohérents avec la fusion.")
        columns = [f"{c}|{b}|{m}" for c in common for b in BANDS for m in measures]
        require(extra.columns.is_unique and set(extra.columns) == set(columns),
                f"{path} : colonnes de caractéristiques incohérentes avec la fusion.")
        extra = extra[columns].apply(pd.to_numeric, errors="raise")
        ent = pd.concat([ent, extra.reindex(meta.index)], axis=1)
    return ent


def forme_tenseur(colonnes):
    """(canaux, bandes, mesures) d'une liste de colonnes « canal|bande|mesure »."""
    canaux = list(dict.fromkeys(c.split("|")[0] for c in colonnes))
    require(len(colonnes) == len(canaux) * len(BANDS) * len(MEASURES), "Colonnes inattendues.")
    return canaux, (len(canaux), len(BANDS), len(MEASURES))


def decomposer_colonnes(colonnes):
    """DataFrame canal, bande, mesure pour chaque colonne « canal|bande|mesure »."""
    parts = [c.split("|") for c in colonnes]
    return pd.DataFrame(parts, columns=["canal", "bande", "mesure"], index=colonnes)
