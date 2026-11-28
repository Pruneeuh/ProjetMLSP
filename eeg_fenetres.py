"""Matrices de fenêtres et partition indépendante par groupe de validation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from eeg_config import BANDS, BASES, GRAINE, MEASURES, MESURES_ETENDUES, MULTIECHELLE, PUISSANCES
from eeg_donnees import dossier_fusion_defaut
from eeg_utils import make_folds, require


META = ["subject_id", "window", "base", "label_binaire", "classe", "groupe", "partition",
        "n_canaux_origine"]


def partition_sujets(meta, seed=GRAINE):
    """Stratifier les groupes, et non les fenêtres ni les enregistrements liés."""
    for column in ["base", "label_binaire"]:
        require(meta.groupby("groupe")[column].nunique().eq(1).all(),
                f"Un groupe de validation possède plusieurs valeurs de {column}.")
    groups = meta.groupby("groupe", sort=True).agg(base=("base", "first"),
                                                   label=("label_binaire", "first"))
    strata = groups["base"].astype(str) + "::" + groups["label"].astype(str)
    train, test = train_test_split(groups.index.to_numpy(), test_size=0.2, random_state=seed,
                                   stratify=strata)
    result = pd.Series("train", index=meta.index, name="partition")
    result.loc[meta["groupe"].isin(test)] = "test"
    require(not set(train).intersection(test), "Groupes partagés entre train et test.")
    return result


def matrice_fenetres(info, canaux, meta, etendues=True):
    """Pivoter les fenêtres sans agréger ni éliminer de fenêtres incomplètes."""
    data = info["data"].loc[info["data"]["canal"].isin(canaux)]
    sizes = data.groupby(["subject_id", "window"], sort=True).size()
    require(sizes.eq(len(canaux) * len(BANDS)).all(),
            f"{info['base']} : fenêtre incomplète (canal ou bande manquant).")
    measures = list(MEASURES) + (MESURES_ETENDUES + PUISSANCES if etendues else [])
    needed = measures + (MULTIECHELLE if etendues else []) + ["log_sd"]
    missing = set(needed) - set(data)
    require(not missing, f"{info['base']} : caractéristiques absentes {sorted(missing)} ; relancer 03.")
    wide = data.pivot(index=["subject_id", "window"], columns=["canal", "band"], values=measures)
    wide = wide.reorder_levels([1, 2, 0], axis=1)
    columns = [(c, b, m) for c in canaux for b in BANDS for m in measures]
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(columns))
    wide.columns = ["|".join(c) for c in wide.columns]
    amplitude = data.pivot(index=["subject_id", "window"], columns=["canal", "band"], values="log_sd")
    amplitude = amplitude.reindex(columns=pd.MultiIndex.from_product([canaux, BANDS]))
    amplitude.columns = ["log_sd|" + "|".join(c) for c in amplitude.columns]
    parts = [wide, amplitude]
    if etendues:
        multi = data.loc[data["band"].eq(BANDS[0])].pivot(
            index=["subject_id", "window"], columns="canal", values=MULTIECHELLE)
        multi = multi.reorder_levels([1, 0], axis=1)
        multi = multi.reindex(columns=pd.MultiIndex.from_product([canaux, MULTIECHELLE]))
        multi.columns = [f"{c}|large_bande|{m}" for c, m in multi.columns]
        parts.append(multi)
    result = pd.concat(parts, axis=1).reset_index()
    meta = meta.copy()
    rec = info["records"]
    meta["n_canaux_origine"] = rec["n_canaux_origine"] if "n_canaux_origine" in rec else np.nan
    result = result.merge(meta.reset_index(), on="subject_id", how="left", validate="many_to_one")
    require(result["partition"].notna().all(), f"{info['base']} : partition absente.")
    return result[META + [c for c in result.columns if "|" in c]]


def charger_fenetres(dossier=None, base=None, choix="etendues_puissances", canaux_communs=False):
    dossier = Path(dossier) if dossier else dossier_fusion_defaut()
    require(dossier is not None, "Aucune fusion trouvée ; lancer 04.")
    name = f"fenetres_{base}.csv" if base else "fenetres_communes.csv"
    path = dossier / name
    require(path.is_file(), f"{path} absent ; relancer 04 pour produire les fenêtres et la partition.")
    info = json.loads((dossier / "fusion.json").read_text(encoding="utf-8"))
    data = pd.read_csv(path, dtype={"subject_id": str, "groupe": str, "partition": str})
    require(set(META) <= set(data), f"{path} : métadonnées incomplètes.")
    require(not data.duplicated(["subject_id", "window"]).any(), f"{path} : fenêtres en double.")
    require(set(data["partition"]) == {"train", "test"}, f"{path} : partition train/test invalide.")
    for col in ["partition", "base", "label_binaire", "classe"]:
        require(data.groupby("groupe")[col].nunique().eq(1).all(), f"{path} : {col} incohérent par groupe.")
    saved = pd.read_csv(dossier / "partition_sujets.csv", dtype={"subject_id": str, "groupe": str})
    require(not saved["subject_id"].duplicated().any(), "Partition : sujets répétés.")
    expected_subjects = saved if base is None else saved.loc[saved["base"].eq(base)]
    require(set(data["subject_id"]) == set(expected_subjects["subject_id"]),
            f"{path} : sujets incohérents avec la partition sauvegardée.")
    joined = data[META].merge(saved[["subject_id", "groupe", "partition", "base", "label_binaire", "classe"]], on="subject_id",
                              suffixes=("", "_saved"), validate="many_to_one", how="left")
    for col in ["groupe", "partition", "base", "label_binaire", "classe"]:
        require(joined[col].eq(joined[f"{col}_saved"]).all(),
                f"{path} : désaccord {col} avec la partition sauvegardée.")
    channels = info["canaux_communs"] if base is None or canaux_communs else info["canaux_par_base"][base]
    measures = list(MEASURES)
    if choix != "classiques":
        measures += MESURES_ETENDUES
    if choix == "etendues_puissances":
        measures += PUISSANCES
    cols = [f"{c}|{b}|{m}" for c in channels for b in BANDS for m in measures]
    if choix != "classiques":
        cols += [f"{c}|large_bande|{m}" for c in channels for m in MULTIECHELLE]
    amp = [f"log_sd|{c}|{b}" for c in channels for b in BANDS]
    require(set(cols + amp) <= set(data), f"{path} : caractéristiques demandées absentes ; relancer 03 et 04.")
    return data[cols].apply(pd.to_numeric, errors="raise"), data[amp].apply(pd.to_numeric, errors="raise"), data[META], info, dossier


def moyenner_sujets(meta, y, proba):
    """Moyenne des probabilités par enregistrement, jamais par fenêtre indépendante."""
    meta = meta.copy()
    if "subject_id" not in meta:
        meta["subject_id"] = meta.index
    tmp = meta[["subject_id", "groupe", "base"]].reset_index(drop=True).copy()
    tmp["classe"] = np.asarray(y)
    pcols = [f"p{i}" for i in range(proba.shape[1])]
    tmp[pcols] = proba
    for col in ["groupe", "base", "classe"]:
        require(tmp.groupby("subject_id")[col].nunique().eq(1).all(),
                f"Valeur {col} incohérente entre fenêtres d'un sujet.")
    agg = tmp.groupby("subject_id", sort=True).agg(
        groupe=("groupe", "first"), base=("base", "first"), classe=("classe", "first"),
        **{c: (c, "mean") for c in pcols})
    return agg, agg[pcols].to_numpy()


def plis_groupes(strata, groups, n_splits, seed, min_par_strate=2):
    """Construire les plis sur les groupes uniques avant de développer en fenêtres."""
    groups, strata = np.asarray(groups), np.asarray(strata)
    frame = pd.DataFrame({"groupe": groups, "strate": strata})
    require(frame.groupby("groupe")["strate"].nunique().eq(1).all(), "Groupe à strates mixtes.")
    unique = frame.drop_duplicates("groupe").sort_values("groupe")
    group_values = unique["groupe"].to_numpy()
    group_strata = unique["strate"].to_numpy()
    folds = make_folds(group_strata, group_values, n_splits, seed,
                       must_have=group_strata, min_par_strate=min_par_strate)
    return [(np.flatnonzero(np.isin(groups, group_values[tr])),
             np.flatnonzero(np.isin(groups, group_values[te]))) for tr, te in folds]
