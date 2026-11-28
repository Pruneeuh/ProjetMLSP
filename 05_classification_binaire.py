"""Étape 5 : classification binaire trouble contre témoins, base par base.

Entrée  : fusion de l'étape 04 (--fusion), fenêtres et partition groupée persistée
Sortie  : sorties/05_binaire/<date>/
    metriques_par_repetition.csv, resume.csv, predictions_test.csv
    ablation.csv               (--ablation) régression logistique sur une seule mesure / une seule bande
    permutations.csv           (--permutations N) test de permutation au niveau des sujets
    modeles/ et modeles.json   modèles principaux ajustés et provenance pour l'étape 09
    roc_<base>.png, confusion_<base>.png
    parametres.json

Pour chaque base : une ligne par fenêtre, tous les canaux de la
base (19 / 16 / 14) ou les 10 canaux communs (--canaux-communs). Modèles (--modeles) :
    reference_majoritaire   classe la plus fréquente
    regression_logistique   classes pondérées (L1/L2 réglées par recherche)
    mlp, svm, svm_rbf, knn, foret_aleatoire, lda, xgboost, lightgbm
Références de confusion (ajoutées d'office) :
    reference_amplitude       régression logistique sur log(écart-type avant standardisation) seul
    reference_canaux_origine  nombre de canaux du fichier d'origine seul (dépression : 20 ou 22)
Validation : partition 80/20 par groupe de l'étape 04, partagée avec 06. Les probabilités des
fenêtres test sont moyennées par sujet. Les répétitions utilisent le même test et changent la graine.
Recherche aléatoire en plis groupés sur train seulement (sauf référence majoritaire).
Les ablations et permutations utilisent une validation croisée sur train seulement.
--sans-tuning désactive la recherche ; --profil-tuning quick permet un essai rapide.
--caracteristiques : classiques, etendues, ou etendues_puissances (défaut).
Les jeux étendus exigent les caractéristiques correspondantes dans l'extraction 03.

Usage :
    python 05_classification_binaire.py
    python 05_classification_binaire.py --bases depression --canaux-origine 20
    python 05_classification_binaire.py --repetitions 10 --ablation --permutations 200
    python 05_classification_binaire.py --profil-tuning quick --modeles regression_logistique svm

Dépendances : numpy pandas scipy scikit-learn matplotlib xgboost lightgbm
"""
import argparse
import sys
import time
from pathlib import Path

RAPIDS_ACTIF = False
if __name__ == "__main__" and "--rapids" in sys.argv and "--cpu" not in sys.argv:
    from gpu_setup import enable_cuml_accel
    RAPIDS_ACTIF = enable_cuml_accel()

from eeg_tuning import detect_devices, predict_proba, tune_and_fit

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from eeg_config import BASES, CLASSES_BINAIRES, GRAINE, NOM_TROUBLE
from eeg_donnees import dossier_fusion_defaut
from eeg_fenetres import charger_fenetres, moyenner_sujets, plis_groupes
from eeg_modeles_sauvegardes import ecrire_manifeste, empreintes_fusion, sauvegarder_modele
from eeg_reprise import Reprise
from eeg_utils import (Compteur, arguments_dict, ecrire_json, entete, fmt_table, log,
                       multiclass_scores, nouveau_dossier, p_valeur_permutation,
                       plot_confusion, plot_roc, require, scores_binaires, summarise, versions,
                       xgboost_disponible)

MODELES = ["reference_majoritaire", "regression_logistique", "mlp", "svm", "svm_rbf",
           "knn", "foret_aleatoire", "lda", "xgboost", "lightgbm"]
DEFAUT = [m for m in MODELES if m != "svm_rbf"]


def cv_proba(nom, x, y, groups, folds, seed, args, modeles=None):
    proba = np.zeros((len(y), 2))
    fold_id = np.zeros(len(y), dtype=int)
    choix = []
    for k, (tr, te) in enumerate(folds, 1):
        est, result = tune_and_fit(nom, x[tr], y[tr], groups[tr], y[tr], seed,
                                   args.devices, args.profil_tuning, args.plis_internes,
                                   args.tirages_tuning, enabled=not args.sans_tuning,
                                   fixed_c=args.c, fixed_mlp_layers=args.mlp_couches,
                                   fixed_mlp_alpha=args.mlp_alpha)
        est.fit(x[tr], y[tr])
        if modeles is not None:
            modeles.append(est)
        if result["best_params"]:
            choix.append(result)
        proba[te] = predict_proba(est, x[te], 2)
        fold_id[te] = k
    return proba, fold_id, choix


def analyser_base(base, args, collect):
    ent, amp, meta, info, dossier = charger_fenetres(args.fusion, base, args.caracteristiques,
                                                     args.canaux_communs)
    if args.canaux_origine:
        require(meta["n_canaux_origine"].notna().all(), "--canaux-origine : métadonnées absentes.")
        keep = meta["n_canaux_origine"].eq(args.canaux_origine)
        ent, amp, meta = ent.loc[keep].reset_index(drop=True), amp.loc[keep].reset_index(drop=True), meta.loc[keep].reset_index(drop=True)
    y = meta["label_binaire"].astype(int).to_numpy()
    groups = meta["groupe"].to_numpy()
    train = np.flatnonzero(meta["partition"].eq("train"))
    test = np.flatnonzero(meta["partition"].eq("test"))
    require(len(train) and len(test) and set(y[train]) == {0, 1} and set(y[test]) == {0, 1},
            f"{base} : la partition après filtrage doit contenir les deux classes dans train et test.")
    inputs = {"entropies": ent}
    if not args.sans_reference_amplitude:
        inputs["amplitude"] = amp
    origin = meta["n_canaux_origine"]
    if origin.notna().all() and origin.iloc[train].nunique() > 1:
        inputs["canaux_origine"] = origin.eq(origin.iloc[train].max()).astype(int).to_frame("canaux_max")
    canaux = info["canaux_communs"] if args.canaux_communs else info["canaux_par_base"][base]
    log(f"\n=== {base} ({NOM_TROUBLE[base]}) : {len(y)} fenêtres, "
        f"{meta.subject_id.nunique()} sujets, {len(canaux)} canaux, {ent.shape[1]} variables")
    specs = [(m, "entropies") for m in args.modeles]
    if "amplitude" in inputs:
        specs.append(("reference_amplitude", "amplitude"))
    if "canaux_origine" in inputs:
        specs.append(("reference_canaux_origine", "canaux_origine"))
        log("  Référence du nombre de canaux d'origine ajoutée")
    task = f"binaire_{base}"
    folds_by_rep = [[(train, test)] for _ in range(args.repetitions)]
    for nom, feat in specs:
        t0 = time.time()
        x = inputs[feat].to_numpy(float)
        kind = "regression_logistique" if nom.startswith("reference_") and nom != "reference_majoritaire" else nom
        for rep, folds in enumerate(folds_by_rep):
            seed = args.graine + rep
            checkpoint = collect["reprise"]
            cle = ["principal", task, nom, feat, rep + 1]
            saved = checkpoint.charger(cle)
            if saved is None:
                modeles = [] if feat == "entropies" and nom != "reference_majoritaire" else None
                proba, _, choix = cv_proba(kind, x, y, groups, folds, seed, args, modeles)
                record = (sauvegarder_modele(collect["out"], task, nom, "aucune", rep + 1, modeles[0])
                          if modeles is not None else None)
                saved = (proba, choix, record)
                checkpoint.enregistrer(cle, saved, [record["fichier"]] if record else [])
            proba, choix, record = saved
            if record is not None:
                collect["modeles"].append(record)
            participants, p = moyenner_sujets(meta.iloc[test], y[test], proba[test])
            yt = participants["classe"].to_numpy()
            row, pred = multiclass_scores(yt, p, [0, 1])
            row.update(scores_binaires(yt, pred))
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
            row.update(vrais_temoins=tn, faux_trouble=fp, faux_temoins=fn, vrais_trouble=tp)
            if choix:
                row["hyperparametres_choisis"] = "; ".join(str(c) for c in choix)
            collect["rows"].append(dict(tache=task, base=base, modele=nom, caracteristiques=feat,
                                        repetition=rep + 1, n_variables=x.shape[1], **row))
            collect["preds"].append(pd.DataFrame({
                "tache": task, "harmonisation": "aucune", "modele": nom, "repetition": rep + 1,
                "subject_id": participants.index, "groupe": participants["groupe"], "pli": 1,
                "classe": yt, "prediction": pred, "proba_temoin": p[:, 0], "proba_trouble": p[:, 1]}))
        sub = pd.DataFrame([r for r in collect["rows"] if r["tache"] == task and r["modele"] == nom])
        log(f"  {nom:<26s} exact. équilibrée {sub.exactitude_equilibree.mean():.3f} ± "
            f"{sub.exactitude_equilibree.std(ddof=1) if len(sub) > 1 else 0:.3f}, AUC {sub.auc.mean():.3f}  "
            f"({time.time() - t0:.0f} s)")

    if args.ablation:
        log("  Ablation sur les seuls sujets d'entraînement (validation croisée groupée) :")
        folds_train = [plis_groupes(y[train], groups[train], args.plis, args.graine + r)
                       for r in range(args.repetitions)]
        cols = pd.Series(ent.columns)
        mesures = list(dict.fromkeys(c.rsplit("|", 1)[-1] for c in ent.columns))
        groupes_var = {f"mesure={m}": cols.str.endswith("|" + m) for m in mesures}
        bandes = list(dict.fromkeys(c.split("|")[1] for c in ent.columns))
        groupes_var.update({f"bande={b}": cols.str.contains(f"|{b}|", regex=False) for b in bandes})
        groupes_var.update({f"sans_mesure={m}": ~cols.str.endswith("|" + m) for m in mesures})
        for nom_g, mask in groupes_var.items():
            cle = ["ablation", base, nom_g]
            saved = collect["reprise"].charger(cle)
            if saved is not None:
                collect["ablation"].append(saved)
                continue
            x = ent.loc[:, mask.to_numpy()].to_numpy(float)[train]
            vals = []
            for rep, folds in enumerate(folds_train):
                proba, _, _ = cv_proba("regression_logistique", x, y[train], groups[train], folds, args.graine + rep, args)
                participants, p = moyenner_sujets(meta.iloc[train], y[train], proba)
                vals.append(multiclass_scores(participants["classe"], p, [0, 1])[0])
            v = pd.DataFrame(vals)
            result = dict(base=base, sous_ensemble=nom_g, n_variables=x.shape[1],
                          exactitude_equilibree=v.exactitude_equilibree.mean(),
                          exactitude_equilibree_std=v.exactitude_equilibree.std(ddof=1) if len(v) > 1 else 0,
                          auc=v.auc.mean())
            collect["reprise"].enregistrer(cle, result)
            collect["ablation"].append(result)
            log(f"    {nom_g:<26s} {v.exactitude_equilibree.mean():.3f} (AUC {v.auc.mean():.3f})")

    if args.permutations:
        g = pd.Series(y[train], index=groups[train]).groupby(level=0).first()
        for nom in args.permutation_modeles:
            if nom not in args.modeles:
                continue
            cle = ["permutation", base, nom]
            saved = collect["reprise"].charger(cle)
            if saved is not None:
                collect["perms"].append(saved)
                continue
            rng = np.random.default_rng(args.graine)
            t0 = time.time()
            x = ent.to_numpy(float)[train]
            yt, gt, mt = y[train], groups[train], meta.iloc[train]
            observed = []
            for rep in range(args.repetitions):
                folds = plis_groupes(yt, gt, args.plis, args.graine + rep)
                proba, _, _ = cv_proba(nom, x, yt, gt, folds, args.graine + rep, args)
                participants, p = moyenner_sujets(mt, yt, proba)
                observed.append(multiclass_scores(participants["classe"], p, [0, 1])[0]["exactitude_equilibree"])
            null = []
            for i in range(args.permutations):
                yp = pd.Series(rng.permutation(g.to_numpy()), index=g.index).loc[gt].to_numpy()
                repeated = []
                for rep in range(args.repetitions):
                    seed = args.graine + rep
                    try:
                        folds = plis_groupes(yp, gt, args.plis, seed)
                    except ValueError as exc:
                        log(f"    permutation {i + 1} ignorée : {exc}")
                        break
                    proba, _, _ = cv_proba(nom, x, yp, gt, folds, seed, args)
                    participants, p = moyenner_sujets(mt, yp, proba)
                    repeated.append(multiclass_scores(participants["classe"], p, [0, 1])[0]["exactitude_equilibree"])
                if len(repeated) != args.repetitions:
                    continue
                null.append(np.mean(repeated))
            obs = np.mean(observed)
            null = np.array(null)
            require(len(null) > 0, f"{base} : aucune permutation valide pour {nom}.")
            result = dict(base=base, modele=nom, observe=obs, moyenne_nulle=null.mean(),
                          quantile_95_nul=np.quantile(null, 0.95), p_valeur=p_valeur_permutation(obs, null),
                          n_permutations=len(null))
            collect["reprise"].enregistrer(cle, result)
            collect["perms"].append(result)
            log(f"  permutations {nom} : nul {null.mean():.3f} (q95 {np.quantile(null, 0.95):.3f}), observé {obs:.3f}, "
                f"p = {collect['perms'][-1]['p_valeur']:.3f} ({time.time() - t0:.0f} s)")
    collect["infos"][base] = dict(dossier=str(dossier), canaux=canaux, sujets=int(meta.subject_id.nunique()),
                                  fenetres=len(y), groupes=len(set(groups)), variables=int(ent.shape[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion", type=Path, default=None, help="Fusion de l'étape 04 avec partition train/test.")
    p.add_argument("--bases", nargs="+", choices=BASES, default=BASES)
    p.add_argument("--canaux-communs", action="store_true", help="Seulement les 10 canaux communs aux 3 bases.")
    p.add_argument("--canaux-origine", type=int, default=None,
                   help="Dépression : garder les fichiers d'origine à ce nombre de canaux (20 ou 22).")
    p.add_argument("--modeles", nargs="+", choices=MODELES, default=DEFAUT)
    p.add_argument("--caracteristiques", choices=["classiques", "etendues", "etendues_puissances"],
                   default="etendues_puissances",
                   help="Jeu de variables (défaut : entropies étendues, multiéchelles et puissances).")
    p.add_argument("--sans-reference-amplitude", action="store_true")
    p.add_argument("--plis", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=3, help="Nombre de répétitions (défaut : 3).")
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--c", type=float, default=1.0, help="C de la régression logistique avec --sans-tuning.")
    p.add_argument("--mlp-couches", type=int, nargs="+", default=[16])
    p.add_argument("--mlp-alpha", type=float, default=1.0)
    p.add_argument("--cv-imbriquee", action="store_true", help="Compatibilité : la recherche imbriquée est maintenant activée par défaut.")
    p.add_argument("--sans-tuning", action="store_true", help="Désactiver la recherche imbriquée.")
    p.add_argument("--profil-tuning", choices=["quick", "comprehensive"], default="comprehensive")
    p.add_argument("--plis-internes", type=int, default=None)
    p.add_argument("--tirages-tuning", type=int, default=None, help="Nombre de candidats par modèle et pli.")
    calcul = p.add_mutually_exclusive_group()
    calcul.add_argument("--cpu", action="store_true", help="Désactiver les sondes GPU et utiliser le CPU.")
    calcul.add_argument("--rapids", action="store_true", help="Essayer RAPIDS pour les modèles compatibles.")
    p.add_argument("--ablation", action="store_true", help="Contribution de chaque mesure et de chaque bande.")
    p.add_argument("--reprendre", action="store_true", help="Réutiliser les calculs terminés compatibles.")
    p.add_argument("--permutations", type=int, default=0)
    p.add_argument("--permutation-modeles", nargs="+", choices=MODELES, default=["regression_logistique"])
    args = p.parse_args()
    if (args.plis < 2 or args.repetitions < 1 or args.permutations < 0
            or (args.plis_internes is not None and args.plis_internes < 2)
            or (args.tirages_tuning is not None and args.tirages_tuning < 1)):
        p.error("--plis >= 2, --repetitions >= 1, --permutations >= 0, --plis-internes >= 2, --tirages-tuning >= 1.")
    if args.cv_imbriquee and args.sans_tuning:
        p.error("--cv-imbriquee et --sans-tuning sont incompatibles.")
    if args.permutations and not set(args.permutation_modeles).intersection(args.modeles):
        p.error("Aucun modèle de permutation n'est sélectionné dans --modeles.")
    if "xgboost" in args.modeles and not xgboost_disponible():
        p.error("xgboost demandé mais absent (pip install xgboost).")
    args.devices = {"xgboost": "cpu", "lightgbm": "cpu"} if args.cpu else detect_devices(args.modeles)
    args.rapids_actif = RAPIDS_ACTIF
    log(f"Calcul : XGBoost={args.devices['xgboost']}, LightGBM={args.devices['lightgbm']}; "
        + ("RAPIDS actif : voir ses journaux GPU/CPU." if RAPIDS_ACTIF else "autres modèles sur CPU."))
    start = time.time()
    suffix = f"canaux{args.canaux_origine}" if args.canaux_origine else ""
    out = nouveau_dossier("binaire", prefixe=suffix)
    collect = dict(rows=[], preds=[], ablation=[], perms=[], infos={}, modeles=[], out=out)
    entete("CLASSIFICATION BINAIRE TROUBLE / TÉMOINS (une base à la fois)")
    try:
        args.fusion = args.fusion or dossier_fusion_defaut()
        require(args.fusion is not None, "Aucune fusion trouvée ; lancer 04.")
        fichiers = ["fusion.json", "partition_sujets.csv"] + [f"fenetres_{base}.csv" for base in args.bases]
        empreintes = empreintes_fusion(args.fusion, fichiers)
        collect["reprise"] = Reprise("binaire", args, empreintes, out)
        for b in args.bases:
            analyser_base(b, args, collect)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Analyse arrêtée : {exc}\n")

    rows = pd.DataFrame(collect["rows"])
    rows.to_csv(out / "metriques_par_repetition.csv", index=False)
    summary = summarise(rows.drop(columns=[c for c in ["hyperparametres_choisis"] if c in rows]),
                        ["tache", "base", "modele", "caracteristiques"])
    summary.to_csv(out / "resume.csv", index=False)
    preds = pd.concat(collect["preds"], ignore_index=True)
    preds.to_csv(out / "predictions_test.csv", index=False)
    if collect["ablation"]:
        pd.DataFrame(collect["ablation"]).to_csv(out / "ablation.csv", index=False)
    if collect["perms"]:
        pd.DataFrame(collect["perms"]).to_csv(out / "permutations.csv", index=False)

    figures = []
    for b in args.bases:
        pb = preds[(preds.tache == f"binaire_{b}") & (preds.repetition == 1)]
        courbes = [(m, g["classe"].to_numpy(), g["proba_trouble"].to_numpy())
                   for m, g in pb.groupby("modele", sort=False) if m != "reference_majoritaire"]
        figures.append(plot_roc(courbes, f"{b} : courbes ROC sur les sujets test (répétition 1)", out / f"roc_{b}.png"))
        s = summary[(summary.base == b) & (summary.modele != "reference_majoritaire")
                    & ~summary.modele.str.startswith("reference_")]
        if len(s):
            best = s.sort_values("exactitude_equilibree", ascending=False).iloc[0]["modele"]
            pa = preds[(preds.tache == f"binaire_{b}") & (preds.modele == best)]
            mat = confusion_matrix(pa["classe"], pa["prediction"], labels=[0, 1])
            figures.append(plot_confusion(mat, ["témoin", NOM_TROUBLE[b]], ["témoin", NOM_TROUBLE[b]],
                                          f"{b} : {best} (cumul des répétitions)", out / f"confusion_{b}.png"))

    log("\n" + "=" * 78 + f"\nRÉSUMÉ (moyenne sur {args.repetitions} répétitions ; hasard = 0.50)")
    log("Même test pour tous les modèles et répétitions : choisir le meilleur sur ce test est optimiste.")
    cols = ["base", "modele", "n_variables", "exactitude_equilibree", "exactitude_equilibree_std", "auc",
            "sensibilite", "specificite", "f1_macro", "brier"]
    log(fmt_table(summary.sort_values(["base", "exactitude_equilibree"], ascending=[True, False]), cols))
    if Compteur.non_converges:
        log(f"Avertissement : {Compteur.non_converges} ajustements non convergés.")
    ecrire_json(out / "parametres.json", {
        "portee": "Évaluation exploratoire sur un test partagé ; choisir le meilleur modèle sur ce test est optimiste.",
        "bases": collect["infos"], "classes": CLASSES_BINAIRES,
        "validation": f"Partition 80/20 groupée de 04 ; {args.repetitions} ajustements sur le même test ; "
                      "probabilités moyennées par sujet ; ablation et permutations en CV sur train seulement.",
        "figures": figures, "ajustements_non_converges": Compteur.non_converges,
        "duree_s": round(time.time() - start, 1), "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "matplotlib", "xgboost"))})
    try:
        ecrire_manifeste(out, args.fusion, empreintes, arguments_dict(args), collect["modeles"])
    except (OSError, ValueError) as exc:
        p.exit(1, f"Provenance invalide : {exc}\n")
    log(f"\nDurée : {time.time() - start:.0f} s")
    log(f"SORTIE : {out}")


if __name__ == "__main__":
    main()
