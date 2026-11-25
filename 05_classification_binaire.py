"""Étape 5 : classification binaire trouble contre témoins, base par base.

Entrée  : dossiers d'extraction (sorties/03_extraction/<base>_<date>/ les plus récents, ou anciens
          dossiers extraction_corrigee.py avec --depression, --schizophrenie, --anxiete)
Sortie  : sorties/05_binaire/<date>/
    metriques_par_repetition.csv, resume.csv, predictions_hors_pli.csv
    ablation.csv               (--ablation) régression logistique sur une seule mesure / une seule bande
    permutations.csv           (--permutations N) test de permutation au niveau des sujets
    roc_<base>.png, confusion_<base>.png
    parametres.json

Pour chaque base : une ligne par sujet (médiane ou moyenne des fenêtres), tous les canaux de la
base (19 / 16 / 14) ou les 10 canaux communs (--canaux-communs). Modèles (--modeles) :
    reference_majoritaire   classe la plus fréquente
    regression_logistique   L2, C = --c, classes pondérées
    mlp                     MLP scikit-learn (lbfgs, alpha = --mlp-alpha)
    svm_rbf, foret_aleatoire, lda (covariance rétrécie), xgboost (si installé)
Références de confusion (ajoutées d'office) :
    reference_amplitude       régression logistique sur log(écart-type avant standardisation) seul
    reference_canaux_origine  nombre de canaux du fichier d'origine seul (dépression : 20 ou 22)
Validation : StratifiedGroupKFold par sujet (387-02w1 / 387-03w1 dans le même groupe), répétée
--repetitions fois avec des graines successives ; prétraitement ajusté dans chaque pli.
--cv-imbriquee : C, alpha (MLP) et C (SVM) choisis par validation croisée interne groupée.

Usage :
    python 05_classification_binaire.py
    python 05_classification_binaire.py --bases depression --canaux-origine 20
    python 05_classification_binaire.py --repetitions 10 --ablation --permutations 200
    python 05_classification_binaire.py --cv-imbriquee --modeles regression_logistique mlp svm_rbf

Dépendances : pip install numpy pandas scikit-learn matplotlib   (+ xgboost en option)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from eeg_config import BANDS, BASES, CLASSES_BINAIRES, GRAINE, MEASURES, NOM_TROUBLE
from eeg_donnees import (agreger, ajouter_arguments_extraction, charger_extraction, dossier_extraction_defaut,
                         lire_carte_canaux)
from eeg_utils import (Compteur, arguments_dict, ecrire_json, entete, fit_predict, fmt_table, log,
                       make_estimator, make_folds, multiclass_scores, nouveau_dossier, p_valeur_permutation,
                       plot_confusion, plot_roc, require, scores_binaires, summarise, versions,
                       xgboost_disponible)

MODELES = ["reference_majoritaire", "regression_logistique", "mlp", "svm_rbf", "foret_aleatoire", "lda", "xgboost"]
DEFAUT = ["reference_majoritaire", "regression_logistique", "mlp", "svm_rbf", "foret_aleatoire", "lda"]
GRILLES = {"regression_logistique": {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]},
           "mlp": {"mlpclassifier__alpha": [0.1, 1.0, 10.0]},
           "svm_rbf": {"svc__C": [0.1, 1.0, 10.0]}}


def estimateur(nom, seed, args, groups_tr=None):
    est = make_estimator(nom, seed, args.c, args.mlp_couches, args.mlp_alpha)
    if args.cv_imbriquee and nom in GRILLES:
        from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
        est = GridSearchCV(est, GRILLES[nom], scoring="balanced_accuracy",
                           cv=StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed), n_jobs=1)
    return est


def cv_proba(nom, x, y, groups, folds, seed, args):
    proba = np.zeros((len(y), 2))
    fold_id = np.zeros(len(y), dtype=int)
    choix = []
    for k, (tr, te) in enumerate(folds, 1):
        est = estimateur(nom, seed, args)
        if args.cv_imbriquee and nom in GRILLES:
            est.fit(x[tr], y[tr], groups=groups[tr])
            choix.append(est.best_params_)
            p = np.zeros((len(te), 2))
            p[:, est.classes_.astype(int)] = est.predict_proba(x[te])
        else:
            p, _ = fit_predict(nom, seed, x[tr], y[tr], x[te], 2, estimateur=est)
        proba[te] = p
        fold_id[te] = k
    return proba, fold_id, choix


def preparer_base(base, run, args, orders):
    info = charger_extraction(run, base, orders, args.agregation)
    rec = info["records"]
    canaux = info["channels"]
    if args.canaux_communs:
        canaux = [c for c in ["F7", "F3", "F4", "F8", "T7", "T8", "P7", "P8", "O1", "O2"] if c in canaux]
    if args.canaux_origine:
        require("n_canaux_origine" in rec, "--canaux-origine : colonne n_canaux_origine absente de l'audit.")
        keep = rec["n_canaux_origine"].eq(args.canaux_origine)
        require(keep.sum() >= 2 * args.plis, f"Trop peu d'enregistrements à {args.canaux_origine} canaux d'origine.")
        info["records"] = rec = rec.loc[keep]
        info["data"] = info["data"].loc[info["data"]["subject_id"].isin(rec.index)]
    ent, amp = agreger(info, canaux)
    inputs = {"entropies": ent}
    if not args.sans_reference_amplitude:
        inputs["amplitude"] = amp
    if "n_canaux_origine" in rec and rec["n_canaux_origine"].nunique() > 1:
        inputs["canaux_origine"] = (rec["n_canaux_origine"] == rec["n_canaux_origine"].max()).astype(int).to_frame("canaux_max")
    return info, canaux, inputs


def analyser_base(base, run, args, orders, collect):
    info, canaux, inputs = preparer_base(base, run, args, orders)
    rec = info["records"]
    y = rec["label"].astype(int).to_numpy()
    groups = rec["group_validation"].astype(str).to_numpy()
    ent = inputs["entropies"]
    require(min(np.bincount(y)) >= args.plis, f"{base} : moins de sujets d'une classe que de plis.")
    log(f"\n=== {base} ({NOM_TROUBLE[base]}) : {len(y)} sujets ({(y == 0).sum()} témoins, {(y == 1).sum()} trouble), "
        f"{len(set(groups))} groupes, {len(canaux)} canaux, {ent.shape[1]} variables")
    specs = [(m, "entropies") for m in args.modeles]
    if "amplitude" in inputs:
        specs.append(("reference_amplitude", "amplitude"))
    if "canaux_origine" in inputs:
        specs.append(("reference_canaux_origine", "canaux_origine"))
        log(f"  canaux d'origine : {rec['n_canaux_origine'].value_counts().to_dict()} → référence ajoutée")
    task = f"binaire_{base}"
    folds_by_rep = [make_folds(y, groups, args.plis, args.graine + r, must_have=y, min_par_strate=2)
                    for r in range(args.repetitions)]
    for nom, feat in specs:
        t0 = time.time()
        x = inputs[feat].to_numpy(float)
        kind = "regression_logistique" if nom.startswith("reference_") and nom != "reference_majoritaire" else nom
        for rep, folds in enumerate(folds_by_rep):
            seed = args.graine + rep
            proba, fold_id, choix = cv_proba(kind, x, y, groups, folds, seed, args)
            row, pred = multiclass_scores(y, proba, [0, 1])
            row.update(scores_binaires(y, pred))
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            row.update(vrais_temoins=tn, faux_trouble=fp, faux_temoins=fn, vrais_trouble=tp)
            if choix:
                row["hyperparametres_choisis"] = "; ".join(str(c) for c in choix)
            collect["rows"].append(dict(tache=task, base=base, modele=nom, caracteristiques=feat,
                                        repetition=rep + 1, n_variables=x.shape[1], **row))
            collect["preds"].append(pd.DataFrame({
                "tache": task, "harmonisation": "aucune", "modele": nom, "repetition": rep + 1,
                "subject_id": rec.index, "groupe": groups, "pli": fold_id, "classe": y, "prediction": pred,
                "proba_temoin": proba[:, 0], "proba_trouble": proba[:, 1]}))
        sub = pd.DataFrame([r for r in collect["rows"] if r["tache"] == task and r["modele"] == nom])
        log(f"  {nom:<26s} exact. équilibrée {sub.exactitude_equilibree.mean():.3f} ± "
            f"{sub.exactitude_equilibree.std(ddof=1) if len(sub) > 1 else 0:.3f}, AUC {sub.auc.mean():.3f}  "
            f"({time.time() - t0:.0f} s)")

    if args.ablation:
        log("  Ablation (régression logistique, une mesure ou une bande seule) :")
        cols = pd.Series(ent.columns)
        groupes_var = {f"mesure={m}": cols.str.endswith("|" + m) for m in MEASURES}
        groupes_var.update({f"bande={b}": cols.str.contains(f"|{b}|", regex=False) for b in BANDS})
        groupes_var.update({f"sans_mesure={m}": ~cols.str.endswith("|" + m) for m in MEASURES})
        for nom_g, mask in groupes_var.items():
            x = ent.loc[:, mask.to_numpy()].to_numpy(float)
            vals = []
            for rep, folds in enumerate(folds_by_rep):
                proba, _, _ = cv_proba("regression_logistique", x, y, groups, folds, args.graine + rep, args)
                vals.append(multiclass_scores(y, proba, [0, 1])[0])
            v = pd.DataFrame(vals)
            collect["ablation"].append(dict(base=base, sous_ensemble=nom_g, n_variables=x.shape[1],
                                            exactitude_equilibree=v.exactitude_equilibree.mean(),
                                            exactitude_equilibree_std=v.exactitude_equilibree.std(ddof=1) if len(v) > 1 else 0,
                                            auc=v.auc.mean()))
            log(f"    {nom_g:<26s} {v.exactitude_equilibree.mean():.3f} (AUC {v.auc.mean():.3f})")

    if args.permutations:
        g = pd.Series(y, index=groups).groupby(level=0).first()
        for nom in args.permutation_modeles:
            if nom not in args.modeles:
                continue
            rng = np.random.default_rng(args.graine)
            t0 = time.time()
            x = ent.to_numpy(float)
            null = []
            for i in range(args.permutations):
                yp = pd.Series(rng.permutation(g.to_numpy()), index=g.index).loc[groups].to_numpy()
                try:
                    folds = make_folds(yp, groups, args.plis, args.graine + 10_000 + i, must_have=yp, min_par_strate=2)
                except ValueError:
                    continue
                proba, _, _ = cv_proba(nom, x, yp, groups, folds, args.graine + i, args)
                null.append(multiclass_scores(yp, proba, [0, 1])[0]["exactitude_equilibree"])
            obs = np.mean([r["exactitude_equilibree"] for r in collect["rows"] if r["tache"] == task and r["modele"] == nom])
            null = np.array(null)
            collect["perms"].append(dict(base=base, modele=nom, observe=obs, moyenne_nulle=null.mean(),
                                         quantile_95_nul=np.quantile(null, 0.95), p_valeur=p_valeur_permutation(obs, null),
                                         n_permutations=len(null)))
            log(f"  permutations {nom} : nul {null.mean():.3f} (q95 {np.quantile(null, 0.95):.3f}), observé {obs:.3f}, "
                f"p = {collect['perms'][-1]['p_valeur']:.3f} ({time.time() - t0:.0f} s)")
    collect["infos"][base] = dict(dossier=str(run), canaux=canaux, sujets=len(y), temoins=int((y == 0).sum()),
                                  trouble=int((y == 1).sum()), groupes=len(set(groups)), variables=int(ent.shape[1]),
                                  groupes_multiples={k: int(v) for k, v in pd.Series(groups).value_counts().items() if v > 1})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ajouter_arguments_extraction(p)
    p.add_argument("--bases", nargs="+", choices=BASES, default=BASES)
    p.add_argument("--canaux-communs", action="store_true", help="Seulement les 10 canaux communs aux 3 bases.")
    p.add_argument("--canaux-origine", type=int, default=None,
                   help="Dépression : garder les fichiers d'origine à ce nombre de canaux (20 ou 22).")
    p.add_argument("--modeles", nargs="+", choices=MODELES, default=DEFAUT)
    p.add_argument("--sans-reference-amplitude", action="store_true")
    p.add_argument("--plis", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--c", type=float, default=1.0, help="Inverse de la pénalité L2 (régression logistique).")
    p.add_argument("--mlp-couches", type=int, nargs="+", default=[16])
    p.add_argument("--mlp-alpha", type=float, default=1.0)
    p.add_argument("--cv-imbriquee", action="store_true", help="Hyperparamètres choisis par CV interne groupée.")
    p.add_argument("--ablation", action="store_true", help="Contribution de chaque mesure et de chaque bande.")
    p.add_argument("--permutations", type=int, default=0)
    p.add_argument("--permutation-modeles", nargs="+", choices=MODELES, default=["regression_logistique"])
    args = p.parse_args()
    if args.plis < 2 or args.repetitions < 1:
        p.error("--plis >= 2 et --repetitions >= 1.")
    if "xgboost" in args.modeles and not xgboost_disponible():
        p.error("xgboost demandé mais absent (pip install xgboost).")
    start = time.time()
    orders = lire_carte_canaux(args.carte_canaux)
    collect = dict(rows=[], preds=[], ablation=[], perms=[], infos={})
    entete("CLASSIFICATION BINAIRE TROUBLE / TÉMOINS (une base à la fois)")
    try:
        for b in args.bases:
            run = getattr(args, b) or dossier_extraction_defaut(b)
            require(run is not None, f"Aucune extraction pour {b} (03_extraction_entropies.py ou --{b}).")
            analyser_base(b, Path(run).expanduser().resolve(), args, orders, collect)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Analyse arrêtée : {exc}\n")

    suffix = f"canaux{args.canaux_origine}" if args.canaux_origine else ""
    out = nouveau_dossier("binaire", prefixe=suffix)
    rows = pd.DataFrame(collect["rows"])
    rows.to_csv(out / "metriques_par_repetition.csv", index=False)
    summary = summarise(rows.drop(columns=[c for c in ["hyperparametres_choisis"] if c in rows]),
                        ["tache", "base", "modele", "caracteristiques"])
    summary.to_csv(out / "resume.csv", index=False)
    preds = pd.concat(collect["preds"], ignore_index=True)
    preds.to_csv(out / "predictions_hors_pli.csv", index=False)
    if collect["ablation"]:
        pd.DataFrame(collect["ablation"]).to_csv(out / "ablation.csv", index=False)
    if collect["perms"]:
        pd.DataFrame(collect["perms"]).to_csv(out / "permutations.csv", index=False)

    figures = []
    for b in args.bases:
        pb = preds[(preds.tache == f"binaire_{b}") & (preds.repetition == 1)]
        courbes = [(m, g["classe"].to_numpy(), g["proba_trouble"].to_numpy())
                   for m, g in pb.groupby("modele", sort=False) if m != "reference_majoritaire"]
        figures.append(plot_roc(courbes, f"{b} : courbes ROC hors pli (répétition 1)", out / f"roc_{b}.png"))
        s = summary[(summary.base == b) & (summary.modele != "reference_majoritaire")
                    & ~summary.modele.str.startswith("reference_")]
        if len(s):
            best = s.sort_values("exactitude_equilibree", ascending=False).iloc[0]["modele"]
            pa = preds[(preds.tache == f"binaire_{b}") & (preds.modele == best)]
            mat = confusion_matrix(pa["classe"], pa["prediction"], labels=[0, 1])
            figures.append(plot_confusion(mat, ["témoin", NOM_TROUBLE[b]], ["témoin", NOM_TROUBLE[b]],
                                          f"{b} : {best} (cumul des répétitions)", out / f"confusion_{b}.png"))

    log("\n" + "=" * 78 + f"\nRÉSUMÉ (moyenne sur {args.repetitions} répétitions ; hasard = 0.50)")
    cols = ["base", "modele", "n_variables", "exactitude_equilibree", "exactitude_equilibree_std", "auc",
            "sensibilite", "specificite", "f1_macro", "brier"]
    log(fmt_table(summary.sort_values(["base", "exactitude_equilibree"], ascending=[True, False]), cols))
    if Compteur.non_converges:
        log(f"Avertissement : {Compteur.non_converges} ajustements non convergés.")
    ecrire_json(out / "parametres.json", {
        "portee": "Évaluation exploratoire interne ; hyperparamètres fixés a priori sauf --cv-imbriquee.",
        "bases": collect["infos"], "classes": CLASSES_BINAIRES,
        "validation": f"StratifiedGroupKFold {args.plis} plis × {args.repetitions} répétitions, graines "
                      f"{args.graine}..{args.graine + args.repetitions - 1}",
        "figures": figures, "ajustements_non_converges": Compteur.non_converges,
        "duree_s": round(time.time() - start, 1), "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "matplotlib", "xgboost"))})
    log(f"\nDurée : {time.time() - start:.0f} s")
    log(f"SORTIE : {out}")


if __name__ == "__main__":
    main()
