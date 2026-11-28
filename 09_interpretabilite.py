"""Étape 9 : interprétabilité — quelles mesures, bandes et électrodes portent la classification ?

Entrée  : dossiers d'extraction (binaire) et sorties/04_fusion/<date>/ (4 classes)
Sortie  : sorties/09_interpretabilite/<date>/
    importance_groupes.csv   importance par permutation de GROUPES de variables (toutes les
                             variables d'une mesure, d'une bande ou d'une électrode permutées
                             ensemble entre sujets test) : baisse d'exactitude équilibrée et d'AUC
                             hors pli, moyenne sur répétitions et tirages
    coefficients_rl.csv      coefficients standardisés de la régression logistique (moyenne des plis)
    shap_valeurs.csv         (--shap) |SHAP| moyen par variable et classe, sur sujets hors pli
    hyperparametres_par_pli.csv paramètres choisis dans chaque pli d'entraînement
    importance_<tâche>.png   barres par mesure, bande et électrode
    interpretabilite.json

Pourquoi des groupes : les 250 à 475 variables sont très corrélées ; permuter une variable seule
sous-estime son rôle. Les importances sont calculées hors pli avec le modèle sélectionné et réglé par recherche groupée
à l'intérieur de chaque pli. En 4 classes, l'harmonisation est ajustée dans chaque pli interne.

Usage :
    python 09_interpretabilite.py
    python 09_interpretabilite.py --taches binaire_depression multiclasse --modele mlp --tirages 10
    python 09_interpretabilite.py --modele xgboost --shap --profil-tuning quick

Dépendances : numpy pandas scipy scikit-learn matplotlib xgboost lightgbm (+ shap en option)
"""
import argparse
from pathlib import Path

from eeg_tuning import detect_devices, predict_proba, tune_and_fit

import numpy as np
import pandas as pd

import eeg_harmonisation as eh
from eeg_config import BANDS, BASES, CLASSES, GRAINE, MEASURES, MESURES_ETENDUES, PUISSANCES
from eeg_donnees import (agreger, caracteristiques_fusion, charger_extraction, charger_fusion,
                         decomposer_colonnes, dossier_extraction_defaut, lire_carte_canaux)
from eeg_utils import (arguments_dict, ecrire_json, entete, log, make_folds, multiclass_scores,
                       nouveau_dossier, plt_agg, require, versions)

TACHES = [f"binaire_{b}" for b in BASES] + ["multiclasse"]


def donnees_tache(tache, args, orders):
    """x (DataFrame), y, groupes, bases (ou None), harmonisation, étiquettes."""
    if tache == "multiclasse":
        common, ent, _, meta, _, dossier = charger_fusion(args.fusion)
        ent = caracteristiques_fusion(common, ent, meta, dossier, args.caracteristiques)
        y = meta["classe"].to_numpy()
        strata = pd.Categorical(meta["base"], BASES).codes * 2 + meta["label_binaire"].to_numpy()
        return ent, y, meta["groupe"].to_numpy(), meta["base"].to_numpy(), strata, args.harmonisation, list(range(4))
    base = tache.replace("binaire_", "")
    run = getattr(args, base) or dossier_extraction_defaut(base)
    if run is None:
        raise ValueError(f"aucune extraction pour {base}")
    info = charger_extraction(Path(run), base, orders, args.agregation)
    rec = info["records"]
    canaux = info["channels"]
    if args.canaux_communs:
        canaux = [c for c in ["F7", "F3", "F4", "F8", "T7", "T8", "P7", "P8", "O1", "O2"] if c in canaux]
    if args.canaux_origine:
        require("n_canaux_origine" in rec, "--canaux-origine : colonne n_canaux_origine absente de l'audit.")
        keep = rec["n_canaux_origine"].eq(args.canaux_origine)
        require(keep.sum() >= 2 * args.plis, "Trop peu d'enregistrements pour --canaux-origine.")
        info["records"] = rec = rec.loc[keep]
        info["data"] = info["data"].loc[info["data"]["subject_id"].isin(rec.index)]
    mesures = list(MEASURES)
    if args.caracteristiques != "classiques":
        mesures += MESURES_ETENDUES
    if args.caracteristiques == "etendues_puissances":
        mesures += PUISSANCES
    ent, _ = agreger(info, canaux, mesures)
    y = rec["label"].astype(int).to_numpy()
    return ent, y, rec["group_validation"].astype(str).to_numpy(), None, y, "aucune", [0, 1]


def groupes_variables(colonnes):
    d = decomposer_colonnes(colonnes)
    g = {}
    for m in dict.fromkeys(d["mesure"]):
        g[("mesure", m)] = np.flatnonzero(d["mesure"].eq(m).to_numpy())
    for b in BANDS:
        g[("bande", b)] = np.flatnonzero(d["bande"].eq(b).to_numpy())
    for c in dict.fromkeys(d["canal"]):
        g[("electrode", c)] = np.flatnonzero(d["canal"].eq(c).to_numpy())
    return g


def score(y, proba, labels):
    row, _ = multiclass_scores(y, proba, labels)
    return row["exactitude_equilibree"], row["auc" if len(labels) == 2 else "auc_ovr_macro"]


def expliquer_shap(model, name, x_tr, x_te, columns, n_classes, args, seed):
    """Explain held-out participants in the transformed feature space of the fitted fold."""
    import shap

    selected = model.named_steps["select"].get_support()
    names = np.asarray(columns)[selected]
    train = model[:-1].transform(x_tr)
    test = model[:-1].transform(x_te)
    rng = np.random.default_rng(seed)
    train = train[rng.choice(len(train), size=min(args.shap_fond, len(train)), replace=False)]
    test = test[rng.choice(len(test), size=min(args.shap_sujets, len(test)), replace=False)]
    estimator = model.named_steps["model"]
    if name in ("foret_aleatoire", "xgboost", "lightgbm"):
        values = shap.TreeExplainer(estimator).shap_values(test, check_additivity=False)
    elif name == "regression_logistique":
        values = shap.LinearExplainer(estimator, train).shap_values(test)
    else:
        explainer = shap.KernelExplainer(estimator.predict_proba, train)
        values = explainer.shap_values(test, nsamples=args.shap_echantillons, silent=True)
    if isinstance(values, list):
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 2:
        values = values[:, :, None]
        class_ids = [1] if n_classes == 2 else []
    elif values.ndim == 3:
        class_ids = list(np.asarray(estimator.classes_, dtype=int))
    else:
        class_ids = []
    if values.shape[:2] != (len(test), len(names)) or values.shape[-1] != len(class_ids):
        raise ValueError(f"{name}: forme SHAP inattendue {values.shape}")
    rows = []
    for k, class_id in enumerate(class_ids):
        importance = np.zeros(len(columns))
        importance[selected] = np.abs(values[:, :, k]).mean(axis=0)
        rows.append(pd.DataFrame({"classe": CLASSES[class_id] if n_classes > 2
                                  else ("temoin" if class_id == 0 else "trouble"),
                                  "variable": columns, "shap_abs_moyen": importance}))
    return pd.concat(rows, ignore_index=True)


def analyser(tache, args, orders, collect):
    x_df, y, groups, bases, strata, harm, labels = donnees_tache(tache, args, orders)
    x = x_df.to_numpy(float)
    n_cls = len(labels)
    gv = groupes_variables(x_df.columns)
    log(f"\n=== {tache} : {len(y)} sujets, {x.shape[1]} variables, modèle {args.modele}, harmonisation {harm}")
    imp_rows, coef_rows, shap_rows = [], [], []
    rng = np.random.default_rng(args.graine)
    for rep in range(args.repetitions):
        seed = args.graine + rep
        folds = make_folds(strata, groups, args.plis, seed, must_have=strata, min_par_strate=2)
        fitted = []
        proba = np.zeros((len(y), n_cls))
        for fold, (tr, te) in enumerate(folds, 1):
            reference = y[tr] == 0 if bases is not None else None
            model, tuning = tune_and_fit(
                args.modele, x[tr], y[tr], groups[tr], strata[tr], seed, args.devices,
                args.profil_tuning, args.plis_internes, args.tirages_tuning,
                bases=None if bases is None else bases[tr], reference=reference,
                harmonisation=harm, enabled=not args.sans_tuning,
                fixed_c=args.c, fixed_mlp_layers=args.mlp_couches, fixed_mlp_alpha=args.mlp_alpha)
            if bases is not None and harm != "aucune":
                x_tr, x_te = eh.harmonise(x[tr], x[te], bases[tr], bases[te], y[tr] == 0, harm)
            else:
                x_tr, x_te = x[tr], x[te]
            model.fit(x_tr, y[tr])
            proba[te] = predict_proba(model, x_te, n_cls)
            fitted.append((te, x_te, model))
            if tuning["best_params"]:
                collect["tuning"].append(dict(tache=tache, modele=args.modele, repetition=rep + 1,
                                              pli=fold, **tuning))
            if args.modele == "regression_logistique":
                lr = model[-1]
                coefs = lr.coef_ if n_cls > 2 else lr.coef_[:1]
                selected = model.named_steps["select"].get_support()
                for k, cvec in enumerate(coefs):
                    cls = CLASSES[lr.classes_[k]] if n_cls > 2 else "trouble"
                    full = np.zeros(x.shape[1])
                    full[selected] = cvec
                    coef_rows.append(pd.DataFrame({"tache": tache, "repetition": rep + 1, "classe": cls,
                                                   "variable": x_df.columns, "coefficient": full}))
            if args.shap:
                shap_rows.append(expliquer_shap(model, args.modele, x_tr, x_te, x_df.columns,
                                                 n_cls, args, seed).assign(tache=tache, repetition=rep + 1, pli=fold))
        base_ba, base_auc = score(y, proba, labels)
        for (kind, nom), cols in gv.items():
            drops_ba, drops_auc = [], []
            for _ in range(args.tirages):
                pp = np.zeros_like(proba)
                for te, x_te, model in fitted:
                    xp = x_te.copy()
                    perm = rng.permutation(len(te))
                    xp[:, cols] = xp[perm][:, cols]       # même permutation pour tout le groupe
                    out = np.zeros((len(te), n_cls))
                    out[:, np.asarray(model.classes_).astype(int)] = model.predict_proba(xp)
                    pp[te] = out
                ba, auc = score(y, pp, labels)
                drops_ba.append(base_ba - ba)
                drops_auc.append(base_auc - auc)
            imp_rows.append(dict(tache=tache, repetition=rep + 1, type=kind, groupe=nom, n_variables=len(cols),
                                 score_reference=base_ba, baisse_exactitude_equilibree=np.mean(drops_ba),
                                 baisse_auc=np.mean(drops_auc)))
        log(f"  répétition {rep + 1} : exactitude équilibrée de référence {base_ba:.3f}")
    imp = pd.DataFrame(imp_rows)
    collect["imp"].append(imp)
    if coef_rows:
        collect["coef"].append(pd.concat(coef_rows))
    if shap_rows:
        collect["shap"].append(pd.concat(shap_rows))
    res = imp.groupby(["type", "groupe"], sort=False)[["baisse_exactitude_equilibree", "baisse_auc"]].mean()
    for kind in ["mesure", "bande"]:
        r = res.loc[kind].sort_values("baisse_auc", ascending=False)
        log(f"  par {kind} (baisse d'AUC) : " + ", ".join(f"{i} {v:+.3f}" for i, v in r["baisse_auc"].items()))
    return imp


def figure(imp, tache, out):
    plt = plt_agg()
    res = imp.groupby(["type", "groupe"], sort=False)[["baisse_auc", "baisse_exactitude_equilibree"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [5, 5, 10]})
    for ax, kind in zip(axes, ["mesure", "bande", "electrode"]):
        r = res.loc[kind]
        ax.bar(range(len(r)), r[("baisse_auc", "mean")], yerr=r[("baisse_auc", "std")].fillna(0), capsize=3,
               color="tab:blue")
        ax.set_xticks(range(len(r)), r.index, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"par {kind}")
    axes[0].set_ylabel("Baisse d'AUC quand le groupe est permuté")
    fig.suptitle(f"{tache} : importance par permutation de groupes (hors pli)")
    fig.tight_layout()
    path = out / f"importance_{tache}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for b in BASES:
        p.add_argument(f"--{b}", type=Path, default=None)
    p.add_argument("--fusion", type=Path, default=None)
    p.add_argument("--carte-canaux", type=Path)
    p.add_argument("--agregation", choices=["moyenne", "mediane"], default="moyenne")
    p.add_argument("--canaux-communs", action="store_true", help="Tâches binaires : mêmes canaux qu'en 05.")
    p.add_argument("--canaux-origine", type=int, default=None, help="Tâches binaires : filtrer comme en 05.")
    p.add_argument("--taches", nargs="+", choices=TACHES, default=TACHES)
    p.add_argument("--modele", choices=["regression_logistique", "mlp", "svm", "svm_rbf", "knn",
                                        "foret_aleatoire", "lda", "xgboost", "lightgbm"],
                   default="regression_logistique")
    p.add_argument("--caracteristiques", choices=["classiques", "etendues", "etendues_puissances"],
                   default="classiques")
    p.add_argument("--sans-tuning", action="store_true", help="Désactiver la recherche imbriquée.")
    p.add_argument("--profil-tuning", choices=["quick", "comprehensive"], default="comprehensive")
    p.add_argument("--plis-internes", type=int, default=None)
    p.add_argument("--tirages-tuning", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--c", type=float, default=1.0, help="C de la régression logistique avec --sans-tuning.")
    p.add_argument("--mlp-couches", type=int, nargs="+", default=[16])
    p.add_argument("--mlp-alpha", type=float, default=1.0)
    p.add_argument("--harmonisation", choices=["aucune"] + eh.HARMONISATIONS[:2], default="zscore",
                   help="4 classes (CORAL exclu : les variables d'origine sont perdues).")
    p.add_argument("--plis", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--tirages", type=int, default=5, help="Permutations par groupe et par répétition.")
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--shap", action="store_true", help="Calculer SHAP sur les sujets hors pli.")
    p.add_argument("--shap-sujets", type=int, default=40, help="Maximum de sujets expliqués par pli.")
    p.add_argument("--shap-fond", type=int, default=20, help="Maximum de sujets d'entraînement pour le fond SHAP.")
    p.add_argument("--shap-echantillons", type=int, default=100, help="Budget Kernel SHAP pour les modèles non linéaires.")
    args = p.parse_args()
    if (args.plis < 2 or args.repetitions < 1 or args.tirages < 1
            or (args.plis_internes is not None and args.plis_internes < 2)
            or (args.tirages_tuning is not None and args.tirages_tuning < 1)
            or min(args.shap_sujets, args.shap_fond, args.shap_echantillons) < 1):
        p.error("Nombre de plis, répétitions, tirages ou paramètres SHAP invalide.")
    if args.shap:
        try:
            import shap  # noqa: F401
        except ModuleNotFoundError:
            p.error("--shap demandé mais le paquet shap est absent (pip install shap).")
    if args.harmonisation == "filtrage":
        p.error("Filtrage : les variables retirées changent d'un pli à l'autre ; utiliser zscore ou aucune.")
    args.devices = {"xgboost": "cpu", "lightgbm": "cpu"} if args.cpu else detect_devices([args.modele])
    log(f"Calcul : XGBoost={args.devices['xgboost']}, LightGBM={args.devices['lightgbm']}; autres modèles sur CPU.")
    orders = lire_carte_canaux(args.carte_canaux)
    out = nouveau_dossier("interpretabilite")
    entete("INTERPRÉTABILITÉ : importance des mesures, bandes et électrodes")
    collect = dict(imp=[], coef=[], shap=[], tuning=[])
    figures = []
    for t in args.taches:
        try:
            imp = analyser(t, args, orders, collect)
            figures.append(figure(imp, t, out))
        except (OSError, ValueError) as exc:
            p.exit(1, f"Analyse arrêtée ({t}) : {exc}\n")
    if not collect["imp"]:
        p.exit(1, "Aucune tâche analysée.\n")
    imp = pd.concat(collect["imp"], ignore_index=True)
    imp.to_csv(out / "importance_groupes_par_repetition.csv", index=False)
    (imp.groupby(["tache", "type", "groupe"], sort=False)[["n_variables", "score_reference",
                                                          "baisse_exactitude_equilibree", "baisse_auc"]]
     .mean().reset_index().to_csv(out / "importance_groupes.csv", index=False))
    if collect["coef"]:
        coef = pd.concat(collect["coef"], ignore_index=True)
        coef = coef.groupby(["tache", "classe", "variable"], sort=False)["coefficient"].agg(["mean", "std"]).reset_index()
        coef = coef.join(decomposer_colonnes(coef["variable"]).reset_index(drop=True))
        coef.to_csv(out / "coefficients_rl.csv", index=False)
    if collect["shap"]:
        sh = pd.concat(collect["shap"], ignore_index=True)
        sh.to_csv(out / "shap_par_pli.csv", index=False)
        sh.groupby(["tache", "classe", "variable"], sort=False)["shap_abs_moyen"].mean().reset_index().to_csv(
            out / "shap_valeurs.csv", index=False)
    if collect["tuning"]:
        pd.DataFrame(collect["tuning"]).to_csv(out / "hyperparametres_par_pli.csv", index=False)
    ecrire_json(out / "interpretabilite.json", {
        "methode": "importance par permutation de groupes de variables, hors pli, sans réentraînement",
        "figures": figures, "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "matplotlib", "shap"))})
    log(f"\nSORTIE : {out}")


if __name__ == "__main__":
    main()
