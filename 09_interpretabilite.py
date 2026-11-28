"""Étape 9 : expliquer les modèles sur fenêtres sauvegardés par 05 et 06.

Les mêmes participants test que 05/06 sont utilisés. Les importances sont exploratoires :
elles ne doivent pas servir à sélectionner des variables puis à réutiliser ce test comme
évaluation finale indépendante. Seuls les fichiers joblib de confiance doivent être chargés.
"""
import argparse
import sys
from pathlib import Path

RAPIDS_ACTIF = False
if __name__ == "__main__" and "--rapids" in sys.argv and "--cpu" not in sys.argv:
    from gpu_setup import enable_cuml_accel
    RAPIDS_ACTIF = enable_cuml_accel()

from eeg_tuning import predict_proba

import joblib
import numpy as np
import pandas as pd

import eeg_harmonisation as eh
from eeg_config import BASES, CLASSES, GRAINE
from eeg_donnees import decomposer_colonnes
from eeg_fenetres import charger_fenetres, moyenner_sujets
from eeg_modeles_sauvegardes import lire_manifeste
from eeg_utils import (arguments_dict, dernier_dossier, ecrire_json, entete, log,
                       multiclass_scores, nouveau_dossier, plt_agg, require, versions)

TACHES = [f"binaire_{b}" for b in BASES] + ["multiclasse"]


def groupes_variables(colonnes):
    d = decomposer_colonnes(colonnes)
    groupes = {}
    for kind, column in (("mesure", "mesure"), ("bande", "bande"), ("electrode", "canal")):
        for value in dict.fromkeys(d[column]):
            groupes[(kind, value)] = np.flatnonzero(d[column].eq(value).to_numpy())
    return groupes


def score(y, proba, labels):
    row, _ = multiclass_scores(y, proba, labels)
    return row["exactitude_equilibree"], row["auc" if len(labels) == 2 else "auc_ovr_macro"]


def expliquer_shap(model, name, x_tr, x_te, columns, n_classes, args, seed,
                    train_subjects=None, test_subjects=None):
    """Mean absolute window SHAP, averaged within and then across test participants."""
    import shap

    selected = model.named_steps["select"].get_support()
    names = np.asarray(columns)[selected]
    train = model[:-1].transform(x_tr)
    test = model[:-1].transform(x_te)
    rng = np.random.default_rng(seed)
    train_subjects = np.arange(len(train)) if train_subjects is None else np.asarray(train_subjects)
    test_subjects = np.arange(len(test)) if test_subjects is None else np.asarray(test_subjects)
    train_ids = np.unique(train_subjects)
    train_ids = rng.choice(train_ids, size=min(args.shap_fond, len(train_ids)), replace=False)
    train_index = [rng.choice(np.flatnonzero(train_subjects == sid)) for sid in train_ids]
    test_ids = np.unique(test_subjects)
    test_ids = rng.choice(test_ids, size=min(args.shap_sujets, len(test_ids)), replace=False)
    test_index = np.flatnonzero(np.isin(test_subjects, test_ids))
    train, test = train[train_index], test[test_index]
    selected_subjects = test_subjects[test_index]
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
        per_subject = pd.DataFrame(np.abs(values[:, :, k])).groupby(selected_subjects).mean()
        importance[selected] = per_subject.mean(axis=0).to_numpy()
        rows.append(pd.DataFrame({"classe": CLASSES[class_id] if n_classes > 2
                                  else ("temoin" if class_id == 0 else "trouble"),
                                  "variable": columns, "shap_abs_moyen": importance}))
    return pd.concat(rows, ignore_index=True)


def permuter_sujets(x, subjects, columns, rng):
    """Exchange complete feature trajectories between participants, resampling unequal lengths."""
    subjects = np.asarray(subjects)
    ids = np.unique(subjects)
    donors = rng.permutation(ids)
    result = x.copy()
    for target, donor in zip(ids, donors):
        target_rows = np.flatnonzero(subjects == target)
        donor_rows = np.flatnonzero(subjects == donor)
        positions = np.arange(len(target_rows)) * len(donor_rows) // len(target_rows)
        result[np.ix_(target_rows, columns)] = x[np.ix_(donor_rows[positions], columns)]
    return result


def verifier_predictions(dossier, tache, nom, harm, repetition, meta, y, proba):
    participants, averaged = moyenner_sujets(meta, y, proba)
    path = Path(dossier) / "predictions_test.csv"
    require(path.is_file(), f"Prédictions de référence absentes : {path}.")
    saved = pd.read_csv(path, dtype={"subject_id": str})
    mask = saved["tache"].eq(tache) & saved["modele"].eq(nom) & saved["repetition"].eq(repetition)
    if "harmonisation" in saved:
        mask &= saved["harmonisation"].eq(harm)
    saved = saved.loc[mask].set_index("subject_id")
    require(not saved.index.has_duplicates and set(saved.index) == set(participants.index.astype(str)),
            f"Participants test incohérents avec {path}.")
    cols = ["proba_temoin", "proba_trouble"] if tache != "multiclasse" else [f"proba_{c}" for c in CLASSES]
    expected = saved.reindex(participants.index.astype(str))[cols].to_numpy(float)
    require(np.allclose(averaged, expected, rtol=1e-5, atol=1e-7),
            f"Le modèle chargé ne reproduit pas les prédictions de {path}.")
    return participants, averaged


def analyser(tache, args, collect):
    binary = tache != "multiclasse"
    dossier = args.binaire if binary else args.multiclasse
    require(dossier is not None, f"Résultats {'05' if binary else '06'} introuvables.")
    manifest = lire_manifeste(dossier, args.fusion)
    if args.fusion is None:
        args.fusion = Path(manifest["fusion"])
    options = manifest["arguments"]
    require(args.caracteristiques is None or args.caracteristiques == options["caracteristiques"],
            f"Jeu de caractéristiques incompatible avec {dossier}.")
    base = tache.removeprefix("binaire_") if binary else None
    ent, _, meta, _, _ = charger_fenetres(
        manifest["fusion"], base, options["caracteristiques"],
        options.get("canaux_communs", False) if binary else False)
    if binary and options.get("canaux_origine"):
        require(meta["n_canaux_origine"].notna().all(), "Nombre de canaux d'origine absent.")
        keep = meta["n_canaux_origine"].eq(options["canaux_origine"])
        ent, meta = ent.loc[keep].reset_index(drop=True), meta.loc[keep].reset_index(drop=True)
    labels = [0, 1] if binary else list(range(len(CLASSES)))
    y = meta["label_binaire" if binary else "classe"].to_numpy(dtype=int)
    train = np.flatnonzero(meta["partition"].eq("train"))
    test = np.flatnonzero(meta["partition"].eq("test"))
    require(len(train) and len(test) and set(y[train]) == set(labels) and set(y[test]) == set(labels),
            f"{tache} : toutes les classes doivent être présentes dans train et test.")
    harm = "aucune" if binary else args.harmonisation
    records = sorted((r for r in manifest["modeles"] if r["tache"] == tache
                      and r["modele"] == args.modele and r["harmonisation"] == harm),
                     key=lambda r: r["repetition"])
    require(records, f"Modèle {args.modele}/{harm} absent de {dossier} pour {tache}.")
    if args.repetitions is not None:
        require(args.repetitions <= len(records), f"Seulement {len(records)} répétitions sauvegardées dans {dossier}.")
        records = records[:args.repetitions]
    x = ent.to_numpy(float)
    x_tr, x_te = x[train], x[test]
    if not binary and harm != "aucune":
        eh.configurer(argparse.Namespace(**options))
        bases = meta["base"].to_numpy()
        x_tr, x_te = eh.harmonise(x_tr, x_te, bases[train], bases[test], y[train] == 0, harm)
    groups = groupes_variables(ent.columns)
    imp_rows, shap_rows, coef_rows = [], [], []
    for record in records:
        rep = record["repetition"]
        path = Path(dossier) / record["fichier"]
        require(path.is_file(), f"Modèle sauvegardé absent : {path}.")
        model = joblib.load(path)
        proba = predict_proba(model, x_te, len(labels))
        participants, p = verifier_predictions(dossier, tache, args.modele, harm, rep, meta.iloc[test], y[test], proba)
        base_ba, base_auc = score(participants["classe"], p, labels)
        rng = np.random.default_rng(args.graine + rep - 1)
        for (kind, name), cols in groups.items():
            ba, auc = [], []
            for _ in range(args.tirages):
                altered = permuter_sujets(x_te, meta.iloc[test]["subject_id"].to_numpy(), cols, rng)
                permuted = predict_proba(model, altered, len(labels))
                part, pp = moyenner_sujets(meta.iloc[test], y[test], permuted)
                b, a = score(part["classe"], pp, labels)
                ba.append(base_ba - b)
                auc.append(base_auc - a)
            imp_rows.append(dict(tache=tache, repetition=rep, type=kind, groupe=name, n_variables=len(cols),
                                 score_reference=base_ba, baisse_exactitude_equilibree=np.mean(ba),
                                 baisse_auc=np.mean(auc)))
        if args.modele == "regression_logistique":
            lr = model.named_steps["model"]
            selected = model.named_steps["select"].get_support()
            for k, cvec in enumerate(lr.coef_ if not binary else lr.coef_[:1]):
                cls = CLASSES[lr.classes_[k]] if not binary else "trouble"
                full = np.zeros(x.shape[1])
                full[selected] = cvec
                coef_rows.append(pd.DataFrame({"tache": tache, "repetition": rep, "classe": cls,
                                               "variable": ent.columns, "coefficient": full}))
        if args.shap:
            shap_rows.append(expliquer_shap(model, args.modele, x_tr, x_te, ent.columns,
                                             len(labels), args, args.graine + rep - 1,
                                             meta.iloc[train]["subject_id"], meta.iloc[test]["subject_id"])
                             .assign(tache=tache, repetition=rep, pli=1))
        log(f"  {tache} répétition {rep} : exactitude équilibrée sur test {base_ba:.3f}")
    imp = pd.DataFrame(imp_rows)
    collect["imp"].append(imp)
    if coef_rows:
        collect["coef"].append(pd.concat(coef_rows))
    if shap_rows:
        collect["shap"].append(pd.concat(shap_rows))
    collect["sources"][tache] = str(Path(dossier).resolve())
    return imp


def figure(imp, tache, out):
    plt = plt_agg()
    res = imp.groupby(["type", "groupe"], sort=False)[["baisse_auc", "baisse_exactitude_equilibree"]].agg(["mean", "std"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [5, 5, 10]})
    for ax, kind in zip(axes, ["mesure", "bande", "electrode"]):
        r = res.loc[kind]
        ax.bar(range(len(r)), r[("baisse_auc", "mean")], yerr=r[("baisse_auc", "std")].fillna(0), capsize=3)
        ax.set_xticks(range(len(r)), r.index, rotation=45, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"par {kind}")
    axes[0].set_ylabel("Baisse d'AUC quand le groupe est permuté")
    fig.suptitle(f"{tache} : importance sur les participants test")
    fig.tight_layout()
    path = out / f"importance_{tache}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binaire", type=Path, default=None, help="Dossier 05 contenant les modèles.")
    p.add_argument("--multiclasse", type=Path, default=None, help="Dossier 06 contenant les modèles.")
    p.add_argument("--fusion", type=Path, default=None, help="Vérifier la fusion utilisée en 05/06.")
    p.add_argument("--taches", nargs="+", choices=TACHES, default=TACHES)
    p.add_argument("--modele", choices=["regression_logistique", "mlp", "svm", "svm_rbf", "knn",
                                        "foret_aleatoire", "lda", "xgboost", "lightgbm"],
                   default="regression_logistique")
    p.add_argument("--caracteristiques", choices=["classiques", "etendues", "etendues_puissances"], default=None)
    p.add_argument("--harmonisation", choices=["aucune", "zscore"], default="zscore")
    p.add_argument("--repetitions", type=int, default=None, help="Nombre de modèles sauvegardés à expliquer (défaut : tous).")
    p.add_argument("--tirages", type=int, default=5)
    p.add_argument("--graine", type=int, default=GRAINE)
    calcul = p.add_mutually_exclusive_group()
    calcul.add_argument("--cpu", action="store_true")
    calcul.add_argument("--rapids", action="store_true")
    p.add_argument("--shap", action="store_true")
    p.add_argument("--shap-sujets", type=int, default=40, help="Maximum de participants test expliqués.")
    p.add_argument("--shap-fond", type=int, default=20, help="Maximum de participants d'entraînement pour le fond SHAP.")
    p.add_argument("--shap-echantillons", type=int, default=100)
    args = p.parse_args()
    if (args.repetitions is not None and args.repetitions < 1) or min(
            args.tirages, args.shap_sujets, args.shap_fond, args.shap_echantillons) < 1:
        p.error("Nombre de répétitions, tirages ou paramètres SHAP invalide.")
    args.binaire = args.binaire or dernier_dossier("binaire", contient="predictions_test.csv")
    args.multiclasse = args.multiclasse or dernier_dossier("multiclasse", contient="predictions_test.csv")
    out = nouveau_dossier("interpretabilite")
    entete("INTERPRÉTABILITÉ DES MODÈLES SUR FENÊTRES")
    log("Modèles sauvegardés en 05/06 ; interprétation exploratoire sur leurs participants test.")
    collect = dict(imp=[], coef=[], shap=[], sources={})
    figures = []
    try:
        for tache in args.taches:
            imp = analyser(tache, args, collect)
            figures.append(figure(imp, tache, out))
    except (OSError, ValueError) as exc:
        p.exit(1, f"Analyse arrêtée : {exc}\n")
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
    ecrire_json(out / "interpretabilite.json", {
        "methode": "Modèles 05/06 sauvegardés ; SHAP absolu moyen par fenêtre puis par participant test ; "
                   "permutation des trajectoires entre participants test.",
        "portee": "Interprétation exploratoire du test commun ; pas une validation indépendante.",
        "sources": collect["sources"], "figures": figures, "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "matplotlib", "shap"))})
    log(f"\nSORTIE : {out}")


if __name__ == "__main__":
    main()
