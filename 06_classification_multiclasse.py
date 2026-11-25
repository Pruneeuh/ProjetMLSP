"""Étape 6 : classification 4 classes (témoin, dépression, schizophrénie, anxiété) avec
harmonisation dans les plis, contrôles de l'effet de site, permutations et leave-one-dataset-out.

Entrée  : sorties/04_fusion/<date>/ (la plus récente, ou --fusion <dossier>)
Sortie  : sorties/06_multiclasse/<date>/
    multiclasse_metriques.csv, multiclasse_resume.csv, predictions_hors_pli.csv
    matrice_confusion_<modèle>__<harmonisation>.png  (4 classes + témoins séparés par base)
    controle_site_metriques.csv, controle_site_resume.csv, controle_site_temoins.png
    permutations.csv, lodo_metriques.csv, lodo_resume.csv
    controle_multiclasse.json

  1. Harmonisation (--harmonisations zscore filtrage coral filtrage_coral), toujours ajustée DANS
     chaque pli sur les seuls témoins d'entraînement ; « aucune » sert de référence.
  2. Modèles : référence majoritaire, régression logistique multinomiale L2 (classes pondérées),
     MLP, XGBoost (--xgboost), SVM / forêt (--modeles), référence amplitude (log écart-type).
  3. Évaluation : StratifiedGroupKFold par sujet (strates = base × étiquette), --repetitions fois ;
     exactitude équilibrée, F1 macro, AUC un-contre-tous, Brier, rappel par classe, part des
     témoins de chaque base reconnus comme témoins.
  4. Contrôles de site :
     (a) prédire la base d'origine, tous les sujets, avant / après harmonisation ;
     (b) prédire la base des TÉMOINS seulement (hasard = 33 % si le site est neutralisé).
         Après z-score, la régression logistique y est au hasard PAR CONSTRUCTION ; le MLP
         (et XGBoost) testent ce qui reste (dispersion, covariance, forme des distributions) ;
     (c) tests de permutation : globale (toutes étiquettes mélangées) et intra-base
         (témoins/patients mélangés dans chaque base : le site reste, le trouble disparaît).
         Le « gain sur nul » intra-base est l'apport lié au trouble au-delà du site.
  5. Leave-one-dataset-out : trouble contre témoin, entraînement sur 2 bases, test sur la 3e ;
     base test harmonisée sur une partie de ses témoins (exclus de l'évaluation) ou sur tous
     ses sujets sans étiquette.

Usage :
    python 06_classification_multiclasse.py
    python 06_classification_multiclasse.py --xgboost --permutations 200
    python 06_classification_multiclasse.py --harmonisations zscore coral --coral-composantes 10
    python 06_classification_multiclasse.py --sans-permutations --sans-lodo --repetitions 3   # rapide

Dépendances : pip install numpy pandas scikit-learn matplotlib   (+ xgboost en option)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score

import eeg_harmonisation as eh
from eeg_config import BASES, CLASS_OF_BASE, CLASSES, GRAINE
from eeg_donnees import charger_fusion
from eeg_utils import (Compteur, arguments_dict, ecrire_json, entete, fit_predict, fmt_table, log, make_folds,
                       multiclass_scores, nouveau_dossier, p_valeur_permutation, plot_confusion, plt_agg,
                       summarise, versions, xgboost_disponible)


class Ctx:
    """Réglages des modèles (remplis depuis args)."""
    c = 1.0
    mlp_couches = (16,)
    mlp_alpha = 1.0


def fp(nom, seed, x_tr, y_tr, x_te, n):
    return fit_predict(nom, seed, x_tr, y_tr, x_te, n, Ctx.c, Ctx.mlp_couches, Ctx.mlp_alpha)[0]


def cross_val_proba(y, folds, prepared, nom, n_classes, seed):
    proba = np.zeros((len(y), n_classes))
    fold_id = np.zeros(len(y), dtype=int)
    for k, ((tr, te), (x_tr, x_te)) in enumerate(zip(folds, prepared), 1):
        proba[te] = fp(nom, seed, x_tr, y[tr], x_te, n_classes)
        fold_id[te] = k
    return proba, fold_id


def spec_id(spec):
    return f"{spec['modele']}__{spec['harmonisation']}"


def strates(meta):
    return pd.Categorical(meta["base"], BASES).codes * 2 + meta["label_binaire"].to_numpy()


# --------------------------------------------------------------------------- analyses
def main_analysis(x_by_feat, meta, args, specs):
    y = meta["classe"].to_numpy()
    bases = meta["base"].to_numpy()
    groups = meta["groupe"].to_numpy()
    strata = strates(meta)
    ref = y == 0
    labels = list(range(len(CLASSES)))
    temoin_base = np.where(y == 0, bases, "")
    row_id = np.where(y == 0, pd.Categorical(bases, BASES).codes, len(BASES) - 1 + y)
    rows, preds = [], []
    confusion = {spec_id(s): np.zeros((len(CLASSES), len(CLASSES)), dtype=int) for s in specs}
    detail = {spec_id(s): np.zeros((len(BASES) + 3, len(CLASSES)), dtype=int) for s in specs}
    for rep in range(args.repetitions):
        seed = args.graine + rep
        folds = make_folds(strata, groups, args.plis, seed, must_have=strata)
        cache = {}
        for spec in specs:
            t0 = time.time()
            key = (spec["caracteristiques"], spec["harmonisation"])
            if key not in cache:
                cache[key] = eh.prepare_folds(x_by_feat[key[0]], bases, ref, folds, key[1])
            proba, fold_id = cross_val_proba(y, folds, cache[key], spec["kind"], len(CLASSES), seed)
            row, pred = multiclass_scores(y, proba, labels)
            for c, cname in enumerate(CLASSES):
                row[f"rappel_{cname}"] = recall_score(y == c, pred == c, zero_division=0)
            for b in BASES:
                row[f"temoins_{b}_bien_classes"] = float(np.mean(pred[temoin_base == b] == 0))
            row["nombre_variables"] = cache[key][0][0].shape[1]
            rows.append(dict(tache="multiclasse", modele=spec["modele"], harmonisation=spec["harmonisation"],
                             repetition=rep + 1, **row))
            sid = spec_id(spec)
            confusion[sid] += confusion_matrix(y, pred, labels=labels)
            np.add.at(detail[sid], (row_id, pred), 1)
            preds.append(pd.DataFrame({
                "tache": "multiclasse", "harmonisation": spec["harmonisation"], "modele": spec["modele"],
                "repetition": rep + 1, "subject_id": meta.index, "base": bases, "groupe": groups, "pli": fold_id,
                "classe": y, "prediction": pred, **{f"proba_{c}": proba[:, i] for i, c in enumerate(CLASSES)}}))
            if rep == 0:
                log(f"  {spec['modele']:<26s} {spec['harmonisation']:<15s} répétition 1 : "
                    f"{row['exactitude_equilibree']:.3f} ({time.time() - t0:.1f} s)")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), confusion, detail


def site_controls(x_by_feat, meta, args, kinds):
    """(a) tous les sujets et (b) témoins seulement : prédire la base d'origine."""
    rows = []
    base_codes = pd.Categorical(meta["base"], BASES).codes.astype(int)
    methods = ["aucune"] + args.harmonisations
    for population in ["tous_les_sujets", "temoins_seulement"]:
        mask = np.ones(len(meta), bool) if population == "tous_les_sujets" else meta["classe"].eq(0).to_numpy()
        y = base_codes[mask]
        bases = meta["base"].to_numpy()[mask]
        groups = meta["groupe"].to_numpy()[mask]
        lab = meta["label_binaire"].to_numpy()[mask]
        strata = y * 2 + lab
        ref = lab == 0
        t0 = time.time()
        for rep in range(args.repetitions):
            seed = args.graine + rep
            folds = make_folds(strata, groups, args.plis, seed, must_have=strata)
            for feat in ["entropies", "amplitude"]:
                for method in methods:
                    prepared = eh.prepare_folds(x_by_feat[feat][mask], bases, ref, folds, method)
                    for kind in kinds:
                        proba, _ = cross_val_proba(y, folds, prepared, kind, len(BASES), seed)
                        row, _ = multiclass_scores(y, proba, list(range(len(BASES))))
                        rows.append(dict(population=population, caracteristiques=feat, harmonisation=method,
                                         modele=kind, repetition=rep + 1, **row))
        log(f"  {population} : {time.time() - t0:.0f} s")
    return pd.DataFrame(rows)


def permute_labels(meta, rng, mode):
    """Permutation au niveau des groupes (un sujet = une étiquette)."""
    g = meta.groupby("groupe").agg(base=("base", "first"), label=("label_binaire", "first"))
    if mode == "globale":
        cls = np.where(g["label"].eq(1), g["base"].map(CLASS_OF_BASE), 0)
        new_cls = pd.Series(rng.permutation(cls), index=g.index)
        classe = meta["groupe"].map(new_cls).to_numpy()
        lab = (classe != 0).astype(int)
    else:
        new_lab = g.groupby("base")["label"].transform(
            lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
        lab = meta["groupe"].map(new_lab).to_numpy()
        classe = np.where(lab == 1, meta["base"].map(CLASS_OF_BASE), 0)
    return classe.astype(int), lab.astype(int)


def permutation_tests(x_by_feat, meta, args, observed, specs):
    rows = []
    bases = meta["base"].to_numpy()
    groups = meta["groupe"].to_numpy()
    base_codes = pd.Categorical(bases, BASES).codes
    labels = list(range(len(CLASSES)))
    for spec in specs:
        for mode in ["globale", "intra_base"]:
            rng = np.random.default_rng(args.graine)   # mêmes permutations pour chaque modèle
            t0 = time.time()
            scores = []
            for i in range(args.permutations):
                classe, lab = permute_labels(meta, rng, mode)
                seed = args.graine + 10_000 + i
                try:
                    folds = make_folds(base_codes * 2 + lab, groups, args.plis, seed)
                    prepared = eh.prepare_folds(x_by_feat[spec["caracteristiques"]], bases, classe == 0, folds,
                                                spec["harmonisation"])
                    proba, _ = cross_val_proba(classe, folds, prepared, spec["kind"], len(CLASSES), seed)
                except ValueError as exc:
                    log(f"    permutation {i + 1} ignorée : {exc}")
                    continue
                scores.append(multiclass_scores(classe, proba, labels)[0])
            perm = pd.DataFrame(scores)
            obs = observed.loc[(observed["modele"] == spec["modele"]) & (observed["harmonisation"] == spec["harmonisation"])]
            for metric in ["exactitude_equilibree", "f1_macro"]:
                o = obs[metric].mean()
                null = perm[metric].to_numpy()
                rows.append(dict(modele=spec["modele"], harmonisation=spec["harmonisation"], permutation=mode,
                                 metrique=metric, observe=o, moyenne_nulle=null.mean(), ecart_type_nul=null.std(ddof=1),
                                 quantile_95_nul=np.quantile(null, 0.95), gain_sur_nul=o - null.mean(),
                                 p_valeur=p_valeur_permutation(o, null), n_permutations=len(null)))
            log(f"  {spec['modele']} [{spec['harmonisation']}] / {mode} : {len(perm)} permutations ({time.time() - t0:.0f} s)")
    return pd.DataFrame(rows)


def lodo(x_by_feat, meta, args, specs):
    rows = []
    bases = meta["base"].to_numpy()
    y = meta["label_binaire"].to_numpy()
    groups = meta["groupe"].to_numpy()
    for test_base in BASES:
        tr = bases != test_base
        te_all = ~tr
        for mode in ["calibration_temoins", "base_entiere"]:
            for rep in range(args.repetitions):
                seed = args.graine + rep
                rng = np.random.default_rng(seed)
                te = te_all.copy()
                if mode == "calibration_temoins":
                    # Une fraction des témoins de la base test sert à l'harmoniser et n'est PAS évaluée.
                    tg = np.unique(groups[te_all & (y == 0)])
                    n_cal = max(3, int(round(args.lodo_calibration * len(tg))))
                    cal = te_all & np.isin(groups, rng.choice(tg, size=min(n_cal, len(tg) - 1), replace=False))
                    te = te_all & ~cal
                    ref_tr, ref_test = y[tr] == 0, cal
                else:
                    # Sans étiquette : chaque base harmonisée sur tous ses sujets.
                    ref_tr, ref_test = np.ones(tr.sum(), bool), te_all
                cache = {}
                for spec in specs:
                    key = (spec["caracteristiques"], spec["harmonisation"])
                    if key not in cache:
                        x = x_by_feat[key[0]]
                        cache[key] = eh.harmonise(x[tr], x[te], bases[tr], bases[te], ref_tr, key[1],
                                                  ref_te_stats={test_base: x[ref_test]})
                    x_tr, x_te = cache[key]
                    proba = fp(spec["kind"], seed, x_tr, y[tr], x_te, 2)
                    row, pred = multiclass_scores(y[te], proba, [0, 1])
                    row["sensibilite"] = recall_score(y[te], pred, zero_division=0)
                    row["specificite"] = recall_score(y[te], pred, pos_label=0, zero_division=0)
                    rows.append(dict(base_test=test_base, harmonisation_base_test=mode, modele=spec["modele"],
                                     harmonisation=spec["harmonisation"], repetition=rep + 1,
                                     n_test=int(te.sum()), temoins_test=int((y[te] == 0).sum()), **row))
        log(f"  base test {test_base} : terminé")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def plot_confusions(confusion, detail, out):
    plt = plt_agg()
    paths = []
    detail_rows = [f"témoins {b}" for b in BASES] + ["dépression", "schizophrénie", "anxiété"]
    col_labels = ["témoin", "dépression", "schizo.", "anxiété"]
    for name in confusion:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
        plot_confusion(confusion[name], col_labels, col_labels, "Matrice 4 classes", None, ax=axes[0])
        plot_confusion(detail[name], detail_rows, col_labels, "Témoins séparés par base d'origine", None, ax=axes[1])
        modele, harm = name.split("__")
        fig.suptitle(f"{modele}, harmonisation {harm} : proportions par ligne (effectifs cumulés sur les répétitions)")
        fig.tight_layout()
        path = out / f"matrice_confusion_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path.name)
    return paths


def plot_site_controls(site_summary, out):
    plt = plt_agg()
    d = site_summary.query("population == 'temoins_seulement'")
    methods = list(dict.fromkeys(d["harmonisation"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, feat in zip(axes, ["entropies", "amplitude"]):
        sub = d[d["caracteristiques"] == feat]
        models = list(dict.fromkeys(sub["modele"]))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            s = sub[sub["modele"] == m].set_index("harmonisation").reindex(methods)
            ax.bar(np.arange(len(methods)) + i * width - 0.4 + width / 2, s["exactitude_equilibree"],
                   width, yerr=s["exactitude_equilibree_std"], label=m, capsize=3)
        ax.axhline(1 / 3, color="black", linestyle="--", linewidth=1, label="hasard")
        ax.set_xticks(range(len(methods)), methods, rotation=20)
        ax.set_title(f"Témoins seulement, {feat}")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Exactitude équilibrée (prédire la base)")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Contrôle (b) : plus c'est proche du hasard, mieux l'effet de site est neutralisé")
    fig.tight_layout()
    fig.savefig(out / "controle_site_temoins.png", dpi=150)
    plt.close(fig)
    return "controle_site_temoins.png"


# --------------------------------------------------------------------------- principal
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fusion", type=Path, default=None, help="Dossier de 04_fusion (défaut : le plus récent).")
    p.add_argument("--harmonisations", nargs="+", choices=eh.HARMONISATIONS, default=eh.HARMONISATIONS)
    p.add_argument("--modeles", nargs="+", default=["regression_logistique", "mlp"],
                   choices=["regression_logistique", "mlp", "svm_rbf", "foret_aleatoire", "lda", "xgboost"])
    p.add_argument("--xgboost", action="store_true", help="Ajouter XGBoost (paquet xgboost requis).")
    p.add_argument("--coral-composantes", type=int, default=20)
    p.add_argument("--filtrage-fraction", type=float, default=0.5)
    p.add_argument("--filtrage-arbres", type=int, default=200)
    p.add_argument("--plis", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=10)
    p.add_argument("--permutations", type=int, default=100)
    p.add_argument("--permutation-modeles", nargs="+", default=["regression_logistique"],
                   choices=["regression_logistique", "mlp", "svm_rbf", "foret_aleatoire", "lda", "xgboost",
                            "reference_amplitude"])
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--mlp-couches", type=int, nargs="+", default=[16])
    p.add_argument("--mlp-alpha", type=float, default=1.0)
    p.add_argument("--lodo-calibration", type=float, default=0.5)
    p.add_argument("--sans-controles-site", action="store_true")
    p.add_argument("--sans-permutations", action="store_true")
    p.add_argument("--sans-lodo", action="store_true")
    args = p.parse_args()
    if args.plis < 2 or args.repetitions < 1 or args.permutations < 1:
        p.error("--plis >= 2, --repetitions >= 1, --permutations >= 1.")
    if not 0 < args.lodo_calibration < 1 or not 0 < args.filtrage_fraction < 1:
        p.error("--lodo-calibration et --filtrage-fraction entre 0 et 1 (exclus).")
    if args.xgboost and "xgboost" not in args.modeles:
        args.modeles.append("xgboost")
    if "xgboost" in args.modeles + args.permutation_modeles and not xgboost_disponible():
        p.error("xgboost demandé mais absent (pip install xgboost).")
    args.harmonisations = list(dict.fromkeys(args.harmonisations))
    eh.configurer(args)
    Ctx.c, Ctx.mlp_couches, Ctx.mlp_alpha = args.c, tuple(args.mlp_couches), args.mlp_alpha

    start = time.time()
    try:
        common, ent, amp, meta, finfo, fdir = charger_fusion(args.fusion)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Chargement arrêté : {exc}\n")
    x_by_feat = {"entropies": ent.to_numpy(float), "amplitude": amp.to_numpy(float)}

    entete("CLASSIFICATION MULTI-CLASSE AVEC HARMONISATION")
    log(f"Fusion : {fdir}")
    log(f"Canaux communs ({len(common)}) : {', '.join(common)}")
    counts = meta["classe"].value_counts().sort_index()
    log("Classes : " + ", ".join(f"{i}={CLASSES[i]} ({counts.get(i, 0)})" for i in range(len(CLASSES))))
    log(f"Variables : {ent.shape[1]} entropies, {amp.shape[1]} amplitudes")
    log(f"Harmonisations : {', '.join(args.harmonisations)} (+ « aucune ») ; modèles : {', '.join(args.modeles)}")
    log(f"Validation : StratifiedGroupKFold {args.plis} plis × {args.repetitions} répétitions "
        "(strates = base × étiquette ; harmonisation ajustée dans chaque pli sur les témoins d'entraînement)")

    specs = [dict(modele="reference_majoritaire", caracteristiques="entropies", kind="reference_majoritaire",
                  harmonisation=args.harmonisations[0])]
    for h in args.harmonisations:
        specs += [dict(modele=k, caracteristiques="entropies", kind=k, harmonisation=h) for k in args.modeles]
        specs.append(dict(modele="reference_amplitude", caracteristiques="amplitude", kind="regression_logistique",
                          harmonisation=h))
    specs += [dict(modele="regression_logistique", caracteristiques="entropies", kind="regression_logistique",
                   harmonisation="aucune"),
              dict(modele="reference_amplitude", caracteristiques="amplitude", kind="regression_logistique",
                   harmonisation="aucune")]
    out = nouveau_dossier("multiclasse")
    figures, site_summary, perm, lodo_summary = [], None, None, None
    try:
        log("\n[1/4] Classification 4 classes")
        metrics, preds, confusion, detail = main_analysis(x_by_feat, meta, args, specs)
        summary = summarise(metrics, ["tache", "harmonisation", "modele"])
        metrics.to_csv(out / "multiclasse_metriques.csv", index=False)
        summary.to_csv(out / "multiclasse_resume.csv", index=False)
        preds.to_csv(out / "predictions_hors_pli.csv", index=False)
        figures += plot_confusions(confusion, detail, out)

        if not args.sans_controles_site:
            log("\n[2/4] Contrôles de site (a) et (b)")
            kinds = [k for k in ["regression_logistique", "mlp", "xgboost"] if k in args.modeles] or ["regression_logistique"]
            site = site_controls(x_by_feat, meta, args, kinds)
            site_summary = summarise(site, ["population", "caracteristiques", "harmonisation", "modele"])
            site.to_csv(out / "controle_site_metriques.csv", index=False)
            site_summary.to_csv(out / "controle_site_resume.csv", index=False)
            figures.append(plot_site_controls(site_summary, out))

        if not args.sans_permutations:
            log(f"\n[3/4] Tests de permutation ({args.permutations} par type et par harmonisation)")
            pspecs = [s for s in specs if s["modele"] in args.permutation_modeles and s["harmonisation"] != "aucune"]
            perm = permutation_tests(x_by_feat, meta, args, metrics, pspecs)
            perm.to_csv(out / "permutations.csv", index=False)

        if not args.sans_lodo:
            log("\n[4/4] Leave-one-dataset-out (trouble contre témoin)")
            lspecs = [s for s in specs if s["harmonisation"] != "aucune"]
            lodo_df = lodo(x_by_feat, meta, args, lspecs)
            lodo_summary = summarise(lodo_df, ["base_test", "harmonisation_base_test", "harmonisation", "modele"])
            lodo_df.to_csv(out / "lodo_metriques.csv", index=False)
            lodo_summary.to_csv(out / "lodo_resume.csv", index=False)
    except ValueError as exc:
        p.exit(1, f"Analyse arrêtée : {exc}\n")

    # ------------------------------------------------------------------ résumé console
    order = {h: i for i, h in enumerate(["aucune"] + args.harmonisations)}
    entete(f"RÉSULTATS 4 CLASSES (moyenne ± écart-type sur {args.repetitions} répétitions ; hasard = 0.25)")
    s = summary.copy()
    s["_o"] = s["harmonisation"].map(order)
    s = s.sort_values(["_o"], kind="stable")
    for m in ["exactitude_equilibree", "f1_macro", "auc_ovr_macro"]:
        s[m + "_txt"] = s[m].map("{:.3f}".format) + " ± " + s[m + "_std"].map("{:.3f}".format)
    log(s[["harmonisation", "modele", "nombre_variables", "exactitude_equilibree_txt", "f1_macro_txt", "auc_ovr_macro_txt"]]
        .rename(columns={"exactitude_equilibree_txt": "exact_equilibree", "f1_macro_txt": "f1_macro",
                         "auc_ovr_macro_txt": "auc_ovr", "nombre_variables": "variables"})
        .to_string(index=False, float_format=lambda v: f"{v:.0f}"))
    log("\nRappel par classe et part des témoins de chaque base classés « témoin » :")
    log(fmt_table(s, ["harmonisation", "modele"] + [f"rappel_{c}" for c in CLASSES]
                  + [f"temoins_{b}_bien_classes" for b in BASES]))

    if site_summary is not None:
        log("\n" + "-" * 78 + "\nCONTRÔLES DE SITE : prédire la base d'origine (exactitude équilibrée ; hasard = 0.33)\n" + "-" * 78)
        piv = site_summary.pivot_table(index=["population", "caracteristiques", "modele"], columns="harmonisation",
                                       values="exactitude_equilibree", sort=False)
        log(piv[[c for c in ["aucune"] + args.harmonisations if c in piv.columns]].to_string(float_format=lambda v: f"{v:.3f}"))
        t = site_summary.query("population == 'temoins_seulement' and caracteristiques == 'entropies' "
                               "and modele != 'regression_logistique'")
        if len(t):
            log("\n→ (b) témoins seulement, entropies, meilleur modèle non linéaire, par harmonisation :")
            for h in args.harmonisations:
                sub = t[t["harmonisation"] == h]
                if not len(sub):
                    continue
                best = sub.loc[sub["exactitude_equilibree"].idxmax()]
                v = best["exactitude_equilibree"]
                verdict = ("effet de site NON neutralisé" if v > 0.45 else
                           "effet de site largement neutralisé" if v < 0.40 else "effet de site résiduel modéré")
                log(f"   {h:<15s} {best['modele']:<8s} {v:.3f} ± {best['exactitude_equilibree_std']:.3f} : {verdict}")

    if perm is not None:
        log("\n" + "-" * 78 + "\nTESTS DE PERMUTATION (exactitude équilibrée)\n" + "-" * 78)
        q = perm[perm["metrique"] == "exactitude_equilibree"].copy()
        q["_o"] = q["harmonisation"].map(order)
        log(fmt_table(q.sort_values(["permutation", "_o"], kind="stable"),
                      ["modele", "harmonisation", "permutation", "observe", "moyenne_nulle", "quantile_95_nul",
                       "gain_sur_nul", "p_valeur"]))
        log("  « intra_base » : patients/témoins mélangés dans chaque base ; le gain mesure l'apport lié au trouble.")

    if lodo_summary is not None:
        log("\n" + "-" * 78 + "\nLEAVE-ONE-DATASET-OUT : trouble contre témoin (exactitude équilibrée ; hasard = 0.50)\n" + "-" * 78)
        lp = lodo_summary.pivot_table(index=["base_test", "harmonisation_base_test", "modele"], columns="harmonisation",
                                      values="exactitude_equilibree", sort=False)
        log(lp[[c for c in args.harmonisations if c in lp.columns]].to_string(float_format=lambda v: f"{v:.3f}"))

    if Compteur.non_converges:
        log(f"\nAvertissement : {Compteur.non_converges} ajustements non convergés (voir max_iter).")
    ecrire_json(out / "controle_multiclasse.json", {
        "portee": "Évaluation exploratoire ; ordre des électrodes schizophrénie/anxiété supposé.",
        "fusion": str(fdir), "dossiers_extraction": finfo.get("dossiers_extraction"),
        "canaux_communs": common,
        "effectifs": finfo.get("effectifs"),
        "harmonisations": {
            "zscore": "z-score par base sur les témoins d'entraînement du pli",
            "filtrage": f"z-score puis retrait de {args.filtrage_fraction:.0%} des variables prédisant la base "
                        f"(forêt d'arbres extrêmes, {args.filtrage_arbres} arbres)",
            "coral": f"z-score, ACP {args.coral_composantes} composantes sur les témoins, CORAL Ledoit-Wolf",
            "filtrage_coral": "filtrage puis CORAL"},
        "validation": f"StratifiedGroupKFold {args.plis} plis, graines {args.graine}..{args.graine + args.repetitions - 1}",
        "figures": figures, "ajustements_non_converges": Compteur.non_converges,
        "duree_s": round(time.time() - start, 1), "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "matplotlib", "xgboost"))})
    log(f"\nDurée : {time.time() - start:.0f} s")
    log(f"SORTIE : {out}")


if __name__ == "__main__":
    main()
