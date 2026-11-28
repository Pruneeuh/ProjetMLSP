"""Étape 10 : synthèse des résultats pour le rapport (intervalles de confiance, calibration,
comparaisons de modèles, tableaux Markdown / LaTeX).

Entrée  : les dossiers les plus récents de 05_binaire, 06_multiclasse et 07_reseaux_neurones
          (ou --binaire, --multiclasse, --profond ; une étape absente est ignorée)
Sortie  : sorties/10_synthese/<date>/
    synthese.csv           une ligne par tâche × harmonisation × modèle :
                           moyenne ± écart-type sur les répétitions (exactitude équilibrée, AUC,
                           F1 macro, Brier) et IC 95 % bootstrap (rééchantillonnage des sujets,
                           probabilités moyennées sur les répétitions, sur test ou hors pli selon la source)
    comparaisons.csv       différence d'exactitude équilibrée avec la régression logistique de la
                           même tâche (IC 95 % bootstrap apparié, part des tirages > 0)
    metriques_par_classe.csv précision, rappel, spécificité, F1 et IC bootstrap du rappel
    shap_principales.csv   dix variables SHAP les plus importantes par tâche et classe (si disponibles)
    ablation.csv           résultats d'ablation de l'étape 05 (si disponibles)
    synthese.md, synthese.tex   tableaux prêts à copier dans le rapport
    calibration_<tâche>.png     diagrammes de fiabilité (tâches binaires)
    synthese_exactitude.png     meilleurs modèles par tâche avec IC 95 %
    synthese.json

Usage :
    python 10_synthese_resultats.py
    python 10_synthese_resultats.py --bootstrap 5000 --binaire sorties/05_binaire/20261124_...

Dépendances : pip install numpy pandas scikit-learn matplotlib
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_config import CLASSES, CLASSES_BINAIRES, GRAINE
from eeg_utils import (arguments_dict, bootstrap_ic, dernier_dossier, ecrire_json, entete, log, nouveau_dossier,
                       plt_agg, versions)

SOURCES = {"binaire": "05_classification_binaire", "multiclasse": "06_classification_multiclasse",
           "profond": "07_reseaux_neurones"}


def lire_predictions(dossier, source):
    path = Path(dossier) / "predictions_test.csv"
    if not path.exists():
        path = Path(dossier) / "predictions_hors_pli.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["source"] = source
    df["validation"] = "test_80_20" if path.name == "predictions_test.csv" else "validation_croisee"
    if "harmonisation" not in df:
        df["harmonisation"] = "aucune"
    return df


def colonnes_proba(df):
    if df["tache"].iloc[0] == "multiclasse":
        return [f"proba_{c}" for c in CLASSES]
    if "proba_trouble" in df:
        return ["proba_temoin", "proba_trouble"]
    return [f"proba_{c}" for c in CLASSES_BINAIRES]


def moyenne_repetitions(g, cols):
    """Probabilités moyennées sur les répétitions, une ligne par sujet."""
    agg = g.groupby("subject_id").agg(classe=("classe", "first"), groupe=("groupe", "first"),
                                      **{c: (c, "mean") for c in cols})
    return agg


def metriques(y, proba, labels, avec_auc=True):
    """Exactitude équilibrée, AUC (un-contre-tous en 4 classes), Brier, F1 macro (version rapide)."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    p = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    pred = p.argmax(axis=1)
    presentes = [c for c in labels if np.any(y == c)]
    rappels = [np.mean(pred[y == c] == c) for c in presentes]
    prec = [np.mean(y[pred == c] == c) if np.any(pred == c) else 0.0 for c in labels]
    rec_all = [np.mean(pred[y == c] == c) if np.any(y == c) else 0.0 for c in labels]
    f1 = [2 * a * b / (a + b) if a + b > 0 else 0.0 for a, b in zip(prec, rec_all)]
    onehot = (y[:, None] == np.asarray(labels)[None, :]).astype(float)
    row = dict(exactitude_equilibree=float(np.mean(rappels)), f1_macro=float(np.mean(f1)),
               brier=float(np.mean(np.sum((p - onehot) ** 2, axis=1))))
    key = "auc" if len(labels) == 2 else "auc_ovr_macro"
    if not avec_auc or len(presentes) < len(labels):
        row[key] = np.nan
    elif len(labels) == 2:
        row[key] = roc_auc_score(y, p[:, 1])
    else:
        row[key] = float(np.mean([roc_auc_score(y == c, p[:, c]) for c in labels]))
    return row


def bootstrap_multi(y, pr, labels, groupes, n_boot, seed, cles):
    """IC 95 % percentile de plusieurs métriques, mêmes tirages (rééchantillonnage des groupes)."""
    rng = np.random.default_rng(seed)
    uniques = np.unique(groupes)
    index = {g: np.flatnonzero(groupes == g) for g in uniques}
    vals = {k: [] for k in cles}
    for _ in range(n_boot):
        idx = np.concatenate([index[g] for g in rng.choice(uniques, size=len(uniques), replace=True)])
        r = metriques(y[idx], pr[idx], labels)
        for k in cles:
            if np.isfinite(r[k]):
                vals[k].append(r[k])
    return {k: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (np.nan, np.nan)
            for k, v in vals.items()}


def rappel_classe(y, proba, classe):
    mask = y == classe
    return float(np.mean(proba.argmax(axis=1)[mask] == classe)) if mask.any() else np.nan


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binaire", type=Path, default=None)
    p.add_argument("--multiclasse", type=Path, default=None)
    profond = p.add_mutually_exclusive_group()
    profond.add_argument("--profond", type=Path, default=None)
    profond.add_argument("--sans-profond", action="store_true", help="Ne pas inclure les résultats de l'étape 07.")
    p.add_argument("--interpretabilite", type=Path, default=None)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--reference", default="regression_logistique", help="Modèle de comparaison.")
    args = p.parse_args()
    def dernier_resultat(etape):
        candidats = [dernier_dossier(etape, contient=nom)
                     for nom in ("predictions_test.csv", "predictions_hors_pli.csv")]
        return max((p for p in candidats if p is not None), default=None)

    dossiers = {
        "binaire": args.binaire or dernier_resultat("binaire"),
        "multiclasse": args.multiclasse or dernier_resultat("multiclasse"),
        "profond": None if args.sans_profond else args.profond or dernier_dossier("profond", contient="predictions_hors_pli.csv"),
    }
    entete("SYNTHÈSE DES RÉSULTATS")
    frames = []
    for src, d in dossiers.items():
        if d is None:
            log(f"  {src} : aucun résultat ({SOURCES[src]}.py non lancé)")
            continue
        df = lire_predictions(d, src)
        if df is None:
            continue
        log(f"  {src:<12s} {d} ({df['validation'].iloc[0]})")
        frames.append(df)
    if not frames:
        p.exit(1, "Aucun fichier de prédictions trouvé.\n")
    out = nouveau_dossier("synthese")
    notes = ["Les résultats test 80/20 portent sur les mêmes sujets test pour toutes les répétitions ; "
             "la sélection du meilleur modèle sur ce test rend sa performance optimiste."]
    if dossiers["binaire"] is not None:
        path_ablation = Path(dossiers["binaire"]) / "ablation.csv"
        if path_ablation.is_file():
            pd.read_csv(path_ablation).to_csv(out / "ablation.csv", index=False)
            notes.append("Ablations binaires : voir ablation.csv.")
        else:
            log("  Ablations binaires absentes de la source 05.")
    interpretation = args.interpretabilite or dernier_dossier("interpretabilite", contient="shap_valeurs.csv")
    if interpretation is not None:
        path_shap = Path(interpretation) / "shap_valeurs.csv"
        if not path_shap.is_file():
            p.exit(1, f"Fichier SHAP introuvable : {path_shap}\n")
        shap = pd.read_csv(path_shap)
        needed = {"tache", "classe", "variable", "shap_abs_moyen"}
        if not needed <= set(shap.columns):
            p.exit(1, f"Colonnes SHAP manquantes : {sorted(needed - set(shap.columns))}\n")
        shap.sort_values("shap_abs_moyen", ascending=False).groupby(
            ["tache", "classe"], sort=False).head(10).to_csv(out / "shap_principales.csv", index=False)
        notes.append(f"SHAP de 09 : modèles distincts sur données agrégées par sujet, pas les modèles "
                     f"sur fenêtres de 05/06 ; peut inclure leurs sujets test (source : {interpretation}).")
    else:
        log("  SHAP absent : lancer 09 avec --shap.")
    rows, comps, figures, classes_rows = [], [], [], []
    oof = {}
    for df in frames:
        for (tache, harm, modele), g in df.groupby(["tache", "harmonisation", "modele"], sort=False):
            cols = colonnes_proba(g)
            labels = list(range(len(cols)))
            per_rep = pd.DataFrame([metriques(r["classe"].to_numpy(), r[cols].to_numpy(), labels)
                                    for _, r in g.groupby("repetition")])
            m = moyenne_repetitions(g, cols)
            y, pr, grp = m["classe"].to_numpy(), m[cols].to_numpy(), m["groupe"].to_numpy()
            oof[(g["source"].iloc[0], tache, harm, modele)] = m
            ens = metriques(y, pr, labels)
            auc_key = "auc" if len(labels) == 2 else "auc_ovr_macro"
            ic = bootstrap_multi(y, pr, labels, grp, args.bootstrap, args.graine,
                                 ["exactitude_equilibree", auc_key, "brier"])
            pred = pr.argmax(axis=1)
            for c in labels:
                rappel = rappel_classe(y, pr, c)
                lo, hi = bootstrap_ic(lambda idx: rappel_classe(y[idx], pr[idx], c),
                                      len(y), grp, args.bootstrap, args.graine)
                tp = int(np.sum((y == c) & (pred == c)))
                fp = int(np.sum((y != c) & (pred == c)))
                tn = int(np.sum((y != c) & (pred != c)))
                fn = int(np.sum((y == c) & (pred != c)))
                classes_rows.append(dict(source=g["source"].iloc[0], tache=tache, harmonisation=harm,
                                         modele=modele, classe=(CLASSES if len(labels) == 4 else CLASSES_BINAIRES)[c],
                                         effectif=int(np.sum(y == c)), rappel=rappel,
                                         ic95_rappel_bas=lo, ic95_rappel_haut=hi,
                                         precision=tp / (tp + fp) if tp + fp else np.nan,
                                         specificite=tn / (tn + fp) if tn + fp else np.nan,
                                         f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan))
            rows.append(dict(source=g["source"].iloc[0], validation=g["validation"].iloc[0],
                             tache=tache, harmonisation=harm, modele=modele,
                             n_sujets=len(y), n_repetitions=int(g["repetition"].nunique()),
                             exactitude_equilibree=per_rep["exactitude_equilibree"].mean(),
                             exactitude_equilibree_std=per_rep["exactitude_equilibree"].std(ddof=1),
                             auc=per_rep[auc_key].mean(), f1_macro=per_rep["f1_macro"].mean(),
                             brier=per_rep["brier"].mean(),
                             exactitude_equilibree_moyennee=ens["exactitude_equilibree"],
                             ic95_exactitude_bas=ic["exactitude_equilibree"][0],
                             ic95_exactitude_haut=ic["exactitude_equilibree"][1],
                             ic95_auc_bas=ic[auc_key][0], ic95_auc_haut=ic[auc_key][1],
                             ic95_brier_bas=ic["brier"][0], ic95_brier_haut=ic["brier"][1]))
    syn = pd.DataFrame(rows)
    hasard = {t: (0.25 if t == "multiclasse" else 0.5) for t in syn["tache"]}
    syn["hasard"] = syn["tache"].map(hasard)
    syn.to_csv(out / "synthese.csv", index=False)
    pd.DataFrame(classes_rows).to_csv(out / "metriques_par_classe.csv", index=False)

    # Comparaison appariée avec la référence (mêmes sujets, mêmes plis dans une même source)
    rng_seed = args.graine
    for (src, tache, harm, modele), m in oof.items():
        ref = oof.get((src, tache, harm, args.reference))
        if ref is None or modele == args.reference:
            continue
        common = m.index.intersection(ref.index)
        cols = [c for c in m.columns if c.startswith("proba_")]
        labels = list(range(len(cols)))
        y = m.loc[common, "classe"].to_numpy()
        pa, pb = m.loc[common, cols].to_numpy(), ref.loc[common, cols].to_numpy()
        grp = m.loc[common, "groupe"].to_numpy()
        f = lambda idx: (metriques(y[idx], pa[idx], labels, False)["exactitude_equilibree"]
                         - metriques(y[idx], pb[idx], labels, False)["exactitude_equilibree"])
        lo, hi = bootstrap_ic(f, len(y), grp, args.bootstrap, rng_seed)
        comps.append(dict(source=src, tache=tache, harmonisation=harm, modele=modele, reference=args.reference,
                          difference=f(np.arange(len(y))), ic95_bas=lo, ic95_haut=hi,
                          conclusion="meilleur" if lo > 0 else "moins bon" if hi < 0 else "non distinguable"))
    comp = pd.DataFrame(comps)
    if len(comp):
        comp.to_csv(out / "comparaisons.csv", index=False)

    # Tableaux pour le rapport
    syn["exact_eq"] = (syn["exactitude_equilibree"] * 100).map("{:.1f}".format) + " ± " + \
        (syn["exactitude_equilibree_std"].fillna(0) * 100).map("{:.1f}".format)
    syn["ic95"] = "[" + (syn["ic95_exactitude_bas"] * 100).map("{:.1f}".format) + " ; " + \
        (syn["ic95_exactitude_haut"] * 100).map("{:.1f}".format) + "]"
    syn["auc_txt"] = syn["auc"].map("{:.3f}".format) + " [" + syn["ic95_auc_bas"].map("{:.2f}".format) + \
        " ; " + syn["ic95_auc_haut"].map("{:.2f}".format) + "]"
    table = syn[["tache", "harmonisation", "modele", "n_sujets", "exact_eq", "ic95", "auc_txt", "brier"]].rename(columns={
        "tache": "Tâche", "harmonisation": "Harmonisation", "modele": "Modèle", "n_sujets": "n",
        "exact_eq": "Exact. équilibrée (%)", "ic95": "IC 95 % (%)", "auc_txt": "AUC [IC 95 %]", "brier": "Brier"})
    table = table.sort_values(["Tâche", "Harmonisation", "Exact. équilibrée (%)"], ascending=[True, True, False])
    lignes = ["| " + " | ".join(table.columns) + " |", "|" + "---|" * len(table.columns)]
    lignes += ["| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
               for r in table.itertuples(index=False)]
    (out / "synthese.md").write_text(
        "# Synthèse des résultats\n\nMoyenne ± écart-type sur les répétitions ; IC 95 % bootstrap par sujet "
        "(probabilités moyennées sur les répétitions). Les métriques par classe et leurs IC de rappel "
        "sont dans metriques_par_classe.csv. Les IC conditionnent sur les prédictions disponibles "
        "et ne comprennent pas toute l'incertitude d'entraînement ou de sélection du modèle.\n\n"
        "La base et le trouble sont confondus en multiclasse : l'harmonisation ne démontre pas une "
        "généralisation clinique. Les étiquettes d'anxiété proviennent du dépistage GAD-7. "
        "Les explications SHAP décrivent le modèle et ne démontrent pas de causalité.\n\n" +
        "\n".join(notes) + "\n\n" +
        "\n".join(lignes) + "\n", encoding="utf-8")
    try:
        tex = table.to_latex(index=False, float_format="%.3f", escape=True,
                             caption="Performances test ou hors pli selon la source (moyenne $\\pm$ écart-type ; IC 95~\\% bootstrap par groupe).",
                             label="tab:synthese")
        (out / "synthese.tex").write_text(tex, encoding="utf-8")
    except ImportError as exc:   # pandas >= 2 : to_latex exige jinja2
        log(f"  synthese.tex non écrit ({exc}) : pip install jinja2")

    # Figures
    plt = plt_agg()
    from sklearn.calibration import calibration_curve
    taches_bin = sorted({k[1] for k in oof if k[1].startswith("binaire_")})
    for t in taches_bin:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for (src, tache, harm, modele), m in oof.items():
            if tache != t or modele == "reference_majoritaire" or modele.startswith("reference_canaux"):
                continue
            col = "proba_trouble" if "proba_trouble" in m else m.columns[-1]
            try:
                fy, fx = calibration_curve(m["classe"], m[col], n_bins=5, strategy="quantile")
            except ValueError:
                continue
            ax.plot(fx, fy, marker="o", label=f"{modele} ({src})", alpha=0.8)
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="calibration parfaite")
        ax.set_xlabel("Probabilité prédite (trouble)")
        ax.set_ylabel("Fréquence observée")
        ax.set_title(f"{t} : diagramme de fiabilité", fontsize=10)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"calibration_{t}.png", dpi=150)
        plt.close(fig)
        figures.append(f"calibration_{t}.png")

    best = (syn[~syn.modele.str.startswith("reference_")].sort_values("exactitude_equilibree", ascending=False)
            .groupby(["tache"], sort=True).head(3))
    if len(best):
        best = best.sort_values(["tache", "exactitude_equilibree_moyennee"], ascending=[True, True])
        fig, ax = plt.subplots(figsize=(9, 0.42 * len(best) + 1.5))
        lab = best["tache"] + " · " + best["modele"] + " · " + best["harmonisation"]
        xerr = np.vstack([best["exactitude_equilibree_moyennee"] - best["ic95_exactitude_bas"],
                          best["ic95_exactitude_haut"] - best["exactitude_equilibree_moyennee"]]).clip(0)
        pos = np.arange(len(best))
        ax.barh(pos, best["exactitude_equilibree_moyennee"], xerr=xerr, capsize=3, color="tab:blue")
        ax.scatter(best["hasard"], pos, marker="|", s=300, color="black", label="hasard", zorder=3)
        ax.set_yticks(pos, lab, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Exactitude équilibrée (probabilités moyennées sur les répétitions, IC 95 % bootstrap)")
        ax.set_title("3 meilleurs modèles par tâche", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "synthese_exactitude.png", dpi=150)
        plt.close(fig)
        figures.append("synthese_exactitude.png")

    log("\nMeilleurs modèles par tâche (exactitude équilibrée, IC 95 % bootstrap) :")
    for _, r in best.iterrows():
        log(f"  {r.tache:<22s} {r.harmonisation:<15s} {r.modele:<24s} {r.exactitude_equilibree:.3f} "
            f"[{r.ic95_exactitude_bas:.3f} ; {r.ic95_exactitude_haut:.3f}]  AUC {r.auc:.3f}  Brier {r.brier:.3f}")
    if len(comp):
        n_mieux = int((comp.conclusion == "meilleur").sum())
        log(f"\nComparaisons avec {args.reference} : {n_mieux} / {len(comp)} modèles significativement meilleurs "
            "(IC 95 % de la différence > 0)")
    ecrire_json(out / "synthese.json", {"dossiers": {k: str(v) if v else None for k, v in dossiers.items()},
                                       "bootstrap": args.bootstrap, "figures": figures,
                                       "arguments": arguments_dict(args), "versions": versions()})
    log(f"\nSORTIE : {out}")


if __name__ == "__main__":
    main()
