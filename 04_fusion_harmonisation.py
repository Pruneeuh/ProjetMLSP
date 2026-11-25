"""Étape 4 : fusion des 3 bases sur les électrodes communes et diagnostic de l'harmonisation.

Entrée  : les 3 dossiers d'extraction (sorties/03_extraction/<base>_<date>/, le plus récent par
          défaut) ou d'anciens dossiers extraction_corrigee.py (--depression, --schizophrenie, --anxiete).
Sortie  : sorties/04_fusion/<date>/
    entropies_sujets.csv     une ligne par sujet, colonnes « canal|bande|mesure »
                             (10 canaux × 5 composantes × 5 mesures = 250 variables)
    amplitude_sujets.csv     log de l'écart-type avant standardisation, « log_sd|canal|bande »
    meta_sujets.csv          base, label_binaire, classe (0 témoin, 1 dépression, 2 schizophrénie,
                             3 anxiété), groupe de validation, nombre de fenêtres
    fusion.json              canaux communs, correspondances, effectifs, dossiers source
    harmonise_<méthode>.csv  (descriptif) données harmonisées sur TOUS les témoins
    effet_site_variables.csv part de variance due à la base (η²) chez les témoins, par variable,
                             avant et après chaque harmonisation
    effet_site_resume.csv    η² moyen par méthode et par mesure
    acp_harmonisations.png   ACP 2D colorée par base, avant / après chaque méthode
    effet_site_carte.png     η² chez les témoins (canal × composante) avant / après z-score

Étapes :
  1. noms d'électrodes ramenés au 10-10 (T3→T7, T4→T8, T5→P7, T6→P8) ; colonnes numérotées
     col01... converties avec l'ordre SUPPOSÉ (Borisov, Emotiv) ou --carte-canaux ;
  2. électrodes communes aux 3 bases (F7, F3, F4, F8, T7, T8, P7, P8, O1, O2 attendues) ;
  3. agrégation des fenêtres par sujet (moyenne ou médiane) ;
  4. fusion en 4 classes ; groupes de validation préfixés par la base.

IMPORTANT : les fichiers harmonise_*.csv et les figures sont DESCRIPTIFS (harmonisation ajustée
sur tous les témoins). Les analyses 06 et 07 ré-ajustent l'harmonisation dans chaque pli,
sur les seuls témoins d'entraînement.

Usage :
    python 04_fusion_harmonisation.py
    python 04_fusion_harmonisation.py --agregation mediane --carte-canaux canaux.json
    python 04_fusion_harmonisation.py --depression 04_extraction_corrigee/depression_... \\
        --schizophrenie 04_extraction_corrigee/schizophrenie_... --anxiete 04_extraction_corrigee/anxiete_...

Dépendances : pip install numpy pandas scikit-learn matplotlib
"""
import argparse

import numpy as np
import pandas as pd

import eeg_harmonisation as eh
from eeg_config import BANDS, BASES, CANAUX, CLASSES, MEASURES, ORDRE_SUPPOSE
from eeg_donnees import (ajouter_arguments_extraction, construire_fusion, decomposer_colonnes, dossiers_extraction,
                         lire_carte_canaux)
from eeg_utils import arguments_dict, ecrire_json, entete, log, nouveau_dossier, plt_agg, versions


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ajouter_arguments_extraction(p)
    p.add_argument("--harmonisations", nargs="+", choices=eh.HARMONISATIONS, default=eh.HARMONISATIONS)
    p.add_argument("--coral-composantes", type=int, default=20)
    p.add_argument("--filtrage-fraction", type=float, default=0.5)
    p.add_argument("--filtrage-arbres", type=int, default=200)
    p.add_argument("--sans-diagnostic", action="store_true", help="Seulement la fusion (pas d'harmonisation descriptive).")
    args = p.parse_args()
    eh.configurer(args)
    try:
        runs = dossiers_extraction(args)
        infos, common, ent, amp, meta = construire_fusion(runs, lire_carte_canaux(args.carte_canaux), args.agregation)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Fusion arrêtée : {exc}\n")
    out = nouveau_dossier("fusion")

    entete("FUSION DES BASES SUR LES CANAUX COMMUNS")
    for b in BASES:
        log(f"  {b:<14s} {runs[b]}")
    log(f"Canaux communs ({len(common)}) : {', '.join(common)}")
    for b in BASES:
        rec = infos[b]["records"]
        log(f"  {b:<14s} {len(rec):3d} enregistrements ({int((rec.label == 0).sum())} témoins, "
            f"{int((rec.label == 1).sum())} trouble), {len(infos[b]['channels'])} canaux d'origine, "
            f"{meta.loc[meta.base == b, 'groupe'].nunique()} groupes")
        if ORDRE_SUPPOSE[b] and not args.carte_canaux:
            log(f"    ATTENTION : ordre des électrodes supposé ({', '.join(CANAUX[b])}).")
    counts = meta["classe"].value_counts().sort_index()
    log("Classes : " + ", ".join(f"{i}={CLASSES[i]} ({counts.get(i, 0)})" for i in range(len(CLASSES))))
    log(f"Variables : {ent.shape[1]} entropies, {amp.shape[1]} amplitudes ; agrégation {args.agregation} ; "
        f"manquantes {int(ent.isna().sum().sum())} / {int(amp.isna().sum().sum())}")

    ent.to_csv(out / "entropies_sujets.csv")
    amp.to_csv(out / "amplitude_sujets.csv")
    meta.to_csv(out / "meta_sujets.csv")
    info = {
        "portee": "Fusion des 3 bases ; ordre des électrodes schizophrénie/anxiété supposé sauf --carte-canaux.",
        "dossiers_extraction": {b: str(runs[b]) for b in BASES},
        "canaux_communs": common,
        "canaux_par_base": {b: infos[b]["channels"] for b in BASES},
        "correspondance_canaux": {b: {str(k): v for k, v in infos[b]["mapping"].items()} for b in BASES},
        "carte_canaux": str(args.carte_canaux) if args.carte_canaux else None,
        "agregation": args.agregation,
        "effectifs": {b: {"temoins": int(((meta.base == b) & (meta.classe == 0)).sum()),
                          "trouble": int(((meta.base == b) & (meta.classe > 0)).sum()),
                          "groupes": int(meta.loc[meta.base == b, "groupe"].nunique())} for b in BASES},
        "groupes_a_plusieurs_enregistrements": {g: int(n) for g, n in meta["groupe"].value_counts().items() if n > 1},
        "classes": CLASSES, "bandes": BANDS, "mesures": MEASURES,
        "arguments": arguments_dict(args), "versions": versions(),
    }
    ecrire_json(out / "fusion.json", info)
    if args.sans_diagnostic:
        log(f"SORTIE : {out}")
        return

    # ------------------------------------------------------------------ diagnostic descriptif
    log("\nDiagnostic de l'effet de site (descriptif, harmonisation sur tous les témoins)")
    x = ent.to_numpy(float)
    bases = meta["base"].to_numpy()
    ref = meta["classe"].eq(0).to_numpy()
    versions_x = {"aucune": x}
    for h in args.harmonisations:
        try:
            versions_x[h] = eh.harmoniser_tout(x, bases, ref, h)
        except ValueError as exc:
            log(f"  {h} : impossible ({exc})")
            continue
        cols = ent.columns if versions_x[h].shape[1] == x.shape[1] else [f"comp{i + 1:02d}" for i in range(versions_x[h].shape[1])]
        if h == "filtrage":
            refs = {b: eh.harmoniser_tout(x, bases, ref, "zscore")[(bases == b) & ref] for b in BASES}
            cols = ent.columns[eh.site_filter(refs)]
        pd.DataFrame(versions_x[h], index=ent.index, columns=cols).join(meta).to_csv(out / f"harmonise_{h}.csv")

    rows, var_rows = [], []
    desc = decomposer_colonnes(ent.columns)
    for h, xv in versions_x.items():
        eta = eh.eta_carre(xv[ref], bases[ref])
        eta_all = eh.eta_carre(xv, bases)
        # Hétérogénéité des dispersions (type Levene) : η² sur l'écart absolu à la médiane de la base.
        xr, br = xv[ref], bases[ref]
        dev = np.vstack([np.abs(xr[br == b] - np.nanmedian(xr[br == b], axis=0)) for b in BASES])
        eta_disp = eh.eta_carre(dev, np.concatenate([br[br == b] for b in BASES]))
        rows.append(dict(harmonisation=h, n_variables=xv.shape[1], eta2_temoins_moyen=np.nanmean(eta),
                         eta2_dispersion_temoins_moyen=np.nanmean(eta_disp),
                         eta2_temoins_max=np.nanmax(eta), eta2_tous_moyen=np.nanmean(eta_all)))
        if xv.shape[1] == x.shape[1]:
            for c, e in zip(ent.columns, eta):
                var_rows.append(dict(harmonisation=h, variable=c, **desc.loc[c].to_dict(), eta2_temoins=e))
        if xv.shape[1] == x.shape[1]:
            for c, e in zip(ent.columns, eta_disp):
                var_rows[-len(ent.columns) + list(ent.columns).index(c)]["eta2_dispersion_temoins"] = e
    res = pd.DataFrame(rows)
    res.to_csv(out / "effet_site_resume.csv", index=False)
    var = pd.DataFrame(var_rows)
    if len(var):
        var.to_csv(out / "effet_site_variables.csv", index=False)
        log("η² de la base chez les témoins, moyenne par mesure (0 = aucun effet de site) :")
        log(var.pivot_table(index="mesure", columns="harmonisation", values="eta2_temoins")
            .to_string(float_format=lambda v: f"{v:.3f}"))
    log("\nRésumé par méthode :")
    log(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    log("  Note : après z-score, η² des moyennes vaut 0 par construction chez les témoins ; la colonne\n"
        "  « dispersion » mesure ce qui reste (écarts de variance), et 06_classification_multiclasse.py teste\n"
        "  l'effet résiduel complet (prédire la base des témoins).")

    # Figures
    from sklearn.decomposition import PCA
    plt = plt_agg()
    n = len(versions_x)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), squeeze=False)
    couleurs = {"depression": "tab:blue", "schizophrenie": "tab:orange", "anxiete": "tab:green"}
    for ax, (h, xv) in zip(axes[0], versions_x.items()):
        xc = np.where(np.isfinite(xv), xv, np.nanmedian(xv, axis=0))
        xc = (xc - xc.mean(0)) / np.where(xc.std(0) > 0, xc.std(0), 1)
        z = PCA(n_components=2, random_state=0).fit_transform(xc)
        for b in BASES:
            for lab, mk in [(0, "o"), (1, "x")]:
                m = (bases == b) & (meta["label_binaire"].to_numpy() == lab)
                ax.scatter(z[m, 0], z[m, 1], s=14, marker=mk, c=couleurs[b], alpha=0.7,
                           label=f"{b} {'témoins' if lab == 0 else 'trouble'}")
        ax.set_title(h)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(fontsize=7, loc="best")
    fig.suptitle("ACP des entropies par sujet (o = témoins, x = trouble) : les bases se superposent-elles ?")
    fig.tight_layout()
    fig.savefig(out / "acp_harmonisations.png", dpi=150)
    plt.close(fig)

    if len(var):
        sub = var[var.harmonisation.isin(["aucune", "zscore"])]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
        for ax, h in zip(axes, ["aucune", "zscore"]):
            mat = sub[sub.harmonisation == h].pivot_table(index="canal", columns="bande", values="eta2_temoins")
            mat = mat.reindex(index=common, columns=BANDS)
            im = ax.imshow(mat.to_numpy(), cmap="magma", vmin=0, vmax=max(0.05, float(np.nanmax(var["eta2_temoins"]))))
            ax.set_xticks(range(len(BANDS)), BANDS)
            ax.set_yticks(range(len(common)), common)
            ax.set_title(f"η² base (témoins), {h}")
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(out / "effet_site_carte.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log(f"\nSORTIE : {out}")


if __name__ == "__main__":
    main()
