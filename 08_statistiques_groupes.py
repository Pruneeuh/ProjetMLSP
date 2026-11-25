"""Étape 8 : statistiques de groupes (tests univariés, tailles d'effet, correction FDR) et
comparaison des signatures entre troubles.

Entrée  : dossiers d'extraction (binaire, tous les canaux) et sorties/04_fusion/<date>/ (10 canaux communs)
Sortie  : sorties/08_statistiques/<date>/
    tests_binaires.csv       par base et par variable (canal|bande|mesure, et log_sd) :
                             médianes, Mann-Whitney U, p, q (Benjamini-Hochberg, par base),
                             delta de Cliff, g de Hedges (trouble - témoins)
    resume_significatif.csv  nombre de variables q < --alpha par base, mesure et bande
    effets_<base>.png        g de Hedges, carte canal × bande pour chaque mesure
    similarite_troubles.csv  corrélation de Spearman entre les cartes d'effet des 3 troubles
                             (10 canaux communs, z-score sur les témoins de chaque base)
    similarite_troubles.png
    tests_entre_troubles.csv Kruskal-Wallis entre les 3 groupes de patients sur les z-scores
                             (relatifs aux témoins de LEUR base) : les troubles diffèrent-ils
                             entre eux une fois le site retiré ?
    statistiques.json

Lecture : une signature commune (corrélations positives élevées) va dans le sens d'un facteur
transdiagnostique ; des corrélations faibles expliquent un leave-one-dataset-out au hasard.

Usage :
    python 08_statistiques_groupes.py
    python 08_statistiques_groupes.py --alpha 0.05 --agregation mediane

Dépendances : pip install numpy scipy pandas matplotlib
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr

import eeg_harmonisation as eh
from eeg_config import BANDS, BASES, MEASURES, NOM_TROUBLE
from eeg_donnees import agreger, charger_extraction, charger_fusion, dossier_extraction_defaut, lire_carte_canaux
from eeg_utils import arguments_dict, ecrire_json, entete, fdr_bh, log, nouveau_dossier, plt_agg, versions


def cliff_delta(a, b):
    """P(b > a) - P(b < a) : > 0 si le trouble (b) a des valeurs plus élevées."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not len(a) or not len(b):
        return np.nan
    diff = b[:, None] - a[None, :]
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def hedges_g(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if sp == 0:
        return np.nan
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(j * (b.mean() - a.mean()) / sp)


def tests_base(x, y):
    rows = []
    for col in x.columns:
        v = x[col].to_numpy(float)
        a, b = v[y == 0], v[y == 1]
        a2, b2 = a[np.isfinite(a)], b[np.isfinite(b)]
        try:
            u, pval = mannwhitneyu(b2, a2, alternative="two-sided")
        except ValueError:
            u, pval = np.nan, np.nan
        rows.append(dict(variable=col, mediane_temoins=np.median(a2) if len(a2) else np.nan,
                         mediane_trouble=np.median(b2) if len(b2) else np.nan, u=u, p=pval,
                         delta_cliff=cliff_delta(a, b), g_hedges=hedges_g(a, b)))
    df = pd.DataFrame(rows)
    df["q"] = fdr_bh(df["p"].to_numpy())
    return df


def decomposer(df):
    parts = df["variable"].str.split("|", expand=True)
    if parts.shape[1] == 3:
        is_amp = parts[0].eq("log_sd")
        df["canal"] = np.where(is_amp, parts[1], parts[0])
        df["bande"] = np.where(is_amp, parts[2], parts[1])
        df["mesure"] = np.where(is_amp, "log_sd", parts[2])
    return df


def carte_effets(df, base, canaux, out):
    plt = plt_agg()
    mesures = MEASURES + ["log_sd"]
    fig, axes = plt.subplots(1, len(mesures), figsize=(3.0 * len(mesures), 0.32 * len(canaux) + 2.2), sharey=True)
    vmax = max(0.5, float(np.nanmax(np.abs(df["g_hedges"]))))
    for ax, m in zip(axes, mesures):
        sub = df[df.mesure == m]
        mat = sub.pivot_table(index="canal", columns="bande", values="g_hedges").reindex(index=canaux, columns=BANDS)
        sig = sub.pivot_table(index="canal", columns="bande", values="q").reindex(index=canaux, columns=BANDS)
        im = ax.imshow(mat.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(sig.iat[i, j]) and sig.iat[i, j] < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=10)
        ax.set_xticks(range(len(BANDS)), BANDS, fontsize=8)
        ax.set_yticks(range(len(canaux)), canaux, fontsize=7)
        ax.set_title(m, fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.8, label="g de Hedges (trouble - témoins)")
    fig.suptitle(f"{base} ({NOM_TROUBLE[base]}) : tailles d'effet ; * q < 0,05 (FDR)")
    path = out / f"effets_{base}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for b in BASES:
        p.add_argument(f"--{b}", type=Path, default=None, help=f"Dossier d'extraction {b}.")
    p.add_argument("--fusion", type=Path, default=None)
    p.add_argument("--carte-canaux", type=Path)
    p.add_argument("--agregation", choices=["moyenne", "mediane"], default="moyenne")
    p.add_argument("--alpha", type=float, default=0.05, help="Seuil des q-valeurs (FDR).")
    args = p.parse_args()
    orders = lire_carte_canaux(args.carte_canaux)
    out = nouveau_dossier("statistiques")
    entete("STATISTIQUES DE GROUPES (tests univariés, FDR, tailles d'effet)")
    all_tests, figures, info = [], [], {}
    for base in BASES:
        run = getattr(args, base) or dossier_extraction_defaut(base)
        if run is None:
            log(f"  {base} : aucune extraction, ignorée")
            continue
        ext = charger_extraction(Path(run), base, orders, args.agregation)
        ent, amp = agreger(ext, ext["channels"])
        y = ext["records"]["label"].astype(int).to_numpy()
        df = decomposer(tests_base(ent.join(amp), y))
        df.insert(0, "base", base)
        all_tests.append(df)
        figures.append(carte_effets(df, base, ext["channels"], out))
        n_sig = int((df.q < args.alpha).sum())
        info[base] = dict(dossier=str(run), sujets=len(y), variables=len(df), significatives=n_sig)
        top = df.sort_values("p").head(5)
        log(f"\n[{base}] {len(y)} sujets, {len(df)} variables, {n_sig} avec q < {args.alpha}")
        for _, r in top.iterrows():
            log(f"    {r.variable:<28s} g = {r.g_hedges:+.2f}, δ = {r.delta_cliff:+.2f}, p = {r.p:.2g}, q = {r.q:.2g}")
    if not all_tests:
        p.exit(1, "Aucune base disponible.\n")
    tests = pd.concat(all_tests, ignore_index=True)
    tests.to_csv(out / "tests_binaires.csv", index=False)
    res = (tests.assign(sig=tests.q < args.alpha).groupby(["base", "mesure", "bande"])["sig"].sum()
           .unstack("bande").reindex(columns=BANDS))
    res.to_csv(out / "resume_significatif.csv")
    log("\nNombre de variables significatives (q < alpha) par base, mesure et bande :")
    log(res.to_string())

    # ------------------------------------------------------------------ comparaison des troubles
    try:
        common, ent, amp, meta, _, fdir = charger_fusion(args.fusion)
    except ValueError as exc:
        log(f"\nComparaison des troubles ignorée : {exc}")
        common = None
    if common is not None:
        x = ent.to_numpy(float)
        bases = meta["base"].to_numpy()
        z = eh.harmoniser_tout(x, bases, meta["classe"].eq(0).to_numpy(), "zscore")
        y = meta["label_binaire"].to_numpy()
        effets = {}
        for b in BASES:
            m = bases == b
            effets[b] = np.array([hedges_g(z[m & (y == 0), j], z[m & (y == 1), j]) for j in range(z.shape[1])])
        eff = pd.DataFrame(effets, index=ent.columns)
        eff.to_csv(out / "effets_communs_par_trouble.csv")
        sim = []
        for i, a in enumerate(BASES):
            for b in BASES[i + 1:]:
                ok = np.isfinite(eff[a]) & np.isfinite(eff[b])
                rho, pv = spearmanr(eff.loc[ok, a], eff.loc[ok, b])
                sim.append(dict(trouble_1=a, trouble_2=b, spearman_rho=rho, p_naif=pv, n_variables=int(ok.sum())))
        sim = pd.DataFrame(sim)
        sim.to_csv(out / "similarite_troubles.csv", index=False)
        log("\nSimilarité des cartes d'effet (10 canaux communs ; z-score sur les témoins de chaque base) :")
        log(sim.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        log("  (p naïf : les variables sont corrélées entre elles, il surestime la significativité)")

        plt = plt_agg()
        fig, axes = plt.subplots(1, len(sim), figsize=(4.3 * len(sim), 4), squeeze=False)
        for ax, (_, r) in zip(axes[0], sim.iterrows()):
            ax.scatter(eff[r.trouble_1], eff[r.trouble_2], s=8, alpha=0.6)
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(0, color="gray", lw=0.5)
            ax.set_xlabel(f"g {r.trouble_1}")
            ax.set_ylabel(f"g {r.trouble_2}")
            ax.set_title(f"ρ = {r.spearman_rho:.2f}")
        fig.suptitle("Tailles d'effet (trouble - témoins de la même base), variable par variable")
        fig.tight_layout()
        fig.savefig(out / "similarite_troubles.png", dpi=150)
        plt.close(fig)
        figures.append("similarite_troubles.png")

        pat = meta["classe"].to_numpy() > 0
        kw = []
        for j, col in enumerate(ent.columns):
            groupes = [z[pat & (bases == b), j] for b in BASES]
            groupes = [g[np.isfinite(g)] for g in groupes]
            try:
                h, pv = kruskal(*groupes)
            except ValueError:
                h, pv = np.nan, np.nan
            kw.append(dict(variable=col, h=h, p=pv, **{f"mediane_z_{b}": np.median(g) if len(g) else np.nan
                                                        for b, g in zip(BASES, groupes)}))
        kw = pd.DataFrame(kw)
        kw["q"] = fdr_bh(kw["p"].to_numpy())
        kw.to_csv(out / "tests_entre_troubles.csv", index=False)
        log(f"\nEntre troubles (Kruskal-Wallis sur z-scores relatifs aux témoins de chaque base) : "
            f"{int((kw.q < args.alpha).sum())} / {len(kw)} variables avec q < {args.alpha}")
        info["fusion"] = str(fdir)
    ecrire_json(out / "statistiques.json", {"bases": info, "alpha": args.alpha, "figures": figures,
                                           "arguments": arguments_dict(args), "versions": versions()})
    log(f"\nSORTIE : {out}")


if __name__ == "__main__":
    main()
