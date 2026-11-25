"""Étape 2 : contrôle de qualité des bases brutes et recherche de confusions d'acquisition.

Entrée  : sorties/01_donnees/<base>_signals.npz + <base>_metadata.csv
Sortie  : sorties/02_controle_qualite/<date>/
    qc_enregistrements.csv   un enregistrement par ligne : durée, NaN, canaux plats,
                             amplitude (écart-type médian, maximum absolu), part d'échantillons
                             saturés, bruit secteur (50/60 Hz), fréquence du pic alpha,
                             puissances relatives delta..gamma, drapeaux
    qc_canaux.csv            un canal × enregistrement par ligne (écart-type, plat, saturé)
    qc_resume_bases.csv      effectifs, durées, amplitudes par base et par groupe
    qc_confusions.csv        variables d'acquisition (durée, nombre de canaux d'origine,
                             amplitude, bruit secteur) : AUC témoin/trouble DANS chaque base.
                             Une AUC loin de 0,5 = confusion possible avec l'étiquette.
    qc_alertes.csv           liste des problèmes détectés
    psd_par_base.png, amplitude_par_base.png, puissances_relatives.png, durees.png
    controle_qualite.json

Les seuils sont réglables (--seuil-plat, --seuil-sature, --seuil-secteur).
Ce script ne modifie pas les données : il informe le choix d'exclusions éventuelles.

Usage :
    python 02_controle_qualite.py
    python 02_controle_qualite.py depression --seuil-sature 150

Dépendances : pip install numpy scipy pandas matplotlib scikit-learn
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from eeg_config import BANDPASS, BASES, NOM_TROUBLE
from eeg_donnees import canaux_sauvegardes, charger_signaux
from eeg_utils import arguments_dict, ecrire_json, entete, log, nouveau_dossier, plt_agg, versions

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy 1.x / 2.x
BANDES_PUISSANCE = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def bandpass(sig, fs, low=BANDPASS[0], high=BANDPASS[1]):
    high = min(high, 0.95 * fs / 2)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1)


def qc_enregistrement(sid, sig, fs, canaux, args):
    """Mesures de qualité d'un enregistrement (canaux, échantillons)."""
    n_nan = int(np.sum(~np.isfinite(sig)))
    sig = np.where(np.isfinite(sig), sig, 0.0)
    filt = bandpass(sig, fs)
    sd_brut = sig.std(axis=1)
    sd = filt.std(axis=1)
    plat = sd < args.seuil_plat
    sature = np.mean(np.abs(filt) > args.seuil_sature, axis=1)
    f, pxx = welch(sig - sig.mean(axis=1, keepdims=True), fs=fs, nperseg=min(sig.shape[1], int(4 * fs)), axis=1)
    psd_moy = np.median(pxx, axis=0)
    def puissance(lo, hi, p=psd_moy):
        m = (f >= lo) & (f < hi)
        return float(_trapz(p[m], f[m])) if m.any() else np.nan
    total = puissance(*BANDPASS)
    secteur = {}
    for hz in (50, 60):
        if hz < fs / 2:
            secteur[hz] = puissance(hz - 1, hz + 1) / max(puissance(hz - 5, hz - 2) + puissance(hz + 2, hz + 5), 1e-20)
    occ = [i for i, c in enumerate(canaux) if c in ("O1", "O2")]
    p_occ = np.median(pxx[occ], axis=0) if occ else psd_moy
    m_alpha = (f >= 7) & (f <= 13)
    pic_alpha = float(f[m_alpha][np.argmax(p_occ[m_alpha])]) if m_alpha.any() else np.nan
    row = dict(subject_id=sid, duree_s=sig.shape[1] / fs, n_canaux=sig.shape[0], n_nan=n_nan,
               sd_brut_median=float(np.median(sd_brut)), sd_filtre_median=float(np.median(sd)),
               sd_filtre_min=float(sd.min()), sd_filtre_max=float(sd.max()),
               max_abs_filtre=float(np.abs(filt).max()), canaux_plats=int(plat.sum()),
               part_saturee_max=float(sature.max()), canaux_satures=int(np.sum(sature > args.part_saturee)),
               ratio_secteur_50=secteur.get(50, np.nan), ratio_secteur_60=secteur.get(60, np.nan),
               pic_alpha_hz=pic_alpha, offset_dc_median=float(np.median(np.abs(sig.mean(axis=1)))))
    for nom, (lo, hi) in BANDES_PUISSANCE.items():
        row[f"puissance_rel_{nom}"] = puissance(lo, hi) / total if total > 0 else np.nan
    canal_rows = [dict(subject_id=sid, canal=c, sd_filtre=float(sd[i]), plat=bool(plat[i]),
                       part_saturee=float(sature[i])) for i, c in enumerate(canaux)]
    return row, canal_rows, f, psd_moy


def auc_simple(y, v):
    """AUC de Mann-Whitney (sans modèle) ; NaN si une seule valeur."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(v)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2 or np.unique(v[ok]).size < 2:
        return np.nan
    return roc_auc_score(y[ok], v[ok])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bases", nargs="*", help=f"parmi {BASES} (défaut : toutes)")
    p.add_argument("--donnees", type=Path, default=None, help="Dossier de 01 (défaut sorties/01_donnees).")
    p.add_argument("--seuil-plat", type=float, default=0.5, help="Écart-type (µV) sous lequel un canal est plat.")
    p.add_argument("--seuil-sature", type=float, default=200.0, help="|amplitude| (µV, après filtrage) jugée artefactuelle.")
    p.add_argument("--part-saturee", type=float, default=0.01, help="Part d'échantillons saturés au-delà de laquelle un canal est signalé.")
    p.add_argument("--seuil-secteur", type=float, default=10.0, help="Rapport pic secteur / voisinage signalé.")
    p.add_argument("--seuil-confusion", type=float, default=0.75, help="AUC (ou 1-AUC) d'une variable d'acquisition signalée.")
    args = p.parse_args()
    bases = args.bases or BASES
    out = nouveau_dossier("qc")
    entete("CONTRÔLE DE QUALITÉ DES BASES BRUTES")
    rec_rows, ch_rows, psd, alertes, confusions = [], [], {}, [], []
    for base in bases:
        try:
            signaux, meta = charger_signaux(base, args.donnees)
        except ValueError as exc:
            log(f"  {base} : {exc}")
            continue
        canaux = canaux_sauvegardes(base, args.donnees)
        log(f"[{base}] {len(signaux)} enregistrements, {len(canaux)} canaux")
        psd[base] = {0: [], 1: []}
        rows = []
        for (_, m), sig in zip(meta.iterrows(), signaux):
            row, crow, f, pm = qc_enregistrement(m["subject_id"], sig, float(m["fs"]), canaux, args)
            row.update(base=base, label=int(m["label"]), n_canaux_origine=m.get("n_canaux_origine"))
            rows.append(row)
            ch_rows += [dict(base=base, **c) for c in crow]
            psd[base][int(m["label"])].append((f, pm))
            if row["n_nan"]:
                alertes.append(dict(base=base, subject_id=row["subject_id"], alerte=f"{row['n_nan']} valeurs non finies"))
            if row["canaux_plats"]:
                alertes.append(dict(base=base, subject_id=row["subject_id"], alerte=f"{row['canaux_plats']} canal(aux) plat(s)"))
            if row["canaux_satures"]:
                alertes.append(dict(base=base, subject_id=row["subject_id"],
                                    alerte=f"{row['canaux_satures']} canal(aux) avec > {args.part_saturee:.0%} d'échantillons > {args.seuil_sature:g} µV"))
            for hz in (50, 60):
                v = row[f"ratio_secteur_{hz}"]
                if np.isfinite(v) and v > args.seuil_secteur:
                    alertes.append(dict(base=base, subject_id=row["subject_id"], alerte=f"bruit secteur {hz} Hz (rapport {v:.1f})"))
        df = pd.DataFrame(rows)
        rec_rows.append(df)
        # Confusions d'acquisition : une variable non EEG qui sépare témoins et patients.
        y = df["label"].to_numpy()
        for var in ["duree_s", "n_canaux_origine", "sd_brut_median", "sd_filtre_median", "offset_dc_median",
                    "ratio_secteur_50", "pic_alpha_hz", "max_abs_filtre"]:
            v = pd.to_numeric(df[var], errors="coerce").to_numpy(float)
            auc = auc_simple(y, v)
            confusions.append(dict(base=base, variable=var, auc_trouble=auc,
                                   moyenne_temoins=np.nanmean(v[y == 0]) if (y == 0).any() else np.nan,
                                   moyenne_trouble=np.nanmean(v[y == 1]) if (y == 1).any() else np.nan,
                                   valeurs_distinctes=int(pd.Series(v).nunique())))
            if np.isfinite(auc) and max(auc, 1 - auc) >= args.seuil_confusion:
                alertes.append(dict(base=base, subject_id="(base entière)",
                                    alerte=f"confusion possible : {var} sépare témoins et {NOM_TROUBLE[base]} (AUC {auc:.2f})"))
        log(f"  durée {df['duree_s'].min():.0f}-{df['duree_s'].max():.0f} s ; écart-type filtré médian "
            f"{df['sd_filtre_median'].median():.1f} µV ; pic alpha médian {df['pic_alpha_hz'].median():.1f} Hz")

    if not rec_rows:
        p.exit(1, "Aucune base chargée.\n")
    rec = pd.concat(rec_rows, ignore_index=True)
    rec.to_csv(out / "qc_enregistrements.csv", index=False)
    pd.DataFrame(ch_rows).to_csv(out / "qc_canaux.csv", index=False)
    conf = pd.DataFrame(confusions)
    conf.to_csv(out / "qc_confusions.csv", index=False)
    al = pd.DataFrame(alertes, columns=["base", "subject_id", "alerte"])
    al.to_csv(out / "qc_alertes.csv", index=False)
    num = ["duree_s", "sd_filtre_median", "max_abs_filtre", "pic_alpha_hz", "ratio_secteur_50",
           "puissance_rel_alpha", "canaux_plats", "canaux_satures"]
    resume = rec.groupby(["base", "label"])[num].median()
    resume.insert(0, "n", rec.groupby(["base", "label"]).size())
    resume.reset_index().to_csv(out / "qc_resume_bases.csv", index=False)

    # ------------------------------------------------------------------ figures
    plt = plt_agg()
    figures = []
    fig, axes = plt.subplots(1, len(psd), figsize=(5 * len(psd), 4), sharey=True, squeeze=False)
    for ax, (base, d) in zip(axes[0], psd.items()):
        for lab, nom in [(0, "témoins"), (1, NOM_TROUBLE[base])]:
            if not d[lab]:
                continue
            f = d[lab][0][0]
            mat = np.vstack([pm for ff, pm in d[lab] if len(ff) == len(f)])
            med = np.median(mat, axis=0)
            q1, q3 = np.quantile(mat, [0.25, 0.75], axis=0)
            ax.semilogy(f, med, label=f"{nom} (n={len(mat)})")
            ax.fill_between(f, q1, q3, alpha=0.2)
        ax.set_xlim(0, 50)
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_title(base)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("DSP médiane (µV²/Hz)")
    fig.suptitle("Densité spectrale de puissance (médiane des canaux ; médiane et IQR des sujets)")
    fig.tight_layout()
    fig.savefig(out / "psd_par_base.png", dpi=150)
    plt.close(fig)
    figures.append("psd_par_base.png")

    for var, nom, fichier, logy in [("sd_filtre_median", "Écart-type médian après filtrage (µV)", "amplitude_par_base.png", True),
                                    ("duree_s", "Durée (s)", "durees.png", False)]:
        fig, ax = plt.subplots(figsize=(7, 4))
        data, labels = [], []
        for base in psd:
            for lab in (0, 1):
                v = rec.loc[(rec.base == base) & (rec.label == lab), var].dropna()
                data.append(v.to_numpy())
                labels.append(f"{base}\n{'témoins' if lab == 0 else 'trouble'}")
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(labels) + 1), labels)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(nom)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        fig.savefig(out / fichier, dpi=150)
        plt.close(fig)
        figures.append(fichier)

    cols = [f"puissance_rel_{b}" for b in BANDES_PUISSANCE]
    g = rec.groupby(["base", "label"])[cols].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(g))
    for c in cols:
        ax.bar(range(len(g)), g[c], bottom=bottom, label=c.replace("puissance_rel_", ""))
        bottom += g[c].to_numpy()
    ax.set_xticks(range(len(g)), [f"{b}\n{'témoins' if l == 0 else 'trouble'}" for b, l in g.index], fontsize=8)
    ax.set_ylabel("Puissance relative (médiane)")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(out / "puissances_relatives.png", dpi=150)
    plt.close(fig)
    figures.append("puissances_relatives.png")

    ecrire_json(out / "controle_qualite.json", {
        "portee": "Contrôle descriptif ; aucune donnée n'est modifiée.",
        "bases": list(psd), "n_alertes": len(al), "figures": figures,
        "confusions_signalees": conf.loc[np.maximum(conf.auc_trouble, 1 - conf.auc_trouble) >= args.seuil_confusion]
        .to_dict(orient="records"),
        "arguments": arguments_dict(args), "versions": versions()})

    log("\nRésumé (médianes par base et étiquette) :")
    log(resume.to_string(float_format=lambda v: f"{v:.2f}"))
    log("\nConfusions d'acquisition (AUC témoin/trouble ; 0,5 = aucune) :")
    log(conf.pivot(index="variable", columns="base", values="auc_trouble").to_string(float_format=lambda v: f"{v:.2f}"))
    log(f"\n{len(al)} alerte(s) ; détail dans qc_alertes.csv")
    for _, a in al.loc[al.subject_id.eq("(base entière)")].iterrows():
        log(f"  ! {a.base} : {a.alerte}")
    log(f"SORTIE : {out}")


if __name__ == "__main__":
    main()
