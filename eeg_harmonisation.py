"""Harmonisation inter-bases (effet de site), module partagé, non exécutable.

Toutes les méthodes sont ajustées sur des sujets de RÉFÉRENCE (les témoins
d'entraînement du pli) : appelées dans chaque pli, elles ne fuient aucune étiquette
du test.

  - zscore         : chaque variable en z-score par rapport aux témoins de sa base
                     (principe de ComBat sans lissage bayésien) ;
  - filtrage       : z-score, puis retrait des variables qui prédisent le mieux la base
                     chez les témoins (importance d'une forêt d'arbres extrêmes) ;
  - coral          : z-score, ACP sur les témoins réunis, puis alignement des covariances :
                     chaque base est blanchie par la covariance Ledoit-Wolf de ses témoins
                     et recolorée par celle des témoins réunis ;
  - filtrage_coral : filtrage puis CORAL.
  - aucune         : données brutes (référence).

Réseaux profonds : forme « masque » du filtrage (variables de site mises à 0 au lieu
d'être retirées) pour garder la structure électrodes × bandes × mesures ; CORAL
n'est pas utilisable (l'ACP détruit cette structure).
"""
import warnings

import numpy as np

from eeg_utils import require

HARMONISATIONS = ["zscore", "filtrage", "coral", "filtrage_coral"]


class Reglages:
    """Paramètres par défaut (modifiables par les scripts via configurer())."""
    filtrage_fraction = 0.5
    filtrage_arbres = 200
    coral_composantes = 20


def configurer(args):
    for nom in ["filtrage_fraction", "filtrage_arbres", "coral_composantes"]:
        if hasattr(args, nom):
            setattr(Reglages, nom, getattr(args, nom))


def _zstats(ref, base):
    require(len(ref) >= 2, f"Harmonisation : moins de 2 sujets de référence pour {base}.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(ref, axis=0)
        sd = np.nanstd(ref, axis=0, ddof=1)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    return mu, sd


def _sym_power(mat, power):
    vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
    vals = np.clip(vals, 1e-10, None)
    return (vecs * vals ** power) @ vecs.T


def site_filter(refs, fraction=None, arbres=None):
    """Indices des variables gardées : on retire celles qui prédisent le mieux la base chez
    les sujets de référence (après z-score). Forêt d'arbres extrêmes : capte aussi les
    différences non linéaires restantes."""
    from sklearn.ensemble import ExtraTreesClassifier
    fraction = Reglages.filtrage_fraction if fraction is None else fraction
    arbres = Reglages.filtrage_arbres if arbres is None else arbres
    bases = sorted(refs)
    x = np.vstack([refs[b] for b in bases])
    y = np.concatenate([[i] * len(refs[b]) for i, b in enumerate(bases)])
    x = np.where(np.isfinite(x), x, 0.0)
    forest = ExtraTreesClassifier(n_estimators=arbres, max_features="sqrt", class_weight="balanced",
                                  random_state=0, n_jobs=1)
    imp = forest.fit(x, y).feature_importances_
    n_keep = max(2, int(round(x.shape[1] * (1 - fraction))))
    return np.sort(np.argsort(imp, kind="stable")[:n_keep])


def coral_align(z_tr, z_te, base_tr, base_te, refs, composantes=None):
    """CORAL sur les témoins (voir l'en-tête). Renvoie les scores alignés (k colonnes)."""
    from sklearn.covariance import LedoitWolf
    from sklearn.decomposition import PCA
    composantes = Reglages.coral_composantes if composantes is None else composantes
    pooled = np.vstack(list(refs.values()))
    clean = lambda a: np.where(np.isfinite(a), a, 0.0)   # 0 = moyenne des témoins après z-score
    k = int(min(composantes, pooled.shape[1], len(pooled) - 1))
    pca = PCA(n_components=k, random_state=0).fit(clean(pooled))
    s_tr, s_te = pca.transform(clean(z_tr)), pca.transform(clean(z_te))
    s_ref = {b: pca.transform(clean(r)) for b, r in refs.items()}
    s_pool = np.vstack(list(s_ref.values()))
    target_half = _sym_power(LedoitWolf().fit(s_pool).covariance_, 0.5)
    target_mu = s_pool.mean(axis=0)
    out_tr, out_te = np.empty_like(s_tr), np.empty_like(s_te)
    for b, r in s_ref.items():
        whiten = _sym_power(LedoitWolf().fit(r).covariance_, -0.5)
        a = whiten @ target_half
        mu = r.mean(axis=0)
        out_tr[base_tr == b] = (s_tr[base_tr == b] - mu) @ a + target_mu
        out_te[base_te == b] = (s_te[base_te == b] - mu) @ a + target_mu
    return out_tr, out_te


def harmonise(x_tr, x_te, base_tr, base_te, ref_tr, method="zscore", ref_te_stats=None, masque=False):
    """Harmonisation par base, ajustée sur les lignes de référence de l'entraînement.

    ref_tr : booléen (lignes d'entraînement servant de référence, en général les témoins).
    ref_te_stats : dict base -> matrice brute de référence pour une base absente de
                   l'entraînement (leave-one-dataset-out).
    masque : filtrage par mise à 0 des variables de site (forme conservée, réseaux profonds).
    """
    if method == "aucune":
        return x_tr, x_te
    require(method in HARMONISATIONS, f"Harmonisation inconnue : {method}")
    require(not (masque and "coral" in method), "CORAL n'est pas compatible avec la forme des réseaux profonds.")
    z_tr = np.full_like(x_tr, np.nan, dtype=float)
    z_te = np.full_like(x_te, np.nan, dtype=float)
    refs = {}
    for b in np.unique(np.concatenate([base_tr, base_te])):
        if ref_te_stats is not None and b in ref_te_stats:
            raw_ref = ref_te_stats[b]
        else:
            require(b in set(base_tr), f"Base {b} absente de l'entraînement.")
            raw_ref = x_tr[(base_tr == b) & ref_tr]
        mu, sd = _zstats(raw_ref, b)
        z_tr[base_tr == b] = (x_tr[base_tr == b] - mu) / sd
        z_te[base_te == b] = (x_te[base_te == b] - mu) / sd
        refs[b] = (raw_ref - mu) / sd
    if method == "zscore":
        return z_tr, z_te
    if "filtrage" in method:
        keep = site_filter(refs)
        if masque:
            drop = np.setdiff1d(np.arange(x_tr.shape[1]), keep)
            z_tr[:, drop] = 0.0
            z_te[:, drop] = 0.0
        else:
            z_tr, z_te = z_tr[:, keep], z_te[:, keep]
            refs = {b: r[:, keep] for b, r in refs.items()}
    if "coral" in method:
        for b, r in refs.items():
            require(len(r) >= 3, f"CORAL : moins de 3 témoins de référence pour {b}.")
        z_tr, z_te = coral_align(z_tr, z_te, base_tr, base_te, refs)
    return z_tr, z_te


def prepare_folds(x, bases, ref_mask, folds, method, masque=False):
    """Données de chaque pli après harmonisation (calculées une fois, réutilisées par les modèles)."""
    return [harmonise(x[tr], x[te], bases[tr], bases[te], ref_mask[tr], method, masque=masque)
            for tr, te in folds]


def harmoniser_tout(x, bases, ref_mask, method):
    """Harmonisation ajustée sur TOUS les témoins (descriptif seulement : figures, statistiques).
    Ne pas utiliser pour estimer une performance (fuite)."""
    z, _ = harmonise(x, x[:1], bases, bases[:1], ref_mask, method)
    return z


def eta_carre(x, groupes):
    """Part de variance expliquée par le groupe (base), variable par variable (ANOVA à un facteur)."""
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape[1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=0)
        tot = np.nansum((x - mu) ** 2, axis=0)
        inter = np.zeros(x.shape[1])
        for g in np.unique(groupes):
            xs = x[groupes == g]
            n = np.sum(np.isfinite(xs), axis=0)
            inter += n * (np.nanmean(xs, axis=0) - mu) ** 2
        ok = tot > 0
        out[ok] = inter[ok] / tot[ok]
    return out
