"""Fonctions utilitaires partagées : messages, dossiers, validation croisée par sujet,
métriques, modèles scikit-learn et figures.

Module importé par les scripts numérotés ; il ne s'exécute pas seul.
(Un nom de module Python ne peut pas commencer par un chiffre : c'est pourquoi les
bibliothèques partagées s'appellent eeg_*.py et les scripts exécutables 01_... à 10_...)

    from eeg_utils import log, require, make_folds, multiclass_scores
"""
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from eeg_config import SORTIES, DOSSIERS


# --------------------------------------------------------------------------
# Messages et contrôles
# --------------------------------------------------------------------------
def log(message=""):
    print(message, flush=True)


def require(ok, message):
    """Arrête l'analyse avec un message clair si une condition n'est pas remplie."""
    if not ok:
        raise ValueError(message)


def entete(titre, car="="):
    log(car * 78)
    log(titre)
    log(car * 78)


# --------------------------------------------------------------------------
# Dossiers et fichiers
# --------------------------------------------------------------------------
def horodatage():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def dossier_etape(etape, racine=None):
    """Dossier parent d'une étape (ex. sorties/05_binaire)."""
    return Path(racine or SORTIES) / DOSSIERS[etape]


def nouveau_dossier(etape, prefixe="", racine=None):
    """Crée sorties/<étape>/<préfixe>_<date>/ et le renvoie."""
    nom = f"{prefixe}_{horodatage()}" if prefixe else horodatage()
    out = dossier_etape(etape, racine) / nom
    out.mkdir(parents=True, exist_ok=False)
    return out


def dernier_dossier(etape, prefixe="", contient=None, racine=None):
    """Dossier le plus récent d'une étape (tri par nom = tri par date). None si absent.
    contient : nom d'un fichier qui doit exister dans le dossier."""
    parent = dossier_etape(etape, racine)
    if not parent.is_dir():
        return None
    candidats = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefixe)
                       and (contient is None or (p / contient).exists()))
    return candidats[-1] if candidats else None


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    return obj


def ecrire_json(path, data):
    Path(path).write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")


def arguments_dict(args):
    return {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}


def versions(paquets=("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")):
    out = {}
    for p in paquets:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


# --------------------------------------------------------------------------
# Validation croisée par sujet
# --------------------------------------------------------------------------
def make_folds(strata, groups, n_splits, seed, must_have=None, min_par_strate=3):
    """StratifiedGroupKFold avec contrôles : aucun groupe (sujet) partagé entre
    entraînement et test, chaque sujet testé une fois, et au moins min_par_strate
    sujets de chaque strate de must_have dans chaque pli d'entraînement."""
    from sklearn.model_selection import StratifiedGroupKFold
    strata, groups = np.asarray(strata), np.asarray(groups)
    folds = list(StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                 .split(np.zeros(len(strata)), strata, groups))
    for train, test in folds:
        require(not (set(groups[train]) & set(groups[test])), "Groupe partagé entre entraînement et test.")
        if must_have is not None:
            for s in np.unique(must_have):
                require((must_have[train] == s).sum() >= min_par_strate,
                        f"Pli d'entraînement avec moins de {min_par_strate} sujets de la strate {s} : réduire --plis.")
    covered = np.sort(np.concatenate([t for _, t in folds]))
    require(np.array_equal(covered, np.arange(len(strata))), "Couverture hors pli incorrecte.")
    return folds


# --------------------------------------------------------------------------
# Métriques
# --------------------------------------------------------------------------
def multiclass_scores(y, proba, labels):
    """Exactitude, exactitude équilibrée, F1 macro, AUC (binaire) ou AUC un-contre-tous,
    score de Brier. Renvoie (dict, prédictions)."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y)
    pred = np.asarray(labels)[proba[:, labels].argmax(axis=1)]
    row = dict(
        exactitude=accuracy_score(y, pred),
        exactitude_equilibree=balanced_accuracy_score(y, pred),
        f1_macro=f1_score(y, pred, labels=labels, average="macro", zero_division=0),
    )
    key = "auc" if len(labels) == 2 else "auc_ovr_macro"
    p = proba[:, labels]
    p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
    if len(np.unique(y)) == len(labels):
        row[key] = (roc_auc_score(y, p[:, 1]) if len(labels) == 2 else
                    roc_auc_score(y, p, multi_class="ovr", average="macro", labels=labels))
    else:
        row[key] = np.nan
    onehot = (y[:, None] == np.asarray(labels)[None, :]).astype(float)
    row["brier"] = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return row, pred


def scores_binaires(y, pred):
    from sklearn.metrics import recall_score, precision_score
    return dict(sensibilite=recall_score(y, pred, zero_division=0),
                specificite=recall_score(y, pred, pos_label=0, zero_division=0),
                precision_trouble=precision_score(y, pred, zero_division=0))


def summarise(df, keys):
    """Moyenne et écart-type (suffixe _std) de chaque métrique numérique par clé."""
    num = [c for c in df.columns if c not in keys + ["repetition"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(keys, sort=False)[num]
    return g.mean().join(g.std().add_suffix("_std")).reset_index()


def fmt_table(df, cols):
    cols = [c for c in cols if c in df.columns]
    return df[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}")


def p_valeur_permutation(observe, nul):
    nul = np.asarray(nul)
    return (1 + np.sum(nul >= observe)) / (1 + len(nul))


def fdr_bh(p):
    """Correction de Benjamini-Hochberg (q-valeurs). Les NaN restent NaN."""
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def bootstrap_ic(fonction, n, groupes=None, n_boot=2000, seed=0, alpha=0.05):
    """Intervalle de confiance bootstrap percentile ; rééchantillonnage par groupe (sujet).
    fonction(indices) -> valeur."""
    rng = np.random.default_rng(seed)
    if groupes is None:
        groupes = np.arange(n)
    groupes = np.asarray(groupes)
    uniques = np.unique(groupes)
    index = {g: np.flatnonzero(groupes == g) for g in uniques}
    vals = []
    for _ in range(n_boot):
        tirage = rng.choice(uniques, size=len(uniques), replace=True)
        idx = np.concatenate([index[g] for g in tirage])
        try:
            v = fonction(idx)
        except ValueError:
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# --------------------------------------------------------------------------
# Modèles scikit-learn (mêmes réglages dans toutes les analyses)
# --------------------------------------------------------------------------
MODELES_SKLEARN = ["reference_majoritaire", "regression_logistique", "mlp", "svm_rbf",
                   "foret_aleatoire", "lda", "xgboost"]


class Compteur:
    non_converges = 0


def make_estimator(nom, seed, c=1.0, mlp_couches=(16,), mlp_alpha=1.0):
    from sklearn.dummy import DummyClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pre = lambda: [SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()]
    if nom == "reference_majoritaire":
        return DummyClassifier(strategy="prior")
    if nom in ("regression_logistique", "reference_amplitude", "reference_canaux_origine"):
        # lbfgs + plusieurs classes = multinomiale ; classes pondérées (témoins plus nombreux).
        return make_pipeline(*pre(), LogisticRegression(C=c, class_weight="balanced", solver="lbfgs",
                                                        max_iter=5000))
    if nom == "mlp":
        # Petit réseau fortement régularisé ; lbfgs convient aux petits jeux, pas d'arrêt précoce.
        return make_pipeline(*pre(), MLPClassifier(hidden_layer_sizes=tuple(mlp_couches), activation="relu",
                                                   solver="lbfgs", alpha=mlp_alpha, max_iter=5000,
                                                   random_state=seed))
    if nom == "svm_rbf":
        from sklearn.svm import SVC
        return make_pipeline(*pre(), SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                                         probability=True, random_state=seed))
    if nom == "foret_aleatoire":
        from sklearn.ensemble import RandomForestClassifier
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=300, max_features="sqrt", min_samples_leaf=2,
                                                    class_weight="balanced", random_state=seed, n_jobs=1))
    if nom == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        # Covariance rétrécie (Ledoit-Wolf) : indispensable quand variables > sujets.
        return make_pipeline(*pre(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
    if nom == "xgboost":
        from xgboost import XGBClassifier
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
            min_child_weight=2, reg_lambda=1.0, random_state=seed, n_jobs=1, verbosity=0))
    raise ValueError(f"Modèle inconnu : {nom}")


def fit_predict(nom, seed, x_tr, y_tr, x_te, n_classes, c=1.0, mlp_couches=(16,), mlp_alpha=1.0,
                estimateur=None):
    """Entraîne et renvoie les probabilités (colonnes 0..n_classes-1)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.utils.class_weight import compute_sample_weight
    model = estimateur if estimateur is not None else make_estimator(nom, seed, c, mlp_couches, mlp_alpha)
    fit_kw = {}
    if nom == "xgboost":
        fit_kw["xgbclassifier__sample_weight"] = compute_sample_weight("balanced", y_tr)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_tr, y_tr, **fit_kw)
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        Compteur.non_converges += 1
    proba = np.zeros((len(x_te), n_classes))
    proba[:, np.asarray(model.classes_).astype(int)] = model.predict_proba(x_te)
    return proba, model


def xgboost_disponible():
    try:
        import xgboost  # noqa: F401
        return True
    except ModuleNotFoundError:
        return False


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def plt_agg():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_confusion(mat, labels_lignes, labels_colonnes, titre, path, ax=None):
    """Matrice de confusion normalisée par ligne, effectifs entre parenthèses."""
    plt = plt_agg()
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(1.3 * len(labels_colonnes) + 2.5, 1.0 * len(labels_lignes) + 2))
    norm = mat / np.clip(mat.sum(axis=1, keepdims=True), 1, None)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}\n({mat[i, j]})", ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels_colonnes)), labels_colonnes, rotation=30, ha="right")
    ax.set_yticks(range(len(labels_lignes)), labels_lignes)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    ax.set_title(titre, fontsize=10)
    if own:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return Path(path).name if path else None


def plot_roc(courbes, titre, path):
    """courbes : liste de (nom, y, score)."""
    from sklearn.metrics import roc_auc_score, roc_curve
    plt = plt_agg()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for nom, y, s in courbes:
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{nom} (AUC {roc_auc_score(y, s):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="hasard")
    ax.set_xlabel("Taux de faux positifs (1 - spécificité)")
    ax.set_ylabel("Taux de vrais positifs (sensibilité)")
    ax.set_title(titre, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path).name


def plot_barres(df, x, y, err, hue, titre, path, hasard=None, ylabel=None):
    """Barres groupées (moyenne ± écart-type)."""
    plt = plt_agg()
    xs = list(dict.fromkeys(df[x]))
    hs = list(dict.fromkeys(df[hue]))
    width = 0.8 / max(1, len(hs))
    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(xs) * max(1, len(hs)) / 2), 4.8))
    for i, h in enumerate(hs):
        s = df[df[hue] == h].set_index(x).reindex(xs)
        ax.bar(np.arange(len(xs)) + i * width - 0.4 + width / 2, s[y], width,
               yerr=s[err] if err in s else None, label=h, capsize=3)
    if hasard is not None:
        ax.axhline(hasard, color="black", linestyle="--", linewidth=1, label="hasard")
    ax.set_xticks(range(len(xs)), xs, rotation=25, ha="right")
    ax.set_ylabel(ylabel or y)
    ax.set_ylim(0, 1.05)
    ax.set_title(titre, fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path).name
