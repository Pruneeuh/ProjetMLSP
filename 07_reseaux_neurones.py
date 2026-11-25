"""Étape 7 : réseaux de neurones profonds (PyTorch, CPU) sur les entropies, binaire et 4 classes.

Entrée  : binaire  → dossiers d'extraction (tous les canaux de chaque base) ;
          4 classes → sorties/04_fusion/<date>/ (10 canaux communs).
Sortie  : sorties/07_reseaux_neurones/<date>/
    resultats_par_repetition.csv, resume.csv, predictions_hors_pli.csv,
    attention_electrodes.csv (poids d'attention par électrode, cnn_attention),
    courbes_apprentissage.csv (perte d'entraînement par époque, répétition 1, pli 1),
    permutations.csv (option), parametres.json, comparaison_modeles.png

Données : pour chaque sujet, un tenseur électrodes × composantes DWT × entropies (C × 5 × 5).
C = 10 canaux communs en 4 classes ; en binaire tous les canaux de la base (19 / 16 / 14), ou
les 10 communs avec --binaire-canaux-communs.

Modèles (--modeles) :
  Références scikit-learn : reference_majoritaire, regression_logistique, mlp_sklearn
  Réseaux profonds :
    cnn1d           CNN 1D le long des composantes (A4 → D1), poids partagés entre électrodes
    cnn_attention   même encodeur + auto-attention entre électrodes + regroupement par attention
    eegnet_adapte   EEGNet (Lawhern et al. 2018) adapté : convolution fréquentielle, convolution en
                    profondeur sur toutes les électrodes, convolution séparable, norme max
    ft_transformer  FT-Transformer (Gorishniy et al. 2021) : chaque variable devient un jeton
    tabm            TabM (Gorishniy et al. 2025) : MLP en ensemble efficace (BatchEnsemble)
  Aucun modèle ne dépend de l'ordre des électrodes (utile car l'ordre est supposé pour 2 bases).

Prétraitement dans chaque pli : harmonisation (4 classes : aucune, zscore par défaut, filtrage
= variables de site mises à 0), imputation médiane, standardisation, écrêtage à ±6.
CORAL n'est pas proposé : l'ACP détruit la structure électrodes × bandes × mesures.
Entraînement : AdamW, montée linéaire puis cosinus, nombre d'époques FIXE (pas d'arrêt précoce,
trop peu de sujets pour une validation interne), entropie croisée pondérée avec lissage, bruit
gaussien d'augmentation, abandon.

Robustesse (Mac, conda « base ») : PyTorch importé AVANT numpy/scikit-learn, OpenMP limité à 1 fil,
chaque réseau d'abord testé dans un sous-processus (--sans-autotest pour l'éviter), et une erreur
dans un modèle n'arrête pas les autres. faulthandler affiche la ligne fautive en cas de plantage.

Usage :
    python 07_reseaux_neurones.py                                   # binaire + 4 classes
    python 07_reseaux_neurones.py --mode multiclasse --harmonisations zscore filtrage
    python 07_reseaux_neurones.py --mode binaire --bases-binaires schizophrenie
    python 07_reseaux_neurones.py --modeles regression_logistique cnn_attention tabm --repetitions 5
    python 07_reseaux_neurones.py --mode multiclasse --permutations 50 --permutation-modeles cnn_attention

Dépendances : pip install torch numpy pandas scikit-learn matplotlib
"""
import faulthandler
import os
import sys

# macOS + conda : PyTorch (libomp) et numpy/scikit-learn (MKL) chargent chacun leur OpenMP ;
# ensemble, ils provoquent souvent une « segmentation fault ». On tolère le doublon, on limite
# OpenMP à 1 fil et on importe PyTorch AVANT numpy/scikit-learn.
faulthandler.enable()
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch est requis : pip install torch") from exc

import argparse
import subprocess
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import eeg_harmonisation as eh
from eeg_config import BANDS, BASES, CLASSES, GRAINE, MEASURES
from eeg_donnees import agreger, charger_extraction, charger_fusion, dossier_extraction_defaut, lire_carte_canaux
from eeg_utils import (arguments_dict, ecrire_json, fit_predict as sk_fit_predict, fmt_table, log, make_folds,
                       multiclass_scores, nouveau_dossier, p_valeur_permutation, plot_barres, require, summarise,
                       versions)
from eeg_config import CLASS_OF_BASE

REFERENCES = ["reference_majoritaire", "regression_logistique", "mlp_sklearn"]
DEEP = ["cnn1d", "cnn_attention", "eegnet_adapte", "ft_transformer", "tabm"]
ALL_MODELS = REFERENCES + DEEP
SK_NAME = {"reference_majoritaire": "reference_majoritaire", "regression_logistique": "regression_logistique",
           "mlp_sklearn": "mlp"}
HARMONISATIONS = ["aucune", "zscore", "filtrage"]


# =========================================================================== architectures
# Toutes reçoivent x de forme (N, C·B·M), ordre canal > composante > entropie.

class ElectrodeEncoder(nn.Module):
    """CNN 1D le long des composantes (A4 → D1), partagé entre électrodes. (N, C, B, M) → (N, C, d)."""

    def __init__(self, n_b, n_m, d, dropout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_m, d, kernel_size=3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
            nn.Conv1d(d, d, kernel_size=3, padding=1), nn.BatchNorm1d(d), nn.GELU(),
            nn.Dropout(dropout))
        self.proj = nn.Linear(d * n_b, d)

    def forward(self, x):
        n, c, b, m = x.shape
        z = x.reshape(n * c, b, m).transpose(1, 2)          # (N·C, M, B)
        z = self.conv(z).flatten(1)                          # (N·C, d·B)
        return F.gelu(self.proj(z)).reshape(n, c, -1)       # (N, C, d)


class CNN1D(nn.Module):
    def __init__(self, shape, n_out, d=16, dropout=0.3):
        super().__init__()
        self.shape = shape
        c, b, m = shape
        self.enc = ElectrodeEncoder(b, m, d, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c * d, n_out))

    def forward(self, x):
        return self.head(self.enc(x.reshape(-1, *self.shape)).flatten(1))


class CNNAttention(nn.Module):
    """Encodeur CNN 1D par électrode + auto-attention entre électrodes + regroupement par attention."""

    def __init__(self, shape, n_out, d=16, heads=4, layers=1, dropout=0.3):
        super().__init__()
        self.shape = shape
        c, b, m = shape
        self.enc = ElectrodeEncoder(b, m, d, dropout)
        self.pos = nn.Parameter(torch.zeros(1, c, d))       # identité de l'électrode
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.attn = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pool = nn.Linear(d, 1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, n_out))
        self.last_weights = None

    def forward(self, x):
        h = self.attn(self.enc(x.reshape(-1, *self.shape)) + self.pos)   # (N, C, d)
        w = torch.softmax(self.pool(h).squeeze(-1), dim=1)                 # (N, C)
        self.last_weights = w.detach()
        return self.head((w.unsqueeze(-1) * h).sum(1))


class EEGNetAdapte(nn.Module):
    """EEGNet adapté : entrée (N, M, C, B), les entropies jouent le rôle de cartes d'entrée."""

    def __init__(self, shape, n_out, f1=8, depth=2, f2=16, dropout=0.3, max_norm=0.25):
        super().__init__()
        self.shape = shape
        c, b, m = shape
        self.block1 = nn.Sequential(
            nn.Conv2d(m, f1, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (c, 1), groups=f1, bias=False), nn.BatchNorm2d(f1 * depth),
            nn.ELU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * depth, f1 * depth, (1, 3), padding=(0, 1), groups=f1 * depth, bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Dropout(dropout))
        self.dense = nn.Linear(f2, n_out)
        self.max_norm = max_norm

    def forward(self, x):
        z = x.reshape(-1, *self.shape).permute(0, 3, 1, 2)   # (N, M, C, B)
        return self.dense(self.block2(self.block1(z)).flatten(1))

    def apply_constraints(self):
        with torch.no_grad():
            w = self.dense.weight
            w.copy_(torch.renorm(w, p=2, dim=0, maxnorm=self.max_norm))


class FTTransformer(nn.Module):
    """FT-Transformer : jeton_j = x_j · W_j + b_j, jeton [CLS], encodeur Transformer pré-normalisé."""

    def __init__(self, n_features, n_out, d=16, heads=4, layers=1, dropout=0.2):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n_features, d))
        self.b = nn.Parameter(torch.empty(n_features, d))
        bound = 1 / d ** 0.5
        nn.init.uniform_(self.w, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)
        self.cls = nn.Parameter(torch.empty(1, 1, d))
        nn.init.uniform_(self.cls, -bound, bound)
        # Pas d'abandon DANS l'attention (chemin ~10× plus lent sur CPU) : abandon des jetons à la place.
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.token_drop = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, n_out))

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.w + self.b                       # (N, F, d)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), self.token_drop(tokens)], 1)
        return self.head(self.encoder(tokens)[:, 0])


class LinearBE(nn.Module):
    """Couche linéaire BatchEnsemble : poids partagé W, adaptateurs r, s et biais propres à chaque membre."""

    def __init__(self, d_in, d_out, k, random_signs=False):
        super().__init__()
        self.weight = nn.Linear(d_in, d_out, bias=False)
        r = torch.ones(k, d_in)
        if random_signs:
            r = torch.where(torch.rand(k, d_in) < 0.5, -1.0, 1.0)
        self.r = nn.Parameter(r)
        self.s = nn.Parameter(torch.ones(k, d_out))
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):      # (N, k, d_in)
        return self.weight(x * self.r) * self.s + self.bias


class TabM(nn.Module):
    def __init__(self, n_features, n_out, width=64, layers=2, k=8, dropout=0.3):
        super().__init__()
        self.k = k
        blocks, d = [], n_features
        for i in range(layers):
            blocks += [LinearBE(d, width, k, random_signs=(i == 0)), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        self.body = nn.Sequential(*blocks)
        self.head = LinearBE(d, n_out, k)

    def forward(self, x):       # (N, k, classes)
        return self.head(self.body(x.unsqueeze(1).expand(-1, self.k, -1)))


def build_model(name, shape, n_out, args):
    n_feat = int(np.prod(shape))
    if name == "cnn1d":
        return CNN1D(shape, n_out, d=args.dim, dropout=args.dropout)
    if name == "cnn_attention":
        return CNNAttention(shape, n_out, d=args.dim, heads=args.tetes, layers=args.couches_attention, dropout=args.dropout)
    if name == "eegnet_adapte":
        return EEGNetAdapte(shape, n_out, dropout=args.dropout)
    if name == "ft_transformer":
        return FTTransformer(n_feat, n_out, d=args.dim, heads=args.tetes, layers=args.couches_attention,
                             dropout=min(args.dropout, 0.2))
    if name == "tabm":
        return TabM(n_feat, n_out, width=args.tabm_largeur, k=args.tabm_k, dropout=args.dropout)
    raise ValueError(name)


# =========================================================================== entraînement
class Stats:
    n_parametres = {}
    echecs = {}
    autotest = {}
    courbes = []


AIDE_PYTORCH = """
  PyTorch plante sur cette machine (souvent un conflit OpenMP entre PyTorch de pip et numpy/scikit-learn
  de conda « base »). Solutions :
    1. conda install -c pytorch pytorch
    2. environnement propre : python3 -m venv ~/venv_ift7030 && source ~/venv_ift7030/bin/activate
                              pip install torch numpy pandas scikit-learn matplotlib scipy pywavelets
    3. pip uninstall torch && pip install torch
  Les modèles de référence (RL, MLP scikit-learn) sont évalués quand même."""


def preprocess(x_tr, x_te):
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    sc = StandardScaler()
    a = sc.fit_transform(imp.fit_transform(x_tr))
    b = sc.transform(imp.transform(x_te))
    return np.clip(a, -6, 6).astype(np.float32), np.clip(b, -6, 6).astype(np.float32)


def train_torch(name, shape, x_tr, y_tr, x_te, n_classes, seed, args, courbe=None):
    """Entraîne un réseau et renvoie (probabilités test, poids d'attention test ou None)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    a, b = preprocess(x_tr, x_te)
    model = build_model(name, shape, n_classes, args)
    Stats.n_parametres[name] = sum(p.numel() for p in model.parameters())
    present = np.unique(y_tr)
    cw = np.ones(n_classes, dtype=np.float32)
    cw[present] = compute_class_weight("balanced", classes=present, y=y_tr)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cw), label_smoothing=args.lissage)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decroissance)
    steps_per_epoch = int(np.ceil(len(a) / args.taille_lot))
    if steps_per_epoch > 1 and len(a) % args.taille_lot == 1:
        steps_per_epoch -= 1
    total = max(1, args.epoques * steps_per_epoch)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (
        1 + np.cos(np.pi * min(1.0, max(0, t + 1 - warm) / max(1, total - warm)))))
    xa, ya = torch.from_numpy(a), torch.from_numpy(y_tr.astype(np.int64))
    model.train()
    for ep in range(args.epoques):
        order = rng.permutation(len(a))
        batches = [order[i:i + args.taille_lot] for i in range(0, len(order), args.taille_lot)]
        if len(batches) > 1 and len(batches[-1]) < 2:   # pas de lot d'un seul sujet (BatchNorm)
            batches[-2] = np.concatenate([batches[-2], batches.pop()])
        pertes = []
        for idx in batches:
            xb = xa[idx]
            if args.bruit > 0:
                xb = xb + args.bruit * torch.randn_like(xb)
            out = model(xb)
            if out.dim() == 3:   # TabM : chaque membre a sa propre perte
                loss = loss_fn(out.reshape(-1, n_classes), ya[idx].repeat_interleave(out.shape[1]))
            else:
                loss = loss_fn(out, ya[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if hasattr(model, "apply_constraints"):
                model.apply_constraints()
            pertes.append(float(loss))
        if courbe is not None:
            Stats.courbes.append(dict(**courbe, modele=name, epoque=ep + 1, perte_entrainement=float(np.mean(pertes))))
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(b))
        proba = (out.softmax(-1).mean(1) if out.dim() == 3 else out.softmax(-1)).numpy()
    attn = model.last_weights.numpy() if isinstance(model, CNNAttention) else None
    return proba.astype(float), attn


def fit_predict(name, shape, x_tr, y_tr, x_te, n_classes, seed, args, courbe=None):
    if name in SK_NAME:
        return sk_fit_predict(SK_NAME[name], seed, x_tr, y_tr, x_te, n_classes, args.c, args.mlp_couches,
                              args.mlp_alpha)[0], None
    return train_torch(name, shape, x_tr, y_tr, x_te, n_classes, seed, args, courbe)


def prepare(x, bases, ref, folds, method):
    """aucune / zscore / filtrage (variables de site mises à 0, forme conservée)."""
    if method == "aucune":
        return [(x[tr], x[te]) for tr, te in folds]
    return eh.prepare_folds(x, bases, ref, folds, method, masque=True)


# =========================================================================== analyses
def run_cv(task, x, y, ids, groups, folds_by_rep, prepared_by_rep, shape, n_classes, args, models,
           harmonisation, class_names, collect):
    labels = list(range(n_classes))
    for name in models:
        t0 = time.time()
        for rep, (folds, prepared) in enumerate(zip(folds_by_rep, prepared_by_rep)):
            seed = args.graine + rep
            proba = np.zeros((len(y), n_classes))
            fold_id = np.zeros(len(y), dtype=int)
            try:
                for k, ((tr, te), (x_tr, x_te)) in enumerate(zip(folds, prepared), 1):
                    courbe = dict(tache=task, harmonisation=harmonisation) if rep == 0 and k == 1 else None
                    proba[te], attn = fit_predict(name, shape, x_tr, y[tr], x_te, n_classes, seed * 100 + k, args, courbe)
                    fold_id[te] = k
                    if attn is not None:
                        collect["attention"].append(pd.DataFrame(attn, columns=collect["canaux"][task]).assign(
                            tache=task, harmonisation=harmonisation, repetition=rep + 1,
                            classe=[class_names[c] for c in y[te]]))
            except Exception as exc:   # un modèle qui échoue n'arrête pas les autres
                msg = f"{type(exc).__name__}: {exc}"
                Stats.echecs[f"{task}/{harmonisation}/{name}"] = msg
                log(f"  {name:<22s} {harmonisation:<9s} ÉCHEC, modèle ignoré : {msg[:200]}")
                break
            row, pred = multiclass_scores(y, proba, labels)
            if n_classes == 2:
                row["sensibilite"] = recall_score(y, pred, zero_division=0)
                row["specificite"] = recall_score(y, pred, pos_label=0, zero_division=0)
            else:
                for c, cname in enumerate(class_names):
                    row[f"rappel_{cname}"] = recall_score(y == c, pred == c, zero_division=0)
            collect["rows"].append(dict(tache=task, harmonisation=harmonisation, modele=name,
                                        famille="reference" if name in REFERENCES else "profond",
                                        repetition=rep + 1, **row))
            collect["preds"].append(pd.DataFrame({
                "tache": task, "harmonisation": harmonisation, "modele": name, "repetition": rep + 1,
                "subject_id": ids, "groupe": groups, "pli": fold_id, "classe": y, "prediction": pred,
                **{f"proba_{c}": proba[:, i] for i, c in enumerate(class_names)}}))
        else:
            sub = pd.DataFrame([r for r in collect["rows"] if r["tache"] == task and r["modele"] == name
                                and r["harmonisation"] == harmonisation])
            log(f"  {name:<22s} {harmonisation:<9s} exact. équilibrée {sub['exactitude_equilibree'].mean():.3f}"
                f" ± {sub['exactitude_equilibree'].std(ddof=1) if len(sub) > 1 else 0:.3f}  ({time.time() - t0:.0f} s)")


def binary_task(base, args, models, collect, orders):
    run = getattr(args, base) or dossier_extraction_defaut(base)
    require(run is not None, f"Aucune extraction pour {base} (03_extraction_entropies.py ou --{base}).")
    info = charger_extraction(Path(run), base, orders, args.agregation)
    channels = info["channels"]
    if args.binaire_canaux_communs:
        channels = [c for c in ["F7", "F3", "F4", "F8", "T7", "T8", "P7", "P8", "O1", "O2"] if c in channels]
    ent, _ = agreger(info, channels)
    rec = info["records"]
    y = rec["label"].astype(int).to_numpy()
    groups = rec["group_validation"].astype(str).to_numpy()
    x = ent.to_numpy(float)
    shape = (len(channels), len(BANDS), len(MEASURES))
    task = f"binaire_{base}"
    collect["canaux"][task] = list(channels)
    log(f"\n=== Binaire {base} : {len(y)} sujets ({(y == 0).sum()} témoins, {(y == 1).sum()} patients), "
        f"{len(channels)} canaux, {x.shape[1]} variables")
    folds_by_rep = [make_folds(y, groups, args.plis, args.graine + r, must_have=y, min_par_strate=2)
                    for r in range(args.repetitions)]
    prepared = [[(x[tr], x[te]) for tr, te in folds] for folds in folds_by_rep]
    run_cv(task, x, y, rec.index, groups, folds_by_rep, prepared, shape, 2, args, models, "aucune",
           ["temoin", "trouble"], collect)
    return x, y, groups, shape


def multiclass_task(ent, meta, common, args, models, collect):
    x = ent.to_numpy(float)
    y = meta["classe"].to_numpy()
    bases = meta["base"].to_numpy()
    groups = meta["groupe"].to_numpy()
    strata = pd.Categorical(bases, BASES).codes * 2 + meta["label_binaire"].to_numpy()
    shape = (len(common), len(BANDS), len(MEASURES))
    collect["canaux"]["multiclasse"] = list(common)
    log(f"\n=== 4 classes : {len(y)} sujets, effectifs {dict(zip(CLASSES, np.bincount(y, minlength=4).tolist()))}, "
        f"{len(common)} canaux communs, {x.shape[1]} variables")
    folds_by_rep = [make_folds(strata, groups, args.plis, args.graine + r, must_have=strata)
                    for r in range(args.repetitions)]
    for method in args.harmonisations:
        prepared = [prepare(x, bases, y == 0, folds, method) for folds in folds_by_rep]
        run_cv("multiclasse", x, y, meta.index, groups, folds_by_rep, prepared, shape, 4, args, models, method,
               CLASSES, collect)
    return x, y, bases, groups, shape


def permute_intra_base(meta, rng):
    g = meta.groupby("groupe").agg(base=("base", "first"), label=("label_binaire", "first"))
    new_lab = g.groupby("base")["label"].transform(lambda s: pd.Series(rng.permutation(s.to_numpy()), index=s.index))
    lab = meta["groupe"].map(new_lab).to_numpy().astype(int)
    return np.where(lab == 1, meta["base"].map(CLASS_OF_BASE), 0).astype(int), lab


def permutation_test(task, x, y, groups, shape, args, collect, bases=None, meta=None):
    """4 classes : permutation intra-base (le site reste, le trouble disparaît). Binaire : globale."""
    n_classes = 4 if task == "multiclasse" else 2
    methods = args.harmonisations if task == "multiclasse" else ["aucune"]
    out = []
    for name in args.permutation_modeles:
        for method in methods:
            rng = np.random.default_rng(args.graine)
            t0 = time.time()
            null = []
            for i in range(args.permutations):
                if task == "multiclasse":
                    yp, lab = permute_intra_base(meta, rng)
                    strata = pd.Categorical(bases, BASES).codes * 2 + lab
                else:
                    g = pd.Series(y, index=groups).groupby(level=0).first()
                    yp = pd.Series(rng.permutation(g.to_numpy()), index=g.index).loc[groups].to_numpy()
                    strata = yp
                seed = args.graine + 10_000 + i
                try:
                    folds = make_folds(strata, groups, args.plis, seed)
                    prepared = prepare(x, bases, yp == 0, folds, method) if task == "multiclasse" else \
                        [(x[tr], x[te]) for tr, te in folds]
                    proba = np.zeros((len(yp), n_classes))
                    for k, ((tr, te), (x_tr, x_te)) in enumerate(zip(folds, prepared), 1):
                        proba[te] = fit_predict(name, shape, x_tr, yp[tr], x_te, n_classes, seed * 100 + k, args)[0]
                except ValueError as exc:
                    log(f"    permutation {i + 1} ignorée : {exc}")
                    continue
                null.append(multiclass_scores(yp, proba, list(range(n_classes)))[0]["exactitude_equilibree"])
            obs = np.mean([r["exactitude_equilibree"] for r in collect["rows"]
                           if r["tache"] == task and r["modele"] == name and r["harmonisation"] == method])
            null = np.array(null)
            out.append(dict(tache=task, modele=name, harmonisation=method,
                            permutation="intra_base" if task == "multiclasse" else "globale",
                            observe=obs, moyenne_nulle=null.mean(), quantile_95_nul=np.quantile(null, 0.95),
                            gain_sur_nul=obs - null.mean(), p_valeur=p_valeur_permutation(obs, null),
                            n_permutations=len(null)))
            log(f"  permutations {task} / {name} [{method}] : nul {null.mean():.3f}, observé {obs:.3f}, "
                f"p = {out[-1]['p_valeur']:.3f} ({time.time() - t0:.0f} s)")
    return out


# =========================================================================== autotest PyTorch
def autotest_child(name, args):
    """Exécuté dans un sous-processus : petit entraînement sur données aléatoires."""
    rng = np.random.default_rng(0)
    shape = (19, len(BANDS), len(MEASURES))
    x = rng.normal(size=(60, int(np.prod(shape))))
    y = np.repeat([0, 1, 2, 3], 15)
    # Mêmes conditions que l'analyse : scikit-learn (OpenMP/MKL) tourne AVANT PyTorch.
    sk_fit_predict("regression_logistique", 0, x, y, x[:10], 4)
    sk_fit_predict("mlp", 0, x, y, x[:10], 4)
    args.epoques = 2
    proba, _ = train_torch(name, shape, x, y, x[:10], 4, 0, args)
    require(np.isfinite(proba).all(), "probabilités non finies")


def autotest_pytorch(models):
    """Teste chaque réseau dans un processus séparé : une « segmentation fault » y tue le
    sous-processus mais pas l'analyse. Renvoie les réseaux utilisables."""
    ok = []
    for name in models:
        cmd = [sys.executable, str(Path(__file__).resolve()), "--autotest", name] + \
              [a for a in sys.argv[1:] if a != "--autotest"]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            code, err = r.returncode, (r.stderr or "").strip().splitlines()
        except subprocess.TimeoutExpired:
            code, err = "délai dépassé (300 s)", []
        if code == 0:
            Stats.autotest[name] = "ok"
            ok.append(name)
            continue
        kind = (f"plantage du processus (signal {-code}, segmentation fault probable)"
                if isinstance(code, int) and code < 0 else f"code de sortie {code}")
        Stats.autotest[name] = f"{kind} ; " + " | ".join(err[-6:])
        log(f"  autotest {name} : ÉCHEC — {Stats.autotest[name][:250]}")
    return ok


# =========================================================================== principal
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for b in BASES:
        p.add_argument(f"--{b}", type=Path, default=None, help=f"Dossier d'extraction {b} (binaire).")
    p.add_argument("--fusion", type=Path, default=None, help="Dossier de 04_fusion (4 classes).")
    p.add_argument("--carte-canaux", type=Path)
    p.add_argument("--agregation", choices=["moyenne", "mediane"], default="moyenne")
    p.add_argument("--mode", choices=["binaire", "multiclasse", "tous"], default="tous")
    p.add_argument("--bases-binaires", nargs="+", choices=BASES, default=BASES)
    p.add_argument("--binaire-canaux-communs", action="store_true")
    p.add_argument("--modeles", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    p.add_argument("--harmonisations", nargs="+", choices=HARMONISATIONS, default=["zscore"])
    p.add_argument("--plis", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--graine", type=int, default=GRAINE)
    p.add_argument("--epoques", type=int, default=150)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--decroissance", type=float, default=1e-2, help="Décroissance des poids (AdamW).")
    p.add_argument("--taille-lot", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bruit", type=float, default=0.1, help="Écart-type du bruit gaussien d'augmentation.")
    p.add_argument("--lissage", type=float, default=0.1, help="Lissage des étiquettes.")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--tetes", type=int, default=4)
    p.add_argument("--couches-attention", type=int, default=1)
    p.add_argument("--tabm-k", type=int, default=8)
    p.add_argument("--tabm-largeur", type=int, default=64)
    p.add_argument("--threads", type=int, default=1, help="Fils PyTorch (défaut 1 : plus stable sur Mac).")
    p.add_argument("--sans-autotest", action="store_true")
    p.add_argument("--autotest", help=argparse.SUPPRESS)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--mlp-couches", type=int, nargs="+", default=[16])
    p.add_argument("--mlp-alpha", type=float, default=1.0)
    p.add_argument("--filtrage-fraction", type=float, default=0.5)
    p.add_argument("--filtrage-arbres", type=int, default=200)
    p.add_argument("--permutations", type=int, default=0)
    p.add_argument("--permutation-modeles", nargs="+", choices=ALL_MODELS, default=["cnn_attention"])
    args = p.parse_args()
    if args.plis < 2 or args.repetitions < 1 or args.epoques < 1:
        p.error("--plis >= 2, --repetitions >= 1, --epoques >= 1.")
    if args.dim % args.tetes:
        p.error("--dim doit être divisible par --tetes.")
    args.modeles = list(dict.fromkeys(args.modeles))
    args.harmonisations = list(dict.fromkeys(args.harmonisations))
    eh.configurer(args)
    torch.set_num_threads(max(1, args.threads))
    warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
    if args.autotest:
        autotest_child(args.autotest, args)
        return
    deep = [m for m in args.modeles if m in DEEP]
    if deep and not args.sans_autotest:
        log(f"Autotest PyTorch {torch.__version__} ({', '.join(deep)}) dans des sous-processus...")
        usable = autotest_pytorch(deep)
        failed = [m for m in deep if m not in usable]
        if failed:
            log(f"Réseaux retirés : {', '.join(failed)}" + (AIDE_PYTORCH if not usable else ""))
            args.modeles = [m for m in args.modeles if m not in failed]
            args.permutation_modeles = [m for m in args.permutation_modeles if m not in failed]
        else:
            log("  autotest : tous les réseaux fonctionnent.")

    start = time.time()
    orders = lire_carte_canaux(args.carte_canaux)
    log(f"PyTorch {torch.__version__}, {torch.get_num_threads()} fil(s) CPU ; modèles : {', '.join(args.modeles)}")
    collect = dict(rows=[], preds=[], attention=[], canaux={})
    perms = []
    fdir = None
    try:
        if args.mode in ("binaire", "tous"):
            for base in args.bases_binaires:
                x, y, groups, shape = binary_task(base, args, args.modeles, collect, orders)
                if args.permutations:
                    perms += permutation_test(f"binaire_{base}", x, y, groups, shape, args, collect)
        if args.mode in ("multiclasse", "tous"):
            common, ent, _, meta, _, fdir = charger_fusion(args.fusion)
            x, y, bases, groups, shape = multiclass_task(ent, meta, common, args, args.modeles, collect)
            if args.permutations:
                perms += permutation_test("multiclasse", x, y, groups, shape, args, collect, bases=bases, meta=meta)
    except (OSError, ValueError) as exc:
        p.exit(1, f"Analyse arrêtée : {exc}\n")

    out = nouveau_dossier("profond")
    rows = pd.DataFrame(collect["rows"])
    rows.to_csv(out / "resultats_par_repetition.csv", index=False)
    summary = summarise(rows, ["tache", "harmonisation", "modele", "famille"])
    summary = summary.sort_values(["tache", "harmonisation", "exactitude_equilibree"], ascending=[True, True, False])
    summary.to_csv(out / "resume.csv", index=False)
    pd.concat(collect["preds"], ignore_index=True).to_csv(out / "predictions_hors_pli.csv", index=False)
    if Stats.courbes:
        pd.DataFrame(Stats.courbes).to_csv(out / "courbes_apprentissage.csv", index=False)
    if collect["attention"]:
        att = pd.concat(collect["attention"], ignore_index=True)
        long = att.melt(id_vars=["tache", "harmonisation", "repetition", "classe"], var_name="electrode",
                        value_name="poids").dropna(subset=["poids"])
        (long.groupby(["tache", "harmonisation", "classe", "electrode"])["poids"].mean().unstack("electrode")
         .to_csv(out / "attention_electrodes.csv"))
    if perms:
        pd.DataFrame(perms).to_csv(out / "permutations.csv", index=False)
    s = summary.copy()
    s["tache_harm"] = s["tache"] + " / " + s["harmonisation"]
    plot_barres(s, "tache_harm", "exactitude_equilibree", "exactitude_equilibree_std", "modele",
                "Exactitude équilibrée par tâche et modèle (moyenne ± écart-type des répétitions)",
                out / "comparaison_modeles.png", ylabel="Exactitude équilibrée")

    log("\n" + "=" * 100 + "\nRÉSUMÉ (moyenne ± écart-type sur les répétitions ; tri par exactitude équilibrée)")
    for (task, method), sub in summary.groupby(["tache", "harmonisation"], sort=False):
        auc = "auc" if "auc" in sub and sub["auc"].notna().any() else "auc_ovr_macro"
        cols = ["modele", "famille", "exactitude_equilibree", "exactitude_equilibree_std", "f1_macro", auc, "brier"]
        cols += [c for c in ["sensibilite", "specificite"] if c in sub and sub[c].notna().any()]
        cols += [c for c in sub.columns if c.startswith("rappel_") and not c.endswith("_std") and sub[c].notna().any()]
        log(f"\n--- {task} (harmonisation : {method})")
        log(fmt_table(sub, cols))
    if perms:
        log("\nTESTS DE PERMUTATION")
        log(fmt_table(pd.DataFrame(perms), ["tache", "modele", "harmonisation", "permutation", "observe",
                                            "moyenne_nulle", "quantile_95_nul", "p_valeur"]))
    for key, msg in Stats.echecs.items():
        log(f"Échec {key} : {msg[:300]}")
    if any(v != "ok" for v in Stats.autotest.values()):
        log("Réseaux retirés par l'autotest : " + ", ".join(k for k, v in Stats.autotest.items() if v != "ok") + AIDE_PYTORCH)
    ecrire_json(out / "parametres.json", {
        "description": "Réseaux profonds sur entropies EEG ; voir l'en-tête du script.",
        "fusion": str(fdir) if fdir else None, "canaux": collect["canaux"],
        "n_parametres_par_reseau": Stats.n_parametres, "autotest_pytorch": Stats.autotest, "echecs": Stats.echecs,
        "validation": f"StratifiedGroupKFold {args.plis} plis × {args.repetitions} répétitions, "
                      f"graines {args.graine}..{args.graine + args.repetitions - 1}",
        "duree_s": round(time.time() - start, 1), "arguments": arguments_dict(args),
        "versions": versions(("numpy", "pandas", "scikit-learn", "torch"))})
    log(f"\nNombre de paramètres par réseau : {Stats.n_parametres}")
    log(f"Durée : {time.time() - start:.0f} s")
    log(f"SORTIE : {out}")


if __name__ == "__main__":
    main()
