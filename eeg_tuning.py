"""Grouped participant-level hyperparameter search shared by steps 05, 06 and 09."""

# LightGBM must load before sklearn.svm / sklearn.neural_network on the Windows setup.
from model_tuning import (PROFILES, TREE_MODELS, build_models, lightgbm_device,
                          search_space_for_splits, xgboost_device)

import numpy as np
from scipy.stats import loguniform
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import eeg_harmonisation as eh
from eeg_utils import make_folds


MODEL_NAMES = {
    "regression_logistique": "logistic_regression",
    "svm": "svm",
    "svm_rbf": "svm",
    "knn": "knn",
    "lda": "lda",
    "foret_aleatoire": "random_forest",
    "xgboost": "xgboost",
    "lightgbm": "lightgbm",
}


def detect_devices(names):
    """Probe only requested tree backends; other models run on CPU."""
    return {
        "xgboost": xgboost_device() if "xgboost" in names else "cpu",
        "lightgbm": lightgbm_device() if "lightgbm" in names else "cpu",
    }


def model_and_space(name, seed, devices, n_classes):
    if name == "reference_majoritaire":
        return DummyClassifier(strategy="prior"), {}
    models = build_models(seed=seed, xgb_device=devices["xgboost"],
                          lgbm_device=devices["lightgbm"])
    if name == "mlp":
        pipeline = clone(models["logistic_regression"][0])
        pipeline.set_params(model=MLPClassifier(solver="lbfgs", max_iter=5000, random_state=seed))
        return pipeline, {
            "select__percentile": [10, 25, 50, 100],
            "model__hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
            "model__alpha": loguniform(0.001, 100),
        }
    key = MODEL_NAMES[name]
    pipeline, space = models[key]
    pipeline.set_params(imputer__keep_empty_features=True)
    if name in ("svm", "svm_rbf"):
        pipeline.set_params(model=SVC(class_weight="balanced", probability=True, random_state=seed,
                                      cache_size=500, max_iter=200000))
        if name == "svm_rbf":
            space = {**space, "model__kernel": ["rbf"]}
    if name == "regression_logistique" and n_classes > 2:
        pipeline.set_params(model__solver="saga")
    return pipeline, space


def predict_proba(model, x, n_classes):
    result = np.zeros((len(x), n_classes))
    result[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(x)
    return result


def tune_and_fit(name, x, y, groups, strata, seed, devices, profile="comprehensive",
                 inner_folds=None, draws=None, bases=None, reference=None, harmonisation="aucune",
                 enabled=True, fixed_c=1.0, fixed_mlp_layers=(16,), fixed_mlp_alpha=1.0):
    """Tune on inner group-disjoint folds, refitting harmonization in each fold."""
    x, y, groups, strata = map(np.asarray, (x, y, groups, strata))
    classes = np.unique(y)
    model, space = model_and_space(name, seed, devices, len(classes))
    if not enabled or not space:
        if space:
            model.set_params(select__percentile=100)
            if name == "regression_logistique":
                model.set_params(model__C=fixed_c)
            elif name == "mlp":
                model.set_params(model__hidden_layer_sizes=tuple(fixed_mlp_layers), model__alpha=fixed_mlp_alpha)
        return model, {"best_params": {}, "inner_balanced_accuracy": None}

    cfg = PROFILES[profile]
    n_splits = inner_folds or cfg["inner_folds"]
    folds = make_folds(strata, groups, n_splits, seed, must_have=strata, min_par_strate=2)
    for tr, va in folds:
        if not np.array_equal(np.unique(y[tr]), classes) or not np.array_equal(np.unique(y[va]), classes):
            raise ValueError(f"{name}: inner folds must contain every class; reduce inner folds.")
    if name == "knn":
        space = search_space_for_splits("knn", space, folds)

    prepared = []
    for tr, va in folds:
        if bases is None or harmonisation == "aucune":
            x_tr, x_va = x[tr], x[va]
        else:
            x_tr, x_va = eh.harmonise(x[tr], x[va], bases[tr], bases[va],
                                       reference[tr], harmonisation)
        prepared.append((tr, va, x_tr, x_va))

    if "select__percentile" in space:
        min_features = min(part[2].shape[1] for part in prepared)
        choices = [p for p in space["select__percentile"] if int(min_features * p / 100) >= 1]
        if not choices:
            raise ValueError(f"{name}: no feature-selection candidate retains a feature.")
        space = {**space, "select__percentile": choices}
    n_iter = draws or (cfg["n_iter_trees"] if MODEL_NAMES.get(name) in TREE_MODELS else cfg["n_iter"])
    best_score, best_params = -np.inf, None
    for params in ParameterSampler(space, n_iter=n_iter, random_state=seed):
        scores = []
        for tr, va, x_tr, x_va in prepared:
            candidate = clone(model).set_params(**params)
            candidate.fit(x_tr, y[tr])
            prediction = candidate.classes_[np.argmax(candidate.predict_proba(x_va), axis=1)]
            scores.append(balanced_accuracy_score(y[va], prediction))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_params = score, params
    if best_params is None:
        raise ValueError(f"{name}: no valid tuning candidates.")
    return clone(model).set_params(**best_params), {
        "best_params": best_params, "inner_balanced_accuracy": best_score,
    }
