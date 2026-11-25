"""Nested grouped cross-validation with randomized hyperparameter search.

Splits are always made on participants (``groups``). Imputation, scaling, feature selection and the
hyperparameter search are all fitted on the outer training participants only, so the outer test
participants never influence any choice. Used for both window-level and participant-level tables.
"""
import time
import warnings

import numpy as np
import pandas as pd
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# n_iter: candidates sampled per model and outer fold (n_iter_trees for the slower tree ensembles);
# repeats: number of different outer splits
PROFILES = {
    'quick': dict(n_iter=8, n_iter_trees=6, outer_folds=3, inner_folds=3, repeats=1),
    'comprehensive': dict(n_iter=40, n_iter_trees=25, outer_folds=5, inner_folds=4, repeats=1),
}
TREE_MODELS = {'random_forest', 'xgboost', 'lightgbm'}

SELECT_PERCENTILE = [10, 25, 50, 100]


def make_pipeline(model, scale=True):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    steps.append(('select', SelectPercentile(f_classif)))
    return Pipeline(steps + [('model', model)])


def build_models(seed=42):
    """name -> (pipeline, search space). Every space also tunes how many features are kept."""
    select = {'select__percentile': SELECT_PERCENTILE}
    return {
        'logistic_regression': (
            make_pipeline(LogisticRegression(class_weight='balanced', solver='liblinear',
                                             max_iter=2000, random_state=seed)),
            {**select, 'model__penalty': ['l1', 'l2'], 'model__C': loguniform(1e-4, 10)}),
        'svm': (
            make_pipeline(SVC(class_weight='balanced', random_state=seed, cache_size=500, max_iter=200000)),
            {**select, 'model__kernel': ['linear', 'rbf', 'poly'], 'model__C': loguniform(1e-2, 1e2),
             # 2e-3 is about 'scale' for ~800 standardized features; larger gamma makes poly/rbf fits extremely slow
             'model__gamma': loguniform(1e-6, 2e-3), 'model__degree': [2, 3]}),
        'knn': (
            make_pipeline(KNeighborsClassifier()),
            {**select, 'model__n_neighbors': randint(3, 40), 'model__weights': ['uniform', 'distance'],
             'model__p': [1, 2]}),
        'lda': (
            make_pipeline(LinearDiscriminantAnalysis(solver='lsqr')),
            {**select, 'model__shrinkage': ['auto', *np.round(np.linspace(0, 1, 11), 1)]}),
        'random_forest': (
            make_pipeline(RandomForestClassifier(random_state=seed, n_jobs=1), scale=False),
            {**select, 'model__n_estimators': randint(100, 300),
             'model__max_depth': [3, 5, 8, 12, None], 'model__min_samples_leaf': randint(1, 31),
             'model__max_features': ['sqrt', 'log2', 0.05, 0.1],
             'model__class_weight': ['balanced', 'balanced_subsample']}),
        'xgboost': (
            make_pipeline(XGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=1), scale=False),
            {**select, 'model__n_estimators': randint(50, 400), 'model__learning_rate': loguniform(0.01, 0.3),
             'model__max_depth': randint(2, 7), 'model__min_child_weight': randint(1, 10),
             'model__subsample': uniform(0.5, 0.5), 'model__colsample_bytree': uniform(0.2, 0.8),
             'model__reg_lambda': loguniform(0.1, 30), 'model__gamma': uniform(0, 2)}),
        'lightgbm': (
            make_pipeline(LGBMClassifier(class_weight='balanced', subsample_freq=1, random_state=seed,
                                         n_jobs=1, verbose=-1), scale=False),
            {**select, 'model__n_estimators': randint(50, 400), 'model__learning_rate': loguniform(0.01, 0.3),
             'model__num_leaves': randint(3, 32), 'model__min_child_samples': randint(5, 60),
             'model__subsample': uniform(0.5, 0.5), 'model__colsample_bytree': uniform(0.2, 0.8),
             'model__reg_lambda': loguniform(0.1, 30)}),
    }


def positive_score(model, X):
    """Probability of the disease class; SVC (no Platt scaling) gets a sigmoid of its margin."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return 1 / (1 + np.exp(-model.decision_function(X)))


def bootstrap_ci(y_true, y_pred, n_boot=1000, seed=0):
    """95% interval of the balanced accuracy, resampling participants."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return np.percentile(scores, [2.5, 97.5])


def nested_cv(X, y, groups, models, profile, seed=42, n_jobs=8, label=''):
    """Out-of-fold probabilities (repeats x samples) per model and the winning hyperparameters."""
    cfg = PROFILES[profile] if isinstance(profile, str) else profile
    oof = {name: np.full((cfg['repeats'], len(y)), np.nan) for name in models}
    tuning_rows = []
    for repeat in range(cfg['repeats']):
        outer = StratifiedGroupKFold(n_splits=cfg['outer_folds'], shuffle=True, random_state=seed + repeat)
        inner = StratifiedGroupKFold(n_splits=cfg['inner_folds'], shuffle=True, random_state=seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
            weights = compute_sample_weight('balanced', y[train_idx])
            for name, (pipeline, space) in models.items():
                start = time.time()
                search = RandomizedSearchCV(
                    clone(pipeline), space,
                    n_iter=cfg['n_iter_trees'] if name in TREE_MODELS else cfg['n_iter'],
                    cv=inner, scoring='roc_auc',
                    n_jobs=n_jobs, random_state=seed + repeat, error_score=np.nan)
                fit_params = {'model__sample_weight': weights} if name == 'xgboost' else {}
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    # threads: LightGBM crashes on the read-only arrays that process workers receive
                    with parallel_backend('threading'):
                        search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx], **fit_params)
                    oof[name][repeat, test_idx] = positive_score(search.best_estimator_, X.iloc[test_idx])
                tuning_rows.append({'repeat': repeat, 'fold': fold, 'model': name,
                                    'inner_roc_auc': search.best_score_, 'best_params': search.best_params_})
                print(f'  {label} repeat {repeat + 1}/{cfg["repeats"]} fold {fold + 1}/{cfg["outer_folds"]} '
                      f'{name}: inner AUC {search.best_score_:.3f} ({time.time() - start:.0f} s)', flush=True)
    return oof, pd.DataFrame(tuning_rows)


def evaluate_dataset(table, feature_columns, task_name, dataset, profile='comprehensive',
                     seed=42, n_jobs=8, models=None):
    """Nested-CV metrics at participant level. ``table`` has one row per window or per participant."""
    table = table.reset_index(drop=True)
    X = table[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = table.label.astype(int).to_numpy()
    groups = table.subject_id.to_numpy()
    models = models or build_models(seed)
    oof, tuning = nested_cv(X, y, groups, models, profile, seed=seed, n_jobs=n_jobs, label=dataset)
    tuning.insert(0, 'dataset', dataset)
    rows, evaluation, prediction_frames = [], {}, []
    for name, probs in oof.items():
        per_repeat = []
        for prob in probs:
            frame = pd.DataFrame({'subject_id': groups, 'label': y, 'probability': prob})
            part = frame.groupby('subject_id').agg(label=('label', 'first'), probability=('probability', 'mean'))
            pred = (part.probability.to_numpy() >= 0.5).astype(int)
            per_repeat.append((balanced_accuracy_score(part.label, pred),
                               roc_auc_score(part.label, part.probability)))
        frame = pd.DataFrame({'subject_id': groups, 'label': y, 'probability': probs.mean(axis=0)})
        participants = frame.groupby('subject_id').agg(
            label=('label', 'first'), probability=('probability', 'mean')).reset_index()
        y_true = participants.label.to_numpy()
        y_prob = participants.probability.to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        low, high = bootstrap_ci(y_true, y_pred)
        ba, auc = np.array(per_repeat).T
        rows.append({'dataset': dataset, 'task': task_name, 'model': name,
            'n_participants': len(y_true), 'n_samples': len(y), 'n_features': len(feature_columns),
            'balanced_accuracy': ba.mean(), 'ba_sd_over_repeats': ba.std(), 'ba_ci_low': low, 'ba_ci_high': high,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': auc.mean(),
            'sample_roc_auc': roc_auc_score(y, probs.mean(axis=0))})
        evaluation[name] = (y_true, y_pred, y_prob)
        prediction_frames.append(participants.assign(dataset=dataset, model=name, prediction=y_pred))
    return pd.DataFrame(rows), evaluation, pd.concat(prediction_frames), tuning
