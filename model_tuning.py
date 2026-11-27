"""Nested grouped cross-validation with randomized hyperparameter search.

Splits are always made on participants (``groups``). Imputation, scaling, feature selection and the
hyperparameter search are all fitted on the outer training participants only, so the outer test
participants never influence any choice. Used for both window-level and participant-level tables.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
from joblib import parallel_backend
from lightgbm import LGBMClassifier
from lightgbm.basic import LightGBMError
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
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
from xgboost.core import XGBoostError

from eeg_config import GROUPES_PRUDENTS

# n_iter: candidates sampled per model and outer fold (n_iter_trees for the slower tree ensembles);
# repeats: number of different outer splits
PROFILES = {
    'quick': dict(n_iter=8, n_iter_trees=6, outer_folds=3, inner_folds=3, repeats=1),
    'comprehensive': dict(n_iter=40, n_iter_trees=25, outer_folds=5, inner_folds=4, repeats=1),
}
TREE_MODELS = {'random_forest', 'xgboost', 'lightgbm'}

SELECT_PERCENTILE = [10, 25, 50, 100]


class BalancedXGBClassifier(XGBClassifier):
    """Compute balanced weights from the labels of each individual fit."""

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight('balanced', y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


def validation_groups(dataset, subjects):
    """Keep potentially related recordings together, without merging their predictions."""
    subjects = pd.Series(subjects, dtype=str)
    if dataset == 'borisov':
        prudent = GROUPES_PRUDENTS['schizophrenie']
        groups = {
            f'SZ_{recording}': f'__validation_group__{group}'
            for recording, group in prudent.items()
        }
        return subjects.replace(groups).to_numpy()
    return subjects.to_numpy()


def permute_group_labels(y, groups, rng):
    """Permute labels between validation groups, keeping all their windows together."""
    frame = pd.DataFrame({'group': np.asarray(groups), 'label': np.asarray(y)})
    if (frame.groupby('group')['label'].nunique() != 1).any():
        raise ValueError('A validation group contains conflicting labels.')
    labels = frame.groupby('group', sort=False)['label'].first()
    shuffled = pd.Series(rng.permutation(labels.to_numpy()), index=labels.index)
    return frame['group'].map(shuffled).to_numpy()


def grouped_splits(X, y, groups, n_splits, seed):
    """Materialize and validate participant-disjoint, binary stratified folds."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = pd.DataFrame({'group': groups, 'label': y}).groupby('group')['label'].nunique()
    if (labels != 1).any():
        raise ValueError('A validation group contains conflicting labels.')
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups))
    for train, test in splits:
        if np.intersect1d(groups[train], groups[test]).size:
            raise ValueError('A validation group occurs in both training and test folds.')
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            raise ValueError('Each training and validation fold must contain both classes.')
    return splits


def search_space_for_splits(name, space, splits):
    """Restrict kNN candidates to the smallest actual inner-training fold."""
    if name != 'knn':
        return space
    space = space.copy()
    candidates = space['model__n_neighbors']
    if hasattr(candidates, 'dist') and candidates.dist.name == 'randint':
        low, high = candidates.support()
        candidates = range(int(low), int(high) + 1)
    max_neighbors = min(len(train) for train, _ in splits)
    candidates = [int(k) for k in candidates if 1 <= k <= max_neighbors]
    if not candidates:
        raise ValueError(f'No kNN candidates fit the smallest inner-training fold ({max_neighbors} rows).')
    space['model__n_neighbors'] = candidates
    return space


def xgboost_device():
    """Test actual CUDA training; XGBoost can silently switch an unavailable GPU to CPU."""
    probe = XGBClassifier(n_estimators=1, max_depth=1, tree_method='hist', device='cuda', n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            probe.fit(np.array([[0.0], [1.0], [2.0], [3.0]]), np.array([0, 0, 1, 1]))
        except XGBoostError as exc:
            print(f'XGBoost GPU probe failed; using CPU: {exc}', flush=True)
            return 'cpu'
    device = json.loads(probe.get_booster().save_config())['learner']['generic_param'].get('device')
    if device is None:
        print('XGBoost does not report its device; using CPU. GPU training requires XGBoost 2.0+.', flush=True)
        return 'cpu'
    if not device.startswith('cuda'):
        details = '; '.join(str(w.message) for w in caught)
        print(f'XGBoost GPU unavailable; using CPU. {details}', flush=True)
        return 'cpu'
    print(f'XGBoost GPU available: {device}', flush=True)
    return 'cuda'


def lightgbm_device():
    """Test LightGBM's CUDA and OpenCL implementations independently of XGBoost."""
    X = np.arange(80, dtype=float).reshape(-1, 1)
    y = np.tile([0, 1], 40)
    for device in ('cuda', 'gpu'):
        try:
            LGBMClassifier(n_estimators=1, min_child_samples=1, device_type=device,
                           n_jobs=1, verbose=-1).fit(X, y)
        except LightGBMError as exc:
            print(f'LightGBM {device} unavailable: {exc}', flush=True)
        else:
            print(f'LightGBM GPU available: {device}', flush=True)
            return device
    print('LightGBM GPU unavailable; using CPU.', flush=True)
    return 'cpu'


def uses_gpu(name, pipeline):
    params = pipeline.get_params()
    return ((name == 'xgboost' and params['model__device'] == 'cuda')
            or (name == 'lightgbm' and params['model__device_type'] in ('cuda', 'gpu')))


def search_n_jobs(name, pipeline, n_jobs):
    """Avoid concurrent GPU fits competing for the same device."""
    if uses_gpu(name, pipeline):
        return 1
    return n_jobs


def make_pipeline(model, scale=True):
    steps = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    steps.append(('select', SelectPercentile(f_classif)))
    return Pipeline(steps + [('model', model)])


def build_models(seed=42, xgb_device='cpu', lgbm_device='cpu'):
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
             'model__max_features': ['sqrt', 'log2', 0.05, 0.1]}),
        'xgboost': (
            make_pipeline(BalancedXGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=1,
                                                tree_method='hist', device=xgb_device), scale=False),
            {**select, 'model__n_estimators': randint(50, 400), 'model__learning_rate': loguniform(0.01, 0.3),
             'model__max_depth': randint(2, 7), 'model__min_child_weight': randint(1, 10),
             'model__subsample': uniform(0.5, 0.5), 'model__colsample_bytree': uniform(0.2, 0.8),
             'model__reg_lambda': loguniform(0.1, 30), 'model__gamma': uniform(0, 2)}),
        'lightgbm': (
            make_pipeline(LGBMClassifier(class_weight='balanced', subsample_freq=1, random_state=seed,
                                         n_jobs=1, verbose=-1, device_type=lgbm_device), scale=False),
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
    """95% interval of mean repeat balanced accuracy, resampling participants."""
    rng = np.random.default_rng(seed)
    y_pred = np.atleast_2d(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            scores.append(np.mean([balanced_accuracy_score(y_true[idx], pred[idx]) for pred in y_pred]))
    return np.percentile(scores, [2.5, 97.5])


def nested_cv(X, y, groups, models, profile, seed=42, n_jobs=8, label=''):
    """Out-of-fold probabilities (repeats x samples) per model and the winning hyperparameters."""
    cfg = PROFILES[profile] if isinstance(profile, str) else profile
    oof = {name: np.full((cfg['repeats'], len(y)), np.nan) for name in models}
    tuning_rows = []
    for repeat in range(cfg['repeats']):
        outer = grouped_splits(X, y, groups, cfg['outer_folds'], seed + repeat)
        for fold, (train_idx, test_idx) in enumerate(outer):
            inner = grouped_splits(X.iloc[train_idx], y[train_idx], groups[train_idx],
                                   cfg['inner_folds'], seed + repeat)
            for name, (pipeline, space) in models.items():
                start = time.time()
                search = RandomizedSearchCV(
                    clone(pipeline), search_space_for_splits(name, space, inner),
                    n_iter=cfg['n_iter_trees'] if name in TREE_MODELS else cfg['n_iter'],
                    cv=inner, scoring='roc_auc',
                    n_jobs=search_n_jobs(name, pipeline, n_jobs), random_state=seed + repeat,
                    error_score='raise' if uses_gpu(name, pipeline) else np.nan)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    warnings.filterwarnings('always', category=FitFailedWarning)
                    warnings.filterwarnings('always', message='Scoring failed.*')
                    if name == 'xgboost' and pipeline.get_params()['model__device'] == 'cuda':
                        warnings.filterwarnings('always', message='.*(No visible GPU|not compiled with CUDA support).*')
                    # threads: LightGBM crashes on the read-only arrays that process workers receive
                    with parallel_backend('threading'):
                        search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
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
    groups = validation_groups(dataset, table.subject_id)
    models = models or build_models(seed)
    oof, tuning = nested_cv(X, y, groups, models, profile, seed=seed, n_jobs=n_jobs, label=dataset)
    tuning.insert(0, 'dataset', dataset)
    rows, evaluation, prediction_frames = [], {}, []
    for name, probs in oof.items():
        per_repeat = []
        repeat_predictions = []
        for prob in probs:
            frame = pd.DataFrame({'subject_id': table.subject_id, 'label': y, 'probability': prob})
            part = frame.groupby('subject_id').agg(label=('label', 'first'), probability=('probability', 'mean'))
            pred = (part.probability.to_numpy() >= 0.5).astype(int)
            per_repeat.append((balanced_accuracy_score(part.label, pred),
                               roc_auc_score(part.label, part.probability),
                               precision_score(part.label, pred, zero_division=0),
                               recall_score(part.label, pred, zero_division=0),
                               f1_score(part.label, pred, zero_division=0)))
            repeat_predictions.append(pred)
        frame = pd.DataFrame({'subject_id': table.subject_id, 'label': y, 'probability': probs.mean(axis=0)})
        participants = frame.groupby('subject_id').agg(
            label=('label', 'first'), probability=('probability', 'mean')).reset_index()
        y_true = participants.label.to_numpy()
        y_prob = participants.probability.to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        low, high = bootstrap_ci(y_true, repeat_predictions)
        ba, auc, precision, recall, f1 = np.array(per_repeat).T
        rows.append({'dataset': dataset, 'task': task_name, 'model': name,
            'n_participants': len(y_true), 'n_samples': len(y), 'n_features': len(feature_columns),
            'balanced_accuracy': ba.mean(), 'ba_sd_over_repeats': ba.std(), 'ba_ci_low': low, 'ba_ci_high': high,
            'precision': precision.mean(), 'recall': recall.mean(), 'f1': f1.mean(),
            'roc_auc': auc.mean(), 'auc_sd_over_repeats': auc.std(),
            'sample_roc_auc': np.mean([roc_auc_score(y, prob) for prob in probs])})
        evaluation[name] = (y_true, y_pred, y_prob)
        prediction_frames.append(participants.assign(dataset=dataset, model=name, prediction=y_pred))
    return pd.DataFrame(rows), evaluation, pd.concat(prediction_frames), tuning
