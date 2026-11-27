"""Pooled EEG classification with common electrodes and group-disjoint nested CV."""
import re
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from eeg_config import CANAUX, ORDRE_PREFERE
from model_tuning import validation_groups


DATASET_BASE = {'mumtaz': 'depression', 'borisov': 'schizophrenie', 'sakib': 'anxiete'}
DATASETS = tuple(DATASET_BASE)
CLASS_NAMES = ('control', 'depression', 'schizophrenia', 'anxiety')
DISEASE_CLASS = {'mumtaz': 1, 'borisov': 2, 'sakib': 3}
FEATURE_PATTERN = re.compile(r'^ch(\d+)_(.+)$')


def prepare_pooled_features(window_features, window_columns, include_power=True):
    """Align the same assumed electrodes and features without fitting on the data."""
    common = set.intersection(*(set(CANAUX[base]) for base in DATASET_BASE.values()))
    channels = [name for name in ORDRE_PREFERE if name in common]
    if not channels:
        raise ValueError('No common electrodes are available across datasets.')
    frames, expected = [], None
    for dataset, base in DATASET_BASE.items():
        subset = window_features.loc[window_features.dataset == dataset].copy()
        if subset.empty:
            raise ValueError(f'{dataset}: no windows available for pooling.')
        if subset[['subject_id', 'window', 'label']].isna().any().any():
            raise ValueError(f'{dataset}: missing participant, window, or label.')
        if not subset.label.isin([0, 1]).all():
            raise ValueError(f'{dataset}: binary labels must be 0 or 1.')
        if subset.duplicated(['subject_id', 'window']).any():
            raise ValueError(f'{dataset}: duplicate participant/window identifiers.')
        if (subset.groupby('subject_id')['label'].nunique() != 1).any():
            raise ValueError(f'{dataset}: participant labels are inconsistent.')
        rename = {}
        for column in window_columns[dataset]:
            match = FEATURE_PATTERN.fullmatch(column)
            if not match:
                raise ValueError(f'{dataset}: invalid feature name {column!r}.')
            number, suffix = int(match.group(1)), match.group(2)
            if not 1 <= number <= len(CANAUX[base]):
                raise ValueError(f'{dataset}: channel index {number} is out of range.')
            if not include_power and suffix.endswith(('_LogPower', '_RelPower')):
                continue
            electrode = CANAUX[base][number - 1]
            if electrode in common:
                rename[column] = f'{electrode}_{suffix}'
        if len(set(rename.values())) != len(rename):
            raise ValueError(f'{dataset}: multiple features map to the same electrode feature.')
        features = set(rename.values())
        if not features or (expected is not None and features != expected):
            raise ValueError(f'{dataset}: common electrode feature sets do not match across datasets.')
        expected = features
        mapped = subset[list(rename)].rename(columns=rename)
        mapped['dataset'] = dataset
        mapped['subject_id'] = subset.subject_id.astype(str).to_numpy()
        mapped['window'] = subset.window.to_numpy()
        mapped['subject_key'] = dataset + '::' + mapped.subject_id
        mapped['validation_group'] = dataset + '::' + pd.Series(
            validation_groups(dataset, mapped.subject_id), index=mapped.index)
        mapped['binary_label'] = subset.label.astype(int).to_numpy()
        mapped['class_label'] = np.where(mapped.binary_label == 0, 0, DISEASE_CLASS[dataset])
        frames.append(mapped)
    columns = sorted(expected)
    pooled = pd.concat(frames, ignore_index=True)
    if pooled[columns].isna().all(axis=1).any():
        raise ValueError('At least one pooled window has no observed features.')
    duplicates = pooled.loc[pooled.duplicated(columns, keep=False), ['validation_group', *columns]]
    if not duplicates.empty:
        counts = duplicates.groupby(columns, dropna=False).validation_group.nunique()
        if (counts > 1).any():
            raise ValueError('Identical pooled feature windows occur in different validation groups.')
    return pooled, columns, channels


def pooled_splits(frame, strata, n_splits, seed):
    """Stratify groups rather than windows, then expand folds back to window indices."""
    rows = pd.DataFrame({'group': frame.validation_group.to_numpy(), 'stratum': np.asarray(strata)})
    if (rows.groupby('group')['stratum'].nunique() != 1).any():
        raise ValueError('A pooled validation group contains conflicting strata.')
    group_rows = rows.drop_duplicates('group').reset_index(drop=True)
    if group_rows.stratum.value_counts().min() < n_splits:
        raise ValueError(f'Every pooled stratum must contain at least {n_splits} validation groups.')
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    group_splits = cv.split(group_rows, group_rows.stratum)
    splits = []
    for train_groups, test_groups in group_splits:
        train = np.flatnonzero(rows.group.isin(group_rows.group.iloc[train_groups]).to_numpy())
        test = np.flatnonzero(rows.group.isin(group_rows.group.iloc[test_groups]).to_numpy())
        splits.append((train, test))
    return splits


class ParticipantBalancedClassifier(ClassifierMixin, BaseEstimator):
    """Balance classes after correcting for the number of windows per participant."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        classes, inverse = np.unique(y, return_inverse=True)
        totals = np.bincount(inverse, weights=weights)
        if (totals <= 0).any():
            raise ValueError('Every class must have positive participant weight.')
        weights = weights * (weights.sum() / (len(classes) * totals[inverse]))
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def pooled_models(xgb_device='cpu'):
    """Small, explicit searches; every estimator supports window weights."""
    def pipeline(estimator, scale):
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale:
            steps.append(('scale', StandardScaler()))
        return Pipeline(steps + [('model', ParticipantBalancedClassifier(estimator))])

    return {
        'logistic_regression': (
            pipeline(LogisticRegression(max_iter=3000, random_state=42), True),
            {'model__estimator__C': [0.001, 0.01, 0.1, 1, 10, 100]}),
        'random_forest': (
            pipeline(RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=42), False),
            {'model__estimator__max_depth': [3, 5, 8, None],
             'model__estimator__min_samples_leaf': [1, 3, 10],
             'model__estimator__max_features': ['sqrt', 0.2, 0.5]}),
        'xgboost': (
            pipeline(XGBClassifier(n_estimators=150, n_jobs=1, random_state=42,
                                   tree_method='hist', device=xgb_device), False),
            {'model__estimator__max_depth': [2, 3, 5],
             'model__estimator__learning_rate': [0.03, 0.1, 0.2],
             'model__estimator__subsample': [0.7, 1.0]}),
    }


def participant_scores(probabilities, labels, frame, class_names):
    """Average window probabilities for each participant, preserving dataset metadata."""
    values = pd.DataFrame(probabilities, columns=[f'prob_{name}' for name in class_names], index=frame.index)
    values['subject_key'] = frame.subject_key
    values['dataset'] = frame.dataset
    values['label'] = np.asarray(labels)
    if (values.groupby('subject_key')['label'].nunique() != 1).any():
        raise ValueError('A participant has conflicting labels.')
    columns = [f'prob_{name}' for name in class_names]
    participants = values.groupby('subject_key', sort=True).agg(
        dataset=('dataset', 'first'), label=('label', 'first'), **{c: (c, 'mean') for c in columns})
    participants['prediction'] = participants[columns].to_numpy().argmax(axis=1)
    return participants


def evaluate_pooled(pooled, feature_columns, task, models, repeats=3, outer_folds=5,
                    inner_folds=3, n_iter=6, n_jobs=1, seed=42):
    """Evaluate binary, four-class or controls-only dataset identification."""
    if task not in ('binary', 'multiclass', 'control_site'):
        raise ValueError(f'Unknown pooled task: {task}')
    frame = pooled.copy()
    if task == 'control_site':
        frame = frame.loc[frame.binary_label == 0].copy()
        labels = frame.dataset.map({name: i for i, name in enumerate(DATASETS)}).to_numpy()
        names = DATASETS
        strata = frame.dataset.to_numpy()
    else:
        labels = frame.binary_label.to_numpy() if task == 'binary' else frame.class_label.to_numpy()
        names = ('control', 'disease') if task == 'binary' else CLASS_NAMES
        strata = (frame.dataset + '::' + frame.binary_label.astype(str)).to_numpy()
    frame = frame.reset_index(drop=True)
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    counts = frame.groupby('subject_key').subject_key.transform('size').to_numpy()
    window_weights = 1.0 / counts
    metrics, predictions, tuning = [], [], []

    for repeat in range(repeats):
        outer = pooled_splits(frame, strata, outer_folds, seed + repeat)
        for name, (pipeline, space) in models.items():
            start = time.time()
            probabilities = np.full((len(frame), len(names)), np.nan)
            fold_ids = np.zeros(len(frame), dtype=int)
            for fold, (train, test) in enumerate(outer, 1):
                inner_frame = frame.iloc[train].reset_index(drop=True)
                inner = pooled_splits(inner_frame, np.asarray(strata)[train], inner_folds, seed + repeat)

                def score_participants(estimator, X_val, y_val):
                    meta = frame.loc[X_val.index]
                    participants = participant_scores(estimator.predict_proba(X_val), y_val, meta, names)
                    return balanced_accuracy_score(participants.label, participants.prediction)

                search = RandomizedSearchCV(
                    clone(pipeline), space, n_iter=n_iter, cv=inner, scoring=score_participants,
                    n_jobs=1 if name == 'xgboost' else n_jobs, random_state=seed + repeat,
                    error_score='raise')
                search.fit(X.iloc[train], labels[train], model__sample_weight=window_weights[train])
                if not np.array_equal(search.best_estimator_.classes_, np.arange(len(names))):
                    raise ValueError(f'{task}: trained estimator has unexpected classes.')
                probabilities[test] = search.predict_proba(X.iloc[test])
                fold_ids[test] = fold
                tuning.append({'task': task, 'model': name, 'repeat': repeat + 1, 'fold': fold,
                               'inner_balanced_accuracy': search.best_score_, 'best_params': search.best_params_})
            if np.isnan(probabilities).any() or (fold_ids == 0).any():
                raise ValueError(f'{task}: missing out-of-fold predictions.')
            part = participant_scores(probabilities, labels, frame, names)
            part['fold'] = frame.assign(fold=fold_ids).groupby('subject_key')['fold'].first()
            y_true, y_pred = part.label.to_numpy(), part.prediction.to_numpy()
            prob = part[[f'prob_{c}' for c in names]].to_numpy()
            auc = (roc_auc_score(y_true, prob[:, 1]) if len(names) == 2
                   else roc_auc_score(y_true, prob, multi_class='ovr', average='macro',
                                      labels=np.arange(len(names))))
            row = {'task': task, 'model': name, 'repeat': repeat + 1,
                   'n_participants': len(part), 'n_windows': len(frame),
                   'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                   'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0), 'roc_auc': auc}
            recalls = recall_score(y_true, y_pred, labels=np.arange(len(names)), average=None, zero_division=0)
            row.update({f'recall_{c}': value for c, value in zip(names, recalls)})
            if task != 'control_site':
                for dataset in DATASETS:
                    mask = (part.dataset == dataset) & (part.label == 0)
                    row[f'control_recall_{dataset}'] = (part.loc[mask, 'prediction'] == 0).mean()
            metrics.append(row)
            predictions.append(part.reset_index().assign(task=task, model=name, repeat=repeat + 1))
            print(f'{task} repeat {repeat + 1}/{repeats} {name}: '
                  f'balanced accuracy {row["balanced_accuracy"]:.3f} ({time.time() - start:.0f} s)', flush=True)
    return pd.DataFrame(metrics), pd.concat(predictions, ignore_index=True), pd.DataFrame(tuning)
