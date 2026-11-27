"""SHAP explanations (main effects and pairwise interactions) for the tuned models.

The explained model is refitted on every participant of one dataset, with the hyperparameters chosen by the
same grouped randomized search as in ``model_tuning``. The explanations therefore describe what the model uses,
not how well it generalizes (that is what the nested cross-validation measures).
"""
import warnings

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import RandomizedSearchCV

# model_tuning imports LightGBM first: importing shap before LightGBM crashes LightGBM's training on Windows
from model_tuning import (PROFILES, TREE_MODELS, grouped_splits, positive_score,
                          search_n_jobs, search_space_for_splits, uses_gpu, validation_groups)  # noqa: I001
import shap

LINEAR_MODELS = {'logistic_regression', 'lda'}
BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']


def _arrays(table, feature_columns):
    X = table[feature_columns].replace([np.inf, -np.inf], np.nan)
    return X, table.label.astype(int).to_numpy(), table.subject_id.to_numpy()


def fit_final_model(table, feature_columns, model_name, models, profile='comprehensive', seed=42, n_jobs=8,
                    dataset=None):
    """Grouped randomized search on all participants; returns the refitted pipeline and its inner AUC."""
    cfg = PROFILES[profile] if isinstance(profile, str) else profile
    X, y, groups = _arrays(table, feature_columns)
    if dataset is not None:
        groups = validation_groups(dataset, groups)
    pipeline, space = models[model_name]
    splits = grouped_splits(X, y, groups, cfg['inner_folds'], seed)
    search = RandomizedSearchCV(
        clone(pipeline), search_space_for_splits(model_name, space, splits),
        n_iter=cfg['n_iter_trees'] if model_name in TREE_MODELS else cfg['n_iter'],
        cv=splits,
        scoring='roc_auc', n_jobs=search_n_jobs(model_name, pipeline, n_jobs), random_state=seed,
        error_score='raise' if uses_gpu(model_name, pipeline) else np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        warnings.filterwarnings('always', category=FitFailedWarning)
        warnings.filterwarnings('always', message='Scoring failed.*')
        if model_name == 'xgboost' and pipeline.get_params()['model__device'] == 'cuda':
            warnings.filterwarnings('always', message='.*(No visible GPU|not compiled with CUDA support).*')
        with parallel_backend('threading'):
            search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_score_, search.best_params_


def transform(pipeline, table, feature_columns, max_samples=None, seed=42):
    """Rows as the final model sees them (imputed, scaled, selected), with readable feature names."""
    X, _, _ = _arrays(table, feature_columns)
    if max_samples is not None and len(X) > max_samples:
        X = X.sample(max_samples, random_state=seed)
    preprocess = pipeline[:-1]
    return pd.DataFrame(preprocess.transform(X), index=X.index,
                        columns=preprocess.get_feature_names_out(X.columns))


def _positive_class(values):
    """Binary models return (n, f), (n, f, 2) or one array per class depending on the library."""
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    return values[..., -1] if values.ndim == 3 else values


def explain(pipeline, model_name, data, seed=42, kernel_samples=100):
    """SHAP values of the disease class for every row of ``data``; the unit is log-odds for most models."""
    model = pipeline[-1]
    background = data
    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(data.to_numpy(), check_additivity=False)
        rows = data
    elif model_name in LINEAR_MODELS or (model_name == 'svm' and model.kernel == 'linear'):
        explainer = shap.LinearExplainer(model, background.to_numpy())
        values = explainer.shap_values(data.to_numpy())
        rows = data
    else:
        # kNN and non-linear SVM: model-agnostic and slow, so fewer rows and a summarized background
        rows = data.sample(min(kernel_samples, len(data)), random_state=seed)
        explainer = shap.KernelExplainer(lambda d: positive_score(model, d), shap.kmeans(background, 20))
        values = explainer.shap_values(rows.to_numpy(), nsamples=200, silent=True)
    return _positive_class(values), rows


def parse_feature(name):
    """'ch03_alpha_SampEn' -> channel 'ch03', band 'alpha', measure 'SampEn'."""
    channel, band, measure = name.split('_', 2)
    return channel, band, measure


def feature_importance(values, names):
    table = pd.DataFrame({'feature': names, 'mean_abs_shap': np.abs(values).mean(axis=0)})
    table[['channel', 'band', 'measure']] = [parse_feature(n) for n in table.feature]
    return table.sort_values('mean_abs_shap', ascending=False, ignore_index=True)


def group_importance(values, names, by):
    """Mean |sum of the SHAP values| of every feature sharing a channel, band or measure."""
    labels = np.array([parse_feature(n)[{'channel': 0, 'band': 1, 'measure': 2}[by]] for n in names])
    rows = {label: np.abs(values[:, labels == label].sum(axis=1)).mean() for label in np.unique(labels)}
    return pd.Series(rows, name='mean_abs_shap').sort_values(ascending=False)


def interactions(pipeline, data, max_samples=200, seed=42):
    """Pairwise SHAP interaction values (rows, features, features) of a tree model, on a subsample of rows."""
    rows = data.sample(min(max_samples, len(data)), random_state=seed)
    values = shap.TreeExplainer(pipeline[-1]).shap_interaction_values(rows.to_numpy())
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    return (values[..., -1] if values.ndim == 4 else values), rows


def interaction_table(values, names, top=15):
    """Strongest off-diagonal pairs; each pair's two symmetric halves are added together."""
    strength = np.abs(values).mean(axis=0)
    pairs = [(names[i], names[j], 2 * strength[i, j]) for i in range(len(names)) for j in range(i + 1, len(names))]
    table = pd.DataFrame(pairs, columns=['feature_a', 'feature_b', 'mean_abs_interaction'])
    return table.sort_values('mean_abs_interaction', ascending=False, ignore_index=True).head(top)
