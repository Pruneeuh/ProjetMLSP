"""Checks for duplicate recordings and windows across validation groups."""
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from model_tuning import validation_groups


def exclude_notebook_recordings(table):
    """Exclude one potentially related Borisov recording from notebook analyses."""
    excluded = table['dataset'].eq('borisov') & table['subject_id'].eq('SZ_387-02w1')
    if excluded.any():
        subjects = sorted(table.loc[excluded, 'subject_id'].unique())
        print(f"Borisov: excluding {', '.join(subjects)} ({excluded.sum()} rows)")
    return table.loc[~excluded].copy()


def audit_window_features(table, feature_columns, dataset):
    """Reject duplicate feature rows belonging to different validation groups."""
    required = {'subject_id', 'label', 'window', *feature_columns}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f'{dataset}: missing columns: {sorted(missing)}')
    if not feature_columns:
        raise ValueError(f'{dataset}: no feature columns to audit')
    if table[['subject_id', 'label', 'window']].isna().any().any():
        raise ValueError(f'{dataset}: missing subject, label, or window identifier')
    if not table['label'].isin([0, 1]).all():
        raise ValueError(f'{dataset}: labels must be binary (0 or 1)')
    if table.duplicated(['subject_id', 'window']).any():
        raise ValueError(f'{dataset}: duplicate subject/window identifiers')
    if (table.groupby('subject_id')['label'].nunique() != 1).any():
        raise ValueError(f'{dataset}: a participant has conflicting labels')
    if table[feature_columns].isna().all(axis=1).any():
        raise ValueError(f'{dataset}: a window has no observed features')

    duplicates = table.loc[table.duplicated(feature_columns, keep=False),
                           ['subject_id', *feature_columns]].copy()
    if not duplicates.empty:
        duplicates['validation_group'] = validation_groups(dataset, duplicates.subject_id)
        counts = duplicates.groupby(feature_columns, dropna=False)['validation_group'].nunique()
        if (counts > 1).any():
            raise ValueError(f'{dataset}: identical feature windows occur in different validation groups')


def audit_recordings(path, table, dataset, fs, window_seconds=10, metadata_path=None):
    """Check exact full-recording and fixed, non-overlapping raw-window copies.

    This does not detect transformed copies, shifted segments, or related people.
    """
    path = Path(path)
    if not path.exists():
        warnings.warn(f'{dataset}: {path} is missing; raw-recording leakage was not checked.', stacklevel=2)
        return False
    participants = table[['subject_id', 'label']].drop_duplicates()
    if metadata_path is not None:
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            warnings.warn(f'{dataset}: {metadata_path} is missing; raw metadata labels were not checked.',
                          stacklevel=2)
        else:
            metadata = pd.read_csv(metadata_path)[['subject_id', 'label']]
            checked = participants.merge(metadata, on='subject_id', how='left', validate='one_to_one',
                                         suffixes=('', '_metadata'))
            if not checked.label.eq(checked.label_metadata).all():
                raise ValueError(f'{dataset}: feature labels or subject IDs do not match raw metadata')
    groups = validation_groups(dataset, participants.subject_id)
    seen_signals, seen_windows = {}, {}
    window = int(round(fs * window_seconds))
    if window <= 0:
        raise ValueError('Raw audit window must contain at least one sample')
    with np.load(path, allow_pickle=False) as recordings:
        for (subject, label), group in zip(participants.itertuples(index=False, name=None), groups):
            signal = recordings[subject]
            if signal.ndim != 2:
                raise ValueError(f'{dataset}: {subject} is not a channels-by-samples recording')
            arrays = [(seen_signals, signal)]
            arrays.extend((seen_windows, signal[:, start:start + window])
                          for start in range(0, signal.shape[1] - window + 1, window))
            for seen, array in arrays:
                digest = hashlib.sha256(np.ascontiguousarray(array).tobytes()).digest()
                if digest in seen:
                    old_subject, old_group, old_label = seen[digest]
                    if old_label != label:
                        raise ValueError(f'{dataset}: identical raw signals have conflicting labels')
                    if old_group != group:
                        raise ValueError(f'{dataset}: {subject} and {old_subject} share raw data '
                                         'across validation groups')
                else:
                    seen[digest] = (subject, group, label)
    return True
