"""Explicit inference from a trusted fitted binary estimator; never fit during audit."""

import numpy as np
import pandas as pd


def positive_probabilities(model, features, *, positive_label=1):
    classes = np.asarray(getattr(model, 'classes_', []))
    if classes.ndim != 1 or len(classes) != 2 or classes.dtype.kind not in 'fiu' or set(classes.tolist()) != {0, 1}:
        raise ValueError('Fitted model must have exactly the binary classes 0 and 1.')
    if type(positive_label) is not int or positive_label != 1:
        raise ValueError('This binary schema requires positive_label=1.')
    if not isinstance(features, pd.DataFrame) or features.empty or features.columns.has_duplicates:
        raise ValueError('Prediction features must be a nonempty table with unique columns.')
    values = np.asarray(model.predict_proba(features.copy(deep=True)))
    if values.shape != (len(features), 2) or values.dtype.kind not in 'fiu':
        raise ValueError('predict_proba must return two numerical columns aligned with input rows.')
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Every class probability must be finite and between zero and one.')
    if not np.allclose(values.sum(axis=1), 1, rtol=0, atol=1e-8):
        raise ValueError('Class probabilities must sum to one on each row.')
    return values[:, list(classes).index(positive_label)].astype(float, copy=True)
