"""Read-only, descriptive comparisons between explicitly declared snapshots."""

from collections import Counter
from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from .evaluation import _threshold, _vector
from .segments import METRICS, PROBABILITY_METRICS, _describe, _support_checks


def _positive_integer(value, name, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a positive integer" +
                         (f" at most {maximum}." if maximum else "."))
    return int(value)


def _roles(numeric, categorical, probability, target):
    if not isinstance(numeric, (list, tuple)) or not isinstance(categorical, (list, tuple)):
        raise ValueError("Numeric and categorical roles must be lists or tuples.")
    names = list(numeric) + list(categorical)
    names += [value for value in (probability, target) if value is not None]
    if (not names or not all(isinstance(name, str) and name.strip() for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Roles must have distinct nonempty string column names.")
    if target is not None and probability is None:
        raise ValueError("A target role requires a probability role.")
    return set(numeric) | set(categorical) | ({probability} if probability is not None else set())


def _missing(value):
    return is_scalar(value) and pd.isna(value)


def _numeric(series):
    values = []
    for value in series.tolist():
        if _missing(value):
            values.append(np.nan)
        elif isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"{series.name} must contain real numbers or missing values.")
        else:
            try:
                number = float(value)
            except OverflowError as exc:
                raise ValueError(f"{series.name} has a number outside float64 range.") from exc
            if not np.isfinite(number):
                raise ValueError(f"{series.name} contains a nonfinite number.")
            if isinstance(value, Integral) and int(number) != value:
                raise ValueError(f"{series.name} has an integer that loses float64 precision.")
            values.append(number)
    return np.array(values, dtype=np.float64)


def _categories(series):
    result = []
    for value in series.tolist():
        if isinstance(value, str):
            result.append(value if value.strip() else None)
        elif _missing(value):
            result.append(None)
        else:
            raise ValueError(f"{series.name} must contain strings or missing values.")
    return result


def _ks(reference, current):
    if not len(reference) or not len(current):
        return None
    reference, current = np.sort(reference), np.sort(current)
    points = np.union1d(reference, current)
    difference = (np.searchsorted(reference, points, side="right") / len(reference)
                  - np.searchsorted(current, points, side="right") / len(current))
    return float(np.max(np.abs(difference)))


def _support(reference_count, current_count, minimum):
    return [{"code": "low_rows", "snapshot": name, "count": count, "minimum": minimum}
            for name, count in (("reference", reference_count), ("current", current_count))
            if count < minimum]


def _missingness(reference_missing, current_missing, reference_rows, current_rows):
    first, second = reference_missing / reference_rows, current_missing / current_rows
    return {"reference_count": reference_missing, "current_count": current_missing,
            "reference_fraction": first, "current_fraction": second,
            "fraction_difference": second - first}


def _numeric_drift(reference, current, minimum):
    first, second = reference[~np.isnan(reference)], current[~np.isnan(current)]
    distance = _ks(first, second)
    return {"kind": "numeric", "reference_observed": len(first), "current_observed": len(second),
            "ks_distance": distance,
            "undefined_reason": "At least one nonmissing value is required in each snapshot."
                                if distance is None else None,
            "missingness": _missingness(len(reference) - len(first), len(current) - len(second),
                                        len(reference), len(current)),
            "warnings": _support(len(first), len(second), minimum)}


def _categorical_drift(reference, current, minimum, maximum, smoothing):
    first, second = Counter(reference), Counter(current)
    levels = sorted((first.keys() | second.keys()) - {None})
    if None in first or None in second:
        levels.append(None)
    if len(levels) > maximum:
        raise ValueError("A categorical feature exceeds maximum_categories; no levels were dropped.")
    rows = [{"value": level, "is_missing": level is None,
             "reference_count": first[level], "current_count": second[level],
             "reference_fraction": first[level] / len(reference),
             "current_fraction": second[level] / len(current),
             "new_in_current": level is not None and level not in first} for level in levels]
    unseen = sum(row["current_count"] for row in rows if row["new_in_current"])
    # Base-2 Jensen-Shannon distance, including missing values as a category.
    # Keep total variation as a separate named measurement for compatibility.
    p = np.array([row["reference_fraction"] for row in rows])
    q = np.array([row["current_fraction"] for row in rows])
    p = (1-smoothing)*p + smoothing/len(p)
    q = (1-smoothing)*q + smoothing/len(q)
    midpoint = (p + q) / 2
    divergence = sum(float(np.sum(v[v > 0] * np.log2(v[v > 0] / midpoint[v > 0])))
                     for v in (p, q)) / 2
    return {"kind": "categorical", "categories": rows,
            "jensen_shannon_distance": float(np.sqrt(max(0.0, divergence))),
            "smoothing": smoothing, "smoothing_definition": "(1-epsilon)*observed_fraction + epsilon/K",
            "total_variation": sum(abs(row["current_fraction"] - row["reference_fraction"])
                                   for row in rows) / 2,
            "new_category_rows": unseen, "new_category_fraction": unseen / len(current),
            "missingness": _missingness(first[None], second[None], len(reference), len(current)),
            "warnings": _support(len(reference), len(current), minimum)}


def _probabilities(series):
    values = _vector(series.to_numpy(), "Probabilities")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")
    return values


def _prediction_drift(reference, current, threshold, n_bins, minimum):
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    first = np.bincount(np.searchsorted(edges[1:-1], reference, side="right"), minlength=n_bins)
    second = np.bincount(np.searchsorted(edges[1:-1], current, side="right"), minlength=n_bins)
    bins = [{"index": index, "lower": float(edges[index]), "upper": float(edges[index + 1]),
             "upper_inclusive": index == n_bins - 1,
             "reference_count": int(first[index]), "current_count": int(second[index]),
             "reference_fraction": float(first[index] / len(reference)),
             "current_fraction": float(second[index] / len(current))} for index in range(n_bins)]
    return {"status": "measured", "ks_distance": _ks(reference, current),
            "mean_probability_difference": float(current.mean() - reference.mean()),
            "predicted_positive_fraction_difference": float(np.mean(current >= threshold)
                                                            - np.mean(reference >= threshold)),
            "histogram": {"strategy": "uniform", "bins": bins,
                          "total_variation": float(np.abs(second / len(current)
                                                          - first / len(reference)).sum() / 2)},
            "warnings": _support(len(reference), len(current), minimum)}


def compare_snapshots(reference: pd.DataFrame, current: pd.DataFrame, *, numeric=(),
                      categorical=(), probability=None, target=None, threshold=0.5,
                      minimum_rows=30, minimum_class_rows=10,
                      minimum_predicted_positives=10, maximum_categories=100, n_bins=10, smoothing=1e-6) -> dict:
    """Compare marginal distributions and supported scores; never infer time order.

    Predictions/labels occupy their own snapshot's row order. The caller supplies
    model/positive-class semantics. Unknown or conflicting column roles fail.
    """
    smoothing = _threshold(smoothing)
    if smoothing >= 1:
        raise ValueError("smoothing must be in [0,1).")
    required = _roles(numeric, categorical, probability, target)
    threshold = _threshold(threshold)
    minimum_rows = _positive_integer(minimum_rows, "minimum_rows")
    minimum_class_rows = _positive_integer(minimum_class_rows, "minimum_class_rows")
    minimum_predicted_positives = _positive_integer(minimum_predicted_positives,
                                                   "minimum_predicted_positives")
    maximum_categories = _positive_integer(maximum_categories, "maximum_categories", 1000)
    n_bins = _positive_integer(n_bins, "n_bins", 100)
    for frame in (reference, current):
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Snapshots must be pandas DataFrames.")
        if frame.empty or frame.columns.has_duplicates:
            raise ValueError("Snapshots must be nonempty with unique columns.")
        optional = {target} if target is not None else set()
        if not required <= set(frame.columns) or set(frame.columns) - required - optional:
            raise ValueError("Snapshot columns do not match the declared roles.")
    features = {name: _numeric_drift(_numeric(reference[name]), _numeric(current[name]), minimum_rows)
                for name in sorted(numeric)}
    features.update({name: _categorical_drift(_categories(reference[name]), _categories(current[name]),
                                             minimum_rows, maximum_categories, smoothing)
                     for name in sorted(categorical)})
    predictions = {"status": "skipped", "reason": "No probability column declared."}
    performance = {"status": "skipped", "reason": "Complete labels and probabilities are required in both snapshots."}
    if probability is not None:
        first, second = _probabilities(reference[probability]), _probabilities(current[probability])
        predictions = _prediction_drift(first, second, threshold, n_bins, minimum_rows)
        descriptions = {}
        for name, frame, values in (("reference", reference, first), ("current", current, second)):
            if target is not None and target in frame:
                description = _describe(frame[target].to_numpy(), values, threshold)
                _support_checks(description, minimum_rows, minimum_class_rows, minimum_predicted_positives)
                descriptions[name] = description
        if len(descriptions) == 2:
            deltas, exclusions = {}, {}
            for metric in METRICS:
                reasons = {name: description["comparison_exclusions"][metric]
                           for name, description in descriptions.items()
                           if metric in description["comparison_exclusions"]}
                deltas[metric] = (None if reasons else descriptions["current"]["metrics"][metric]
                                 - descriptions["reference"]["metrics"][metric])
                if reasons:
                    exclusions[metric] = reasons
            performance = {"status": "measured", **descriptions, "difference": deltas,
                           "difference_undefined_reasons": exclusions,
                           "positive_prevalence_difference": descriptions["current"]["support"]["positive_prevalence"]
                                                            - descriptions["reference"]["support"]["positive_prevalence"],
                           "metric_direction": {name: "lower_is_better" if name in PROBABILITY_METRICS
                                                else "higher_is_better" for name in METRICS}}
    return {"format_version": 1, "reference_rows": len(reference), "current_rows": len(current),
            "roles": {"numeric": sorted(numeric), "categorical": sorted(categorical),
                      "probability": probability, "target": target},
            "threshold": threshold, "difference_direction": "current_minus_reference",
            "policy": {"minimum_rows": minimum_rows, "minimum_class_rows": minimum_class_rows,
                       "minimum_predicted_positives": minimum_predicted_positives,
                       "maximum_categories": maximum_categories, "n_bins": n_bins, "smoothing": smoothing},
            "features": features, "predictions": predictions, "performance": performance,
            "limitations": ["Marginal descriptive distances are not significance tests or deployment gates.",
                            "No time ordering, population independence, model identity or causal explanation is inferred.",
                            "No measured drift does not establish safety; drift does not establish performance loss.",
                            "Support thresholds are review rules, not uncertainty intervals."]}
