"""Descriptive segment comparisons with metric-specific support requirements."""

from numbers import Integral

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar

from .evaluation import evaluate_binary

CLASSIFICATION_METRICS = ("accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1")
RANKING_METRICS = ("roc_auc", "pr_auc_trapezoidal", "average_precision")
PROBABILITY_METRICS = ("brier_score", "log_loss")
METRICS = CLASSIFICATION_METRICS + RANKING_METRICS + PROBABILITY_METRICS


def _describe(labels, probabilities, threshold):
    # One fixed threshold and one bin: Phase 3 owns the metric definitions.
    result = evaluate_binary(labels, probabilities, threshold=threshold,
                             thresholds=(threshold,), n_bins=1, minimum_bin_rows=1)
    metrics = {name: result["classification"][name] for name in CLASSIFICATION_METRICS}
    metrics.update({name: result["ranking"][name] for name in RANKING_METRICS})
    metrics.update({name: result["probability_scores"][name] for name in PROBABILITY_METRICS})
    return {"support": {**result["support"],
                        "predicted_positives": result["classification"]["predicted_positive_count"]},
            "confusion_matrix": result["classification"]["confusion_matrix"],
            "metrics": metrics,
            "undefined_reasons": {**result["classification"]["undefined_reasons"],
                                  **result["ranking"]["undefined_reasons"]}}


def _support_checks(description, minimum_rows, minimum_class_rows, minimum_predicted_positives):
    support = description["support"]
    warnings = []
    for key, minimum, code in (("rows", minimum_rows, "low_rows"),
                               ("positives", minimum_class_rows, "low_positive_support"),
                               ("negatives", minimum_class_rows, "low_negative_support"),
                               ("predicted_positives", minimum_predicted_positives,
                                "low_predicted_positive_support")):
        if support[key] < minimum:
            warnings.append({"code": code, "count": support[key], "minimum": minimum})
    if not support["positives"] or not support["negatives"]:
        warnings.append({"code": "single_class", "message": "Some metrics are undefined for this group."})
    excluded = {}
    for metric in METRICS:
        if description["metrics"][metric] is None:
            excluded[metric] = description["undefined_reasons"][metric]
        elif support["rows"] < minimum_rows:
            excluded[metric] = "Too few rows for comparison."
        elif metric == "precision" and support["predicted_positives"] < minimum_predicted_positives:
            excluded[metric] = "Too few predicted positives for a precision comparison."
        elif metric == "recall" and support["positives"] < minimum_class_rows:
            excluded[metric] = "Too few actual positives for a recall comparison."
        elif metric == "specificity" and support["negatives"] < minimum_class_rows:
            excluded[metric] = "Too few actual negatives for a specificity comparison."
        elif (metric in ("balanced_accuracy", "f1") + RANKING_METRICS
              and min(support["positives"], support["negatives"]) < minimum_class_rows):
            excluded[metric] = "Too few observations of one or both classes for comparison."
    description["warnings"] = warnings
    description["comparison_exclusions"] = excluded


def evaluate_segments(labels, probabilities, segments: pd.DataFrame, *, threshold=0.5,
                      minimum_rows=30, minimum_class_rows=10,
                      minimum_predicted_positives=10, maximum_groups=50) -> dict:
    """Compare each categorical column independently using positional row alignment.

    Missing/blank groups are retained as a distinct null category. Observed groups
    partition all rows. No intersectional analysis or statistical inference is made.
    """
    if not isinstance(segments, pd.DataFrame):
        raise TypeError("segments must be a pandas DataFrame.")
    if (not len(segments.columns) or segments.columns.has_duplicates
            or not all(isinstance(name, str) and name.strip() for name in segments.columns)):
        raise ValueError("Segment columns must have unique nonempty string names.")
    for name, value in (("minimum_rows", minimum_rows), ("minimum_class_rows", minimum_class_rows),
                        ("minimum_predicted_positives", minimum_predicted_positives),
                        ("maximum_groups", maximum_groups)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if maximum_groups > 1000:
        raise ValueError("maximum_groups must not exceed 1000.")
    minimum_rows, minimum_class_rows = int(minimum_rows), int(minimum_class_rows)
    minimum_predicted_positives, maximum_groups = int(minimum_predicted_positives), int(maximum_groups)
    overall = _describe(labels, probabilities, threshold)
    if len(segments) != overall["support"]["rows"]:
        raise ValueError("Segment rows must align positionally with labels and probabilities.")
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    _support_checks(overall, minimum_rows, minimum_class_rows, minimum_predicted_positives)
    columns = []
    for name in sorted(segments.columns):
        normalized = []
        for value in segments[name].tolist():
            if isinstance(value, str):
                normalized.append(value if value.strip() else None)
            elif is_scalar(value) and pd.isna(value):
                normalized.append(None)
            else:
                raise ValueError(f"Segment column {name} must contain only strings or missing values.")
        levels = sorted({value for value in normalized if value is not None})
        missing_rows = normalized.count(None)
        if missing_rows:
            levels.append(None)
        if len(levels) > maximum_groups:
            raise ValueError(f"Segment column {name} exceeds maximum_groups; no groups are silently dropped.")
        array = np.array(normalized, dtype=object)
        rows = []
        for level in levels:
            mask = np.array([value == level for value in array], dtype=bool)
            description = _describe(labels[mask], probabilities[mask], threshold)
            _support_checks(description, minimum_rows, minimum_class_rows, minimum_predicted_positives)
            reasons, deltas = {}, {}
            for metric in METRICS:
                reason = description["comparison_exclusions"].get(metric)
                if reason is None and metric in overall["comparison_exclusions"]:
                    reason = "Overall population: " + overall["comparison_exclusions"][metric]
                if reason:
                    reasons[metric], deltas[metric] = reason, None
                else:
                    deltas[metric] = description["metrics"][metric] - overall["metrics"][metric]
            rows.append({"value": level, "is_missing": level is None,
                         "population_fraction": description["support"]["rows"] / len(labels),
                         **description, "difference_from_overall": deltas,
                         "difference_undefined_reasons": reasons})
        ranges = {}
        for metric in METRICS:
            eligible = [row["metrics"][metric] for row in rows
                        if metric not in row["comparison_exclusions"]]
            enough = len(eligible) >= 2
            ranges[metric] = {"eligible_groups": len(eligible), "excluded_groups": len(rows) - len(eligible),
                              "minimum": min(eligible) if enough else None,
                              "maximum": max(eligible) if enough else None,
                              "gap": max(eligible) - min(eligible) if enough else None,
                              "undefined_reason": None if enough else "At least two supported groups are required."}
        columns.append({"column": name, "missing_rows": missing_rows,
                        "warnings": ([{"code": "missing_segment_values", "count": missing_rows}]
                                     if missing_rows else []),
                        "groups": rows, "supported_ranges": ranges})
    return {"format_version": 1, "threshold": float(threshold),
            "policy": {"minimum_rows": int(minimum_rows), "minimum_class_rows": int(minimum_class_rows),
                       "minimum_predicted_positives": int(minimum_predicted_positives),
                       "maximum_groups": int(maximum_groups)},
            "overall": overall, "segments": columns,
            "metric_direction": {metric: "lower_is_better" if metric in PROBABILITY_METRICS
                                 else "higher_is_better" for metric in METRICS},
            "limitations": ["Differences and ranges are descriptive, not significance tests or fairness conclusions.",
                            "The overall population includes the group being compared; comparisons are not independent.",
                            "Support thresholds are review rules, not confidence intervals or proof of reliability.",
                            "Each column is assessed separately; no intersectional groups or threshold tuning are performed."]}
