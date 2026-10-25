"""Binary probability evaluation without fitting or selecting a threshold."""

from numbers import Integral, Real

import numpy as np
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score

DEFAULT_THRESHOLDS = tuple(i / 10 for i in range(1, 10))


def _vector(value, name):
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional real numerical array.")
    array = array.astype(np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _threshold(value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("A threshold must be a finite real number in [0, 1].")
    return float(value)


def _classification(labels, probabilities, threshold):
    predicted = probabilities >= threshold
    positive = labels == 1
    tp = int(np.count_nonzero(predicted & positive))
    fp = int(np.count_nonzero(predicted & ~positive))
    fn = int(np.count_nonzero(~predicted & positive))
    tn = int(np.count_nonzero(~predicted & ~positive))
    reasons = {}

    def ratio(name, numerator, denominator, reason):
        if not denominator:
            reasons[name] = reason
            return None
        return numerator / denominator

    precision = ratio("precision", tp, tp + fp, "No predicted positives.")
    recall = ratio("recall", tp, tp + fn, "No actual positives.")
    specificity = ratio("specificity", tn, tn + fp, "No actual negatives.")
    f1 = ratio("f1", 2 * tp, 2 * tp + fp + fn, "No actual or predicted positives.")
    balanced = None
    if recall is None or specificity is None:
        reasons["balanced_accuracy"] = "Both actual classes are required."
    else:
        balanced = (recall + specificity) / 2
    return {"threshold": threshold, "confusion_matrix": [[tn, fp], [fn, tp]],
            "accuracy": (tp + tn) / len(labels), "balanced_accuracy": balanced,
            "precision": precision, "recall": recall, "specificity": specificity,
            "f1": f1, "predicted_positive_count": tp + fp,
            "predicted_positive_fraction": (tp + fp) / len(labels),
            "undefined_reasons": reasons}


def evaluate_binary(labels, probabilities, *, threshold=0.5,
                    thresholds=DEFAULT_THRESHOLDS, n_bins=10, minimum_bin_rows=20) -> dict:
    """Evaluate aligned numeric labels (0/1) and P(label=1); inputs are not changed.

    Single-class samples retain meaningful scores, with unsupported scores null.
    Calibration is measured, not fitted. Threshold analysis is descriptive only.
    """
    labels, probabilities = _vector(labels, "Labels"), _vector(probabilities, "Probabilities")
    if labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must have equal lengths and matching row order.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be numerical 0 or 1.")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be in [0, 1].")
    threshold = _threshold(threshold)
    grid = _vector(thresholds, "Threshold grid")
    if (len(grid) > 201 or ((grid < 0) | (grid > 1)).any()
            or not (np.diff(grid) > 0).all()):
        raise ValueError("Threshold grid requires at most 201 strictly increasing values in [0, 1].")
    for name, value in (("n_bins", n_bins), ("minimum_bin_rows", minimum_bin_rows)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if n_bins > 100:
        raise ValueError("n_bins must not exceed 100.")
    n_bins, minimum_bin_rows = int(n_bins), int(minimum_bin_rows)
    positives = int(labels.sum())
    ranking = {"roc_auc": None, "pr_auc_trapezoidal": None, "average_precision": None}
    warnings, ranking_reasons = [], {}
    if 0 < positives < len(labels):
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ranking.update(roc_auc=float(roc_auc_score(labels, probabilities)),
                       pr_auc_trapezoidal=float(auc(recall, precision)),
                       average_precision=float(average_precision_score(labels, probabilities)))
    else:
        ranking_reasons = {name: "Both actual classes are required by this evaluation policy."
                           for name in ranking}
        warnings.append({"code": "single_class", "message": "Ranking metrics require both classes."})
    ranking["undefined_reasons"] = ranking_reasons
    epsilon = float(np.finfo(np.float64).eps)
    clipped = np.clip(probabilities, epsilon, 1 - epsilon)
    log_loss = float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped)))
    # Divide each integer boundary directly: linspace can put 0.3 just above 0.3.
    edges = np.arange(n_bins + 1, dtype=np.float64) / n_bins
    assignments = np.searchsorted(edges[1:-1], probabilities, side="right")
    bins, sparse, weighted_gap = [], [], 0.0
    for index in range(n_bins):
        selected = assignments == index
        count = int(np.count_nonzero(selected))
        mean_probability = float(probabilities[selected].mean()) if count else None
        observed_rate = float(labels[selected].mean()) if count else None
        gap = abs(mean_probability - observed_rate) if count else None
        if count:
            weighted_gap += count / len(labels) * gap
            if count < minimum_bin_rows:
                sparse.append(index)
        bins.append({"index": index, "lower": float(edges[index]),
                     "upper": float(edges[index + 1]), "upper_inclusive": index == n_bins - 1,
                     "count": count, "mean_probability": mean_probability,
                     "observed_positive_fraction": observed_rate, "absolute_gap": gap})
    if sparse:
        warnings.append({"code": "sparse_calibration_bins", "bin_indices": sparse,
                         "minimum_rows": minimum_bin_rows,
                         "message": "Some occupied calibration bins have limited support."})
    return {"format_version": 1,
            "support": {"rows": len(labels), "positives": positives,
                        "negatives": len(labels) - positives, "positive_prevalence": positives / len(labels)},
            "classification": _classification(labels, probabilities, threshold),
            "ranking": ranking,
            "probability_scores": {"brier_score": float(np.mean((probabilities - labels) ** 2)),
                                   "log_loss": log_loss, "log_loss_clip_epsilon": epsilon},
            "calibration": {"strategy": "uniform", "n_bins": n_bins,
                            "minimum_bin_rows": minimum_bin_rows, "bins": bins,
                            "expected_calibration_error": float(weighted_gap)},
            "threshold_analysis": {"selection_performed": False,
                                   "rows": [_classification(labels, probabilities, float(value)) for value in grid]},
            "warnings": warnings,
            "limitations": ["Threshold comparisons on this evaluation sample are descriptive; no optimum is selected.",
                            "Calibration gaps depend on bins and sample size; probability scores do not isolate calibration.",
                            "No calibration model, uncertainty interval, or deployment approval is produced."]}
