"""Train fixed baseline pipelines and record honest holdout evidence."""

from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .artifacts import fingerprint, run_context, write_json
from .dataset import (CATEGORICAL, FEATURES, IDENTIFIER, MISSING_CATEGORY,
                      NUMERIC, TARGET, load_partitions)


def build_pipeline(kind: str, *, seed: int = 42) -> Pipeline:
    if kind == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=seed)
    elif kind == "random_forest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
    else:
        raise ValueError(f"Unknown baseline: {kind}")
    numerical = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY,
                                 keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transform = ColumnTransformer([
        ("numeric", numerical, NUMERIC), ("categorical", categorical, CATEGORICAL),
    ], remainder="drop")
    return Pipeline([("preprocess", transform), ("classifier", classifier)])


def baseline_metrics(labels, probabilities) -> dict:
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.shape != probabilities.shape:
        raise ValueError("Labels and probabilities must be aligned one-dimensional arrays.")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("Baseline evaluation requires both binary classes.")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1].")
    predicted = probabilities >= 0.5
    tn, fp, fn, tp = (int(value) for value in
                    confusion_matrix(labels, predicted, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn)
    curve_precision, curve_recall, _ = precision_recall_curve(labels, probabilities)
    return {
        "threshold": 0.5, "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn),
        "pr_auc_trapezoidal": float(auc(curve_recall, curve_precision)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "positive_prevalence": float(np.mean(labels)),
        "precision_undefined_reason": "No predicted positives" if precision is None else None,
    }


def train_baselines(prepared: str | Path, destination: str | Path, *, seed: int = 42) -> dict:
    destination, prepared = Path(destination), Path(prepared)
    if destination.exists():
        raise FileExistsError(f"Choose a new model directory: {destination}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    train, reference, manifest = load_partitions(prepared)
    summary = {
        "format_version": 1, "context": run_context(), "seed": seed,
        "preparation_manifest_sha256": fingerprint(prepared / "manifest.json"),
        "partitions": manifest["files"], "features": FEATURES,
        "excluded_columns": [IDENTIFIER, TARGET], "models": {},
    }
    destination.mkdir(parents=True, exist_ok=False)
    for kind in ("logistic_regression", "random_forest"):
        pipeline = build_pipeline(kind, seed=seed)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            pipeline.fit(train[FEATURES], train[TARGET])
            positive = list(pipeline.classes_).index(1)
            probabilities = pipeline.predict_proba(reference[FEATURES])[:, positive]
        model_path = destination / f"{kind}.joblib"
        with model_path.open("xb") as stream:
            joblib.dump(pipeline, stream)
        # Only reload the artifact just created by this process, never arbitrary input.
        reloaded = joblib.load(model_path)
        np.testing.assert_allclose(
            reloaded.predict_proba(reference[FEATURES])[:, positive], probabilities,
            rtol=0, atol=1e-12,
        )
        prediction_path = destination / f"{kind}_probabilities.npy"
        with prediction_path.open("xb") as stream:
            np.save(stream, probabilities, allow_pickle=False)
        summary["models"][kind] = {
            "metrics": baseline_metrics(reference[TARGET], probabilities),
            "parameters": pipeline.named_steps["classifier"].get_params(),
            "transformed_features": len(pipeline.named_steps["preprocess"].get_feature_names_out()),
            "warnings": [str(item.message) for item in observed],
            "artifact_sha256": fingerprint(model_path),
            "probabilities_sha256": fingerprint(prediction_path),
            "serialization_verified": True,
        }
    write_json(destination / "summary.json", summary)
    return summary
