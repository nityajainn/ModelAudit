"""Audit hash-bound Phase 1 probability arrays without deserializing models."""

import json
from pathlib import Path

import numpy as np

from .artifacts import fingerprint, run_context, write_json
from .dataset import TARGET, load_partitions
from .evaluation import DEFAULT_THRESHOLDS, evaluate_binary

BASELINE_NAMES = ("logistic_regression", "random_forest")


def audit_baseline_predictions(prepared: str | Path, models: str | Path, output: str | Path,
                               *, threshold=0.5, thresholds=DEFAULT_THRESHOLDS,
                               n_bins=10, minimum_bin_rows=20) -> dict:
    """Verify prediction/data pairing and write a new aggregate report.

    Hashes detect changes, not malicious replacement of both data and manifest.
    Only the fixed Phase 1 filenames are read. No joblib file is loaded.
    """
    prepared, models, output = Path(prepared), Path(models), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    paths = {f"prepared/{name}": prepared / name
             for name in ("manifest.json", "train.csv", "reference.csv")}
    paths["models/summary.json"] = models / "summary.json"
    for name in BASELINE_NAMES:
        paths[f"models/{name}_probabilities.npy"] = models / f"{name}_probabilities.npy"
    before = {key: fingerprint(path) for key, path in paths.items()}
    _, reference, manifest = load_partitions(prepared)
    with (models / "summary.json").open(encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict) or type(summary.get("format_version")) is not int or summary["format_version"] != 1:
        raise ValueError("Unsupported baseline summary.")
    if (summary.get("preparation_manifest_sha256") != before["prepared/manifest.json"]
            or summary.get("partitions") != manifest["files"]):
        raise ValueError("Baseline summary does not match the verified preparation and row order.")
    entries = summary.get("models")
    if not isinstance(entries, dict) or set(entries) != set(BASELINE_NAMES):
        raise ValueError("Baseline summary must contain exactly the two Phase 1 models.")
    evaluations = {}
    for name in BASELINE_NAMES:
        entry = entries[name]
        key = f"models/{name}_probabilities.npy"
        if not isinstance(entry, dict) or entry.get("probabilities_sha256") != before[key]:
            raise ValueError(f"Probability checksum mismatch for {name}.")
        try:
            probabilities = np.load(paths[key], allow_pickle=False)
        except (EOFError, ValueError) as exc:
            raise ValueError(f"Invalid numerical probability file for {name}.") from exc
        if not isinstance(probabilities, np.ndarray):
            probabilities.close()
            raise ValueError("Expected a single numerical .npy array.")
        evaluations[name] = evaluate_binary(reference[TARGET].to_numpy(), probabilities,
                                            threshold=threshold, thresholds=thresholds,
                                            n_bins=n_bins, minimum_bin_rows=minimum_bin_rows)
    after = {key: fingerprint(path) for key, path in paths.items()}
    if before != after:
        raise ValueError("An input changed during evaluation; no report was written.")
    result = {"format_version": 1, "context": run_context(), "inputs": before,
              "prediction_source": "Phase 1 P(Churn=1) arrays in verified reference CSV row order",
              "model_deserialization_performed": False, "models": evaluations}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
