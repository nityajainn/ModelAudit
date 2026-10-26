"""Segment evaluation of the verified Phase 1 prediction artifacts."""

from pathlib import Path

from .artifacts import run_context, write_json
from .dataset import CATEGORICAL, TARGET
from .model_audit import _load_prediction_inputs, _verify_prediction_inputs
from .segments import evaluate_segments


def audit_baseline_segments(prepared: str | Path, models: str | Path, output: str | Path,
                            *, columns=("Contract",), threshold=0.5, minimum_rows=30,
                            minimum_class_rows=10, minimum_predicted_positives=10,
                            maximum_groups=50) -> dict:
    """Use declared categorical predictors, never identifiers/targets as segments."""
    prepared, models, output = Path(prepared), Path(models), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    if (not isinstance(columns, (list, tuple)) or not columns
            or not all(isinstance(column, str) and column in CATEGORICAL for column in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError("Select distinct categorical predictor columns for segmentation.")
    reference, predictions, paths, before = _load_prediction_inputs(prepared, models)
    results = {name: evaluate_segments(reference[TARGET].to_numpy(), probabilities,
                                       reference[list(columns)], threshold=threshold,
                                       minimum_rows=minimum_rows, minimum_class_rows=minimum_class_rows,
                                       minimum_predicted_positives=minimum_predicted_positives,
                                       maximum_groups=maximum_groups)
               for name, probabilities in predictions.items()}
    _verify_prediction_inputs(paths, before)
    result = {"format_version": 1, "context": run_context(), "inputs": before,
              "prediction_source": "Phase 1 P(Churn=1) arrays in verified reference CSV row order",
              "model_deserialization_performed": False, "models": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
