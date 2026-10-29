"""Deterministic copied-data experiments; expectation success is not audit approval."""

import math

import numpy as np
import pandas as pd

from .data_audit import TELCO_SCHEMA
from .data_checks import DataSchema, audit_data
from .dataset import CATEGORICAL, FEATURES, IDENTIFIER, NUMERIC, TARGET
from .decisions import summarize_results
from .drift import compare_snapshots
from .evaluation import _vector
from .model_audit import BASELINE_NAMES
from .segments import evaluate_segments

COPY_COLUMN = "simulated_target_copy"
NEW_CATEGORY = "simulated-new-contract"


def _finding(result, check, dataset="reference", column=None):
    matches = [item for item in result["findings"] if item["check"] == check
               and item["dataset"] == dataset
               and (column is None or item["evidence"].get("column") == column)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {check} finding for {dataset}.")
    return matches[0]


def _case(identifier, mutation, parameters, results, evidence, expectations, gates=()):
    decision = summarize_results(results, gates)
    observations = [{"description": description, "met": bool(met)} for description, met in expectations]
    return {"id": identifier, "mutation": mutation, "parameters": parameters,
            "evidence": evidence, "expectations": observations,
            "expectations_met": all(row["met"] for row in observations),
            "audit_outcome": decision["status"], "decision": decision, "results": results}


def _comparisons(reference, predictions, transform):
    results = {}
    for name in BASELINE_NAMES:
        original = reference[FEATURES + [TARGET]].copy(deep=True)
        original["probability"] = np.array(predictions[name], copy=True)
        changed = original.copy(deep=True)
        changed["probability"] = transform(np.array(predictions[name], copy=True))
        results[name] = compare_snapshots(original, changed, numeric=NUMERIC,
                                          categorical=CATEGORICAL, probability="probability", target=TARGET)
    return results


def _prediction_evidence(comparisons):
    return {name: {"reference_roc_auc": comparison["performance"]["reference"]["metrics"]["roc_auc"],
                   "current_roc_auc": comparison["performance"]["current"]["metrics"]["roc_auc"],
                   "supported_roc_auc_difference": comparison["performance"]["difference"]["roc_auc"],
                   "supported_brier_difference": comparison["performance"]["difference"]["brier_score"],
                   "prediction_ks": comparison["predictions"]["ks_distance"],
                   "prediction_histogram_tv": comparison["predictions"]["histogram"]["total_variation"]}
            for name, comparison in comparisons.items()}


def build_scenarios(train, reference, predictions):
    """Return eleven aggregate scenarios from copies of valid paired Telco inputs.

    The file-level checksum scenario is handled separately by experiment_suite.
    No estimator is fitted or invoked on altered features.
    """
    if not isinstance(predictions, dict) or set(predictions) != set(BASELINE_NAMES):
        raise ValueError("The two verified Phase 1 probability arrays are required.")
    predictions = {name: _vector(values, "Probabilities") for name, values in predictions.items()}
    if any(len(values) != len(reference) for values in predictions.values()):
        raise ValueError("Prediction arrays must align with every reference row.")
    baseline = audit_data(train, reference, TELCO_SCHEMA)
    if baseline["counts"]["fail"]:
        raise ValueError("Experiment control has structural data failures; stop before injecting defects.")
    if COPY_COLUMN in train or COPY_COLUMN in reference or NEW_CATEGORY in set(reference["Contract"].dropna()):
        raise ValueError("A reserved scenario feature/category already exists.")
    rows = len(reference)
    missing_count = math.ceil(.30 * rows)
    observed_positions = np.flatnonzero(reference["MonthlyCharges"].notna().to_numpy())
    if len(observed_positions) < missing_count:
        raise ValueError("Not enough observed MonthlyCharges rows for the missingness experiment.")
    control = _comparisons(reference, predictions, lambda values: values)
    control_equal = all(
        all(feature.get("ks_distance", feature.get("total_variation")) == 0
            for feature in comparison["features"].values())
        and comparison["predictions"]["ks_distance"] == 0
        and comparison["predictions"]["histogram"]["total_variation"] == 0
        and all(value is None or value == 0 for value in comparison["performance"]["difference"].values())
        for comparison in control.values())
    # All-missing numeric features cannot establish an identity distance of zero.
    if any(feature.get("kind") == "numeric" and feature["ks_distance"] is None
           for comparison in control.values() for feature in comparison["features"].values()):
        raise ValueError("Control requires at least one observed value per numeric feature.")
    cases = [_case("control", "No change; compare each snapshot with itself.", {"reference_rows": rows},
                   {"data": baseline, "drift": {"models": control}},
                   {"data_counts": baseline["counts"], "models": _prediction_evidence(control)},
                   [("Feature/prediction distances and defined performance changes are zero.", control_equal)])]

    changed = reference.copy(deep=True)
    changed["MonthlyCharges"] = changed["MonthlyCharges"].astype(float)
    changed.iloc[observed_positions[:missing_count], changed.columns.get_loc("MonthlyCharges")] = np.nan
    result = audit_data(train, changed, TELCO_SCHEMA)
    finding = _finding(result, "missingness", column="MonthlyCharges")
    before_missing = int(reference["MonthlyCharges"].isna().sum())
    cases.append(_case("missingness", "Make the first selected observed MonthlyCharges values missing.",
                       {"fraction_selected": .30, "newly_missing_rows": missing_count}, {"data": result},
                       {"before_missing": before_missing, "finding": finding},
                       [("Missing count increases by the selected rows and warns.", finding["status"] == "warn"
                         and finding["evidence"]["count"] == before_missing + missing_count)]))

    count = min(25, rows)
    changed = pd.concat([reference, reference.iloc[:count]], ignore_index=True)
    result = audit_data(train, changed, TELCO_SCHEMA)
    duplicates, ids = _finding(result, "duplicate_rows"), _finding(result, "duplicate_identifiers")
    cases.append(_case("duplicate_rows", "Append the first K complete reference rows.", {"appended_rows": count},
                       {"data": result}, {"duplicates": duplicates, "identifiers": ids},
                       [("Duplicate rows warn and repeated identifiers fail.", duplicates["status"] == "warn"
                         and duplicates["evidence"]["count"] == count and ids["status"] == "fail"
                         and ids["evidence"]["count"] == count)]))

    count = min(25, rows, len(train))
    changed = reference.copy(deep=True)
    changed.iloc[:count, changed.columns.get_loc(IDENTIFIER)] = train[IDENTIFIER].iloc[:count].to_numpy(copy=True)
    result = audit_data(train, changed, TELCO_SCHEMA)
    overlap = _finding(result, "identifier_overlap", "cross_split")
    cases.append(_case("identifier_overlap", "Replace K reference identifiers with training identifiers.",
                       {"replaced_ids": count}, {"data": result}, {"overlap": overlap},
                       [("Identifier overlap fails with exactly K affected rows.", overlap["status"] == "fail"
                         and overlap["evidence"]["reference_rows"] == count)]))

    left, right = train.copy(deep=True), reference.copy(deep=True)
    left[COPY_COLUMN], right[COPY_COLUMN] = left[TARGET].copy(), right[TARGET].copy()
    schema = DataSchema(tuple(NUMERIC) + (COPY_COLUMN,), tuple(CATEGORICAL), TARGET, IDENTIFIER)
    result = audit_data(left, right, schema)
    signals = {name: _finding(result, "leakage_heuristic", name, COPY_COLUMN) for name in ("train", "reference")}
    cases.append(_case("target_copy", "Add a declared numeric predictor copied directly from Churn.",
                       {"added_feature": COPY_COLUMN}, {"data": result}, {"signals": signals},
                       [("Both partitions flag the deliberate target copy.", all(item["status"] == "warn"
                         and "target_copy" in item["evidence"]["signals"] for item in signals.values()))]))

    result = audit_data(train, reference.drop(columns=["Contract"]), TELCO_SCHEMA)
    schema_finding = _finding(result, "schema")
    cases.append(_case("missing_column", "Drop Contract from the reference copy.", {"column": "Contract"},
                       {"data": result}, {"schema": schema_finding},
                       [("Schema fails and names Contract as missing.", schema_finding["status"] == "fail"
                         and "Contract" in schema_finding["evidence"]["missing_columns"])]))

    changed = reference[FEATURES].copy(deep=True)
    changed["MonthlyCharges"] = changed["MonthlyCharges"].astype(float) + 100.0
    result = compare_snapshots(reference[FEATURES], changed, numeric=NUMERIC, categorical=CATEGORICAL)
    metric = result["features"]["MonthlyCharges"]
    cases.append(_case("numeric_shift", "Add 100 to observed MonthlyCharges; do not recompute predictions.",
                       {"additive_shift": 100.0}, {"drift": {"comparison": result}}, {"MonthlyCharges": metric},
                       [("Numeric distribution changes without changing missingness.", metric["ks_distance"] > 0
                         and metric["missingness"]["fraction_difference"] == 0)],
                       [{"name": "Demonstration numeric-distance bound", "path": ["drift", "comparison", "features", "MonthlyCharges", "ks_distance"], "maximum": .10}]))

    changed = reference[FEATURES].copy(deep=True)
    changed["Contract"] = NEW_CATEGORY
    result = compare_snapshots(reference[FEATURES], changed, numeric=NUMERIC, categorical=CATEGORICAL)
    metric = result["features"]["Contract"]
    cases.append(_case("unseen_category", "Replace every Contract value with an unseen synthetic level.",
                       {"replacement": NEW_CATEGORY}, {"drift": {"comparison": result}}, {"Contract": metric},
                       [("All rows are unseen and categorical distance is one.", metric["new_category_fraction"] == 1
                         and math.isclose(metric["total_variation"], 1, abs_tol=1e-12))],
                       [{"name": "Demonstration categorical-distance bound", "path": ["drift", "comparison", "features", "Contract", "total_variation"], "maximum": .10}]))

    shift = max(1, rows // 3)
    for identifier, mutation, transform in (
        ("inverted_probabilities", "Replace saved probabilities by 1-p; retain labels/features.", lambda values: 1-values),
        ("permuted_probabilities", "Rotate saved probabilities; retain original label/feature order.", lambda values: np.roll(values, shift))):
        comparisons = _comparisons(reference, predictions, transform)
        evidence = _prediction_evidence(comparisons)
        expectations, gates = [], []
        for name, values in evidence.items():
            if identifier == "inverted_probabilities":
                met = math.isclose(values["reference_roc_auc"] + values["current_roc_auc"], 1, abs_tol=1e-12)
                description = f"{name}: inverted ROC AUC complements the original ROC AUC."
            else:
                met = values["prediction_ks"] == 0 and values["prediction_histogram_tv"] == 0
                description = f"{name}: marginal prediction distances cannot detect row misalignment."
                gates.append({"name": f"{name} demonstration distribution bound", "path": ["drift", "models", name, "predictions", "ks_distance"], "maximum": .10})
            expectations.append((description, met))
            gates.append({"name": f"{name} demonstration performance bound", "path": ["drift", "models", name, "performance", "difference", "roc_auc"], "minimum": -.05})
        cases.append(_case(identifier, mutation, {"rotation_rows": shift} if identifier == "permuted_probabilities" else {},
                           {"drift": {"models": comparisons}}, {"models": evidence}, expectations, gates))

    labels = reference[TARGET].to_numpy()
    selected = np.r_[np.flatnonzero(labels == 1)[:1], np.flatnonzero(labels == 0)[:30]]
    cohort = pd.DataFrame({"cohort": ["controlled-small-group"] * len(selected)})
    grouped, evidence, expectations, gates = {}, {}, [], []
    for name in BASELINE_NAMES:
        grouped[name] = evaluate_segments(labels[selected], np.asarray(predictions[name])[selected], cohort)
        group = grouped[name]["segments"][0]["groups"][0]
        evidence[name] = {"support": group["support"], "warnings": group["warnings"],
                          "supported_roc_auc_difference": group["difference_from_overall"]["roc_auc"]}
        expectations.append((f"{name}: one-positive support warns and ROC comparison is withheld.",
                             group["support"]["positives"] == 1 and group["difference_from_overall"]["roc_auc"] is None
                             and any(item["code"] == "low_positive_support" for item in group["warnings"])))
        gates.append({"name": f"{name} requires supported group comparison", "path": ["segments", "models", name, "segments", 0, "groups", 0, "difference_from_overall", "roc_auc"], "minimum": -.05})
    cases.append(_case("sparse_segment", "Use a label-selected artificial cohort with one positive and up to thirty negatives.",
                       {"selected_rows": len(selected), "selection_uses_labels": True},
                       {"segments": {"models": grouped}}, {"models": evidence}, expectations, gates))
    return cases
