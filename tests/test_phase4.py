import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from modelaudit.artifacts import fingerprint
from modelaudit.evaluation import evaluate_binary
from modelaudit.model_audit import audit_baseline_predictions
from modelaudit.segment_audit import audit_baseline_segments
from modelaudit.segments import evaluate_segments
import test_phase3


def group(result, value, column="Contract"):
    part = next(s for s in result["segments"] if s["column"] == column)
    return next(g for g in part["groups"] if g["value"] == value)


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        self.probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.9, 0.1, 0.8, 0.2])
        self.frame = pd.DataFrame({"Contract": ["A"] * 4 + ["B"] * 4})

    def evaluate(self, **kwargs):
        options = dict(minimum_rows=4, minimum_class_rows=2, minimum_predicted_positives=2)
        options.update(kwargs)
        return evaluate_segments(self.labels, self.probabilities, self.frame, **options)

    def test_hand_calculated_group_scores_and_overall_differences(self):
        result = self.evaluate()
        a, b = group(result, "A"), group(result, "B")
        self.assertEqual(a["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(b["confusion_matrix"], [[0, 2], [2, 0]])
        for metric in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
            self.assertEqual(a["metrics"][metric], 1)
            self.assertEqual(b["metrics"][metric], 0)
        self.assertEqual(a["difference_from_overall"]["accuracy"], 0.5)
        self.assertEqual(b["difference_from_overall"]["accuracy"], -0.5)
        self.assertAlmostEqual(a["difference_from_overall"]["brier_score"], -0.35)
        self.assertAlmostEqual(b["difference_from_overall"]["brier_score"], 0.35)
        self.assertEqual(result["metric_direction"]["brier_score"], "lower_is_better")

    def test_supported_ranges_are_max_minus_min(self):
        ranges = self.evaluate()["segments"][0]["supported_ranges"]
        self.assertEqual(ranges["accuracy"], dict(eligible_groups=2, excluded_groups=0,
                         minimum=0, maximum=1, gap=1, undefined_reason=None))
        self.assertAlmostEqual(ranges["brier_score"]["gap"], 0.7)

    def test_group_support_and_confusion_counts_partition_population(self):
        result = self.evaluate()
        groups = result["segments"][0]["groups"]
        for key in ("rows", "positives", "negatives", "predicted_positives"):
            self.assertEqual(sum(g["support"][key] for g in groups), result["overall"]["support"][key])
        np.testing.assert_array_equal(np.sum([g["confusion_matrix"] for g in groups], axis=0),
                                      result["overall"]["confusion_matrix"])
        self.assertEqual(sum(g["population_fraction"] for g in groups), 1)

    def test_exact_support_thresholds_are_eligible(self):
        for row in self.evaluate()["segments"][0]["groups"]:
            self.assertEqual(row["warnings"], [])
            self.assertEqual(row["comparison_exclusions"], {})
            self.assertEqual(row["difference_undefined_reasons"], {})

    def test_low_rows_keep_measurements_but_suppress_all_comparisons(self):
        result = self.evaluate(minimum_rows=5)
        a = group(result, "A")
        self.assertEqual(a["metrics"]["accuracy"], 1)
        self.assertTrue(all(value is None for value in a["difference_from_overall"].values()))
        self.assertIn("low_rows", [w["code"] for w in a["warnings"]])
        self.assertIsNone(result["segments"][0]["supported_ranges"]["accuracy"]["gap"])

    def test_low_class_support_suppresses_only_relevant_comparisons(self):
        result = self.evaluate(minimum_class_rows=3)
        a = group(result, "A")
        for metric in ("recall", "specificity", "f1", "balanced_accuracy", "roc_auc", "average_precision"):
            self.assertIsNone(a["difference_from_overall"][metric])
        for metric in ("accuracy", "precision", "brier_score", "log_loss"):
            self.assertIsNotNone(a["difference_from_overall"][metric])
        self.assertIn("low_positive_support", [w["code"] for w in a["warnings"]])
        self.assertIn("low_negative_support", [w["code"] for w in a["warnings"]])

    def test_precision_support_uses_predicted_positives(self):
        result = self.evaluate(minimum_predicted_positives=3)
        a = group(result, "A")
        self.assertIsNone(a["difference_from_overall"]["precision"])
        self.assertIsNotNone(a["difference_from_overall"]["recall"])
        self.assertIsNotNone(a["difference_from_overall"]["f1"])
        self.assertIn("low_predicted_positive_support", [w["code"] for w in a["warnings"]])

    def test_single_class_groups_keep_undefined_metrics_and_reasons(self):
        self.frame["Contract"] = ["negative" if y == 0 else "positive" for y in self.labels]
        result = self.evaluate()
        negative = group(result, "negative")
        self.assertIsNone(negative["metrics"]["recall"])
        self.assertIsNone(negative["metrics"]["roc_auc"])
        self.assertIn("recall", negative["undefined_reasons"])
        self.assertIn("single_class", [w["code"] for w in negative["warnings"]])
        self.assertIsNone(result["segments"][0]["supported_ranges"]["roc_auc"]["gap"])
        json.dumps(result, allow_nan=False)

    def test_no_predicted_positives_retains_undefined_precision(self):
        self.probabilities[:4] = 0.1
        result = self.evaluate()
        a = group(result, "A")
        self.assertIsNone(a["metrics"]["precision"])
        self.assertEqual(a["metrics"]["recall"], 0)
        self.assertEqual(a["metrics"]["f1"], 0)
        self.assertIn("precision", a["difference_undefined_reasons"])

    def test_missing_and_blank_values_form_one_explicit_group(self):
        self.frame["Contract"] = [None, np.nan, pd.NA, "  ", "B", "B", "B", "B"]
        result = self.evaluate()
        missing = group(result, None)
        self.assertTrue(missing["is_missing"])
        self.assertEqual(missing["support"]["rows"], 4)
        self.assertEqual(result["segments"][0]["missing_rows"], 4)
        self.assertEqual(result["segments"][0]["warnings"][0]["code"], "missing_segment_values")

    def test_literal_missing_marker_does_not_collide_with_missing_group(self):
        self.frame["Contract"] = [None] * 4 + ["__MISSING__"] * 4
        result = self.evaluate()
        self.assertTrue(group(result, None)["is_missing"])
        self.assertFalse(group(result, "__MISSING__")["is_missing"])

    def test_nonblank_category_spelling_is_preserved(self):
        self.frame["Contract"] = ["A"] * 4 + [" A "] * 4
        result = self.evaluate()
        self.assertEqual({g["value"] for g in result["segments"][0]["groups"]}, {"A", " A "})

    def test_all_missing_and_single_group_ranges_are_undefined(self):
        self.frame["Contract"] = None
        result = self.evaluate()
        self.assertEqual(group(result, None)["metrics"], result["overall"]["metrics"])
        ranges = result["segments"][0]["supported_ranges"]
        self.assertEqual(ranges["accuracy"]["eligible_groups"], 1)
        self.assertIsNone(ranges["accuracy"]["gap"])
        self.assertTrue(ranges["accuracy"]["undefined_reason"])

    def test_multiple_columns_are_separate_partitions_not_intersections(self):
        self.frame["Other"] = ["X", "Y"] * 4
        result = self.evaluate()
        self.assertEqual([s["column"] for s in result["segments"]], ["Contract", "Other"])
        for part in result["segments"]:
            self.assertEqual(sum(g["support"]["rows"] for g in part["groups"]), 8)
            self.assertEqual(len(part["groups"]), 2)

    def test_unused_categorical_levels_do_not_create_empty_groups(self):
        self.frame["Contract"] = pd.Categorical(self.frame.Contract, categories=["B", "A", "unused"])
        result = self.evaluate()
        self.assertEqual([g["value"] for g in result["segments"][0]["groups"]], ["A", "B"])

    def test_index_labels_are_not_used_for_alignment(self):
        expected = self.evaluate()
        self.frame.index = [99] * 8
        self.assertEqual(self.evaluate(), expected)

    def test_inputs_remain_unchanged_and_rows_can_be_jointly_reordered(self):
        before = self.labels.copy(), self.probabilities.copy(), self.frame.copy(deep=True)
        result = self.evaluate()
        np.testing.assert_array_equal(self.labels, before[0])
        np.testing.assert_array_equal(self.probabilities, before[1])
        assert_frame_equal(self.frame, before[2])
        order = [4, 5, 6, 7, 0, 1, 2, 3]
        self.labels, self.probabilities = self.labels[order], self.probabilities[order]
        self.frame = self.frame.iloc[order]
        repeated = self.evaluate()
        for old, new in zip(result["segments"][0]["groups"], repeated["segments"][0]["groups"]):
            self.assertEqual(old, new)

    def test_threshold_is_shared_by_every_group_and_equality_is_positive(self):
        self.probabilities[:] = 0.5
        result = self.evaluate(threshold=0.5)
        self.assertEqual(result["overall"]["support"]["predicted_positives"], 8)
        self.assertTrue(all(g["support"]["predicted_positives"] == 4 for g in result["segments"][0]["groups"]))
        result = self.evaluate(threshold=0.6)
        self.assertEqual(result["overall"]["support"]["predicted_positives"], 0)

    def test_overall_metrics_agree_with_phase3(self):
        result = self.evaluate()["overall"]
        phase3 = evaluate_binary(self.labels, self.probabilities)
        self.assertEqual(result["confusion_matrix"], phase3["classification"]["confusion_matrix"])
        self.assertEqual(result["metrics"]["roc_auc"], phase3["ranking"]["roc_auc"])
        self.assertEqual(result["metrics"]["brier_score"], phase3["probability_scores"]["brier_score"])

    def test_cardinality_limit_is_not_silent_truncation(self):
        self.frame["Contract"] = [str(i) for i in range(8)]
        with self.assertRaisesRegex(ValueError, "maximum_groups"):
            self.evaluate(maximum_groups=7)
        self.assertEqual(len(self.evaluate(maximum_groups=8)["segments"][0]["groups"]), 8)

    def test_invalid_group_values_are_not_coerced_to_strings(self):
        for value in (1, True, np.inf, ["A"], {"A": 1}):
            with self.subTest(value=value):
                self.frame = pd.DataFrame({"Contract": [value] * 8})
                with self.assertRaisesRegex(ValueError, "only strings or missing"):
                    self.evaluate()

    def test_invalid_group_tables_and_alignment_are_rejected(self):
        for frame in (pd.DataFrame(index=range(8)), pd.DataFrame({"": ["A"] * 8}),
                      pd.DataFrame(np.full((8, 2), "A"), columns=["same", "same"]),
                      pd.DataFrame({"Contract": ["A"] * 7})):
            with self.subTest(columns=frame.columns):
                self.frame = frame
                with self.assertRaises(ValueError):
                    self.evaluate()
        self.frame = ["A"] * 8
        with self.assertRaises(TypeError):
            self.evaluate()

    def test_invalid_policy_and_binary_inputs_are_rejected(self):
        for options in ({"minimum_rows": 0}, {"minimum_rows": True}, {"minimum_class_rows": 1.5},
                        {"minimum_predicted_positives": -1}, {"maximum_groups": 1001},
                        {"maximum_groups": False}, {"threshold": np.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.evaluate(**options)
        self.probabilities[0] = np.nan
        with self.assertRaises(ValueError):
            self.evaluate()

    def test_numpy_policy_and_missing_groups_serialize_strictly(self):
        self.frame.loc[0, "Contract"] = None
        result = self.evaluate(minimum_rows=np.int64(5), minimum_class_rows=np.int64(3))
        json.dumps(result, allow_nan=False)


class SegmentArtifactTests(unittest.TestCase):
    setUp = test_phase3.PredictionArtifactTests.setUp
    save_summary = test_phase3.PredictionArtifactTests.save_summary

    def audit(self, **kwargs):
        return audit_baseline_segments(self.prepared, self.models, self.output, **kwargs)

    def test_integration_matches_phase3_overall_and_preserves_artifacts(self):
        before = {str(p): fingerprint(p) for d in (self.prepared, self.models) for p in d.iterdir()}
        with patch("joblib.load", side_effect=AssertionError("No model deserialization")):
            result = self.audit()
        phase3 = audit_baseline_predictions(self.prepared, self.models, self.root / "phase3.json")
        for name, model in result["models"].items():
            self.assertEqual(model["overall"]["metrics"]["f1"], phase3["models"][name]["classification"]["f1"])
            self.assertEqual(model["segments"][0]["column"], "Contract")
            self.assertEqual(sum(g["support"]["rows"] for g in model["segments"][0]["groups"]), len(self.labels))
        self.assertEqual(result, json.loads(self.output.read_text()))
        self.assertFalse(result["model_deserialization_performed"])
        self.assertEqual(len(result["inputs"]), 6)
        self.assertEqual(before, {str(p): fingerprint(p) for d in (self.prepared, self.models) for p in d.iterdir()})

    def test_custom_categorical_columns_are_supported(self):
        result = self.audit(columns=["Contract", "PaymentMethod"])
        for model in result["models"].values():
            self.assertEqual([s["column"] for s in model["segments"]], ["Contract", "PaymentMethod"])

    def test_ids_targets_numeric_unknown_and_duplicate_columns_rejected(self):
        for columns in (("customerID",), ("Churn",), ("tenure",), ("unknown",),
                        ("Contract", "Contract"), (), "Contract"):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                self.audit(columns=columns)
        self.assertFalse(self.output.exists())

    def test_existing_report_is_preserved(self):
        self.audit()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual(before, self.output.read_bytes())

    def test_corrupt_predictions_stop_before_writing(self):
        path = self.models / "random_forest_probabilities.npy"
        path.write_bytes(b"broken")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_wrong_preparation_binding_stops_before_writing(self):
        self.summary["preparation_manifest_sha256"] = "wrong"
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_changed_input_during_segment_evaluation_is_rejected(self):
        def mutate(*args, **kwargs):
            self.summary_path.write_text("changed")
            return evaluate_segments(*args, **kwargs)
        with patch("modelaudit.segment_audit.evaluate_segments", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.audit()
        self.assertFalse(self.output.exists())

    def test_invalid_shape_with_matching_checksum_is_rejected(self):
        path = self.models / "logistic_regression_probabilities.npy"
        np.save(path, np.ones((10, 1)))
        self.summary["models"]["logistic_regression"]["probabilities_sha256"] = fingerprint(path)
        self.save_summary()
        with self.assertRaises(ValueError):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_command_success_overwrite_and_invalid_column_exit(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_segments.py"
        command = [sys.executable, str(script), "--prepared", str(self.prepared),
                   "--models", str(self.models), "--output", str(self.output)]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("Evaluated segments for 2", first.stdout)
        second = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(second.returncode, 2)
        third = subprocess.run(command + ["--output", str(self.root / "invalid.json"),
                                          "--columns", "customerID"], text=True, capture_output=True)
        self.assertEqual(third.returncode, 2)
        self.assertFalse((self.root / "invalid.json").exists())


if __name__ == "__main__":
    unittest.main()
