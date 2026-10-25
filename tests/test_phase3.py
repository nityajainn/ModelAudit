import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss

from modelaudit.artifacts import fingerprint, write_json
from modelaudit.baselines import baseline_metrics
from modelaudit.dataset import load_partitions, prepare_dataset
from modelaudit.evaluation import evaluate_binary
from modelaudit.model_audit import BASELINE_NAMES, audit_baseline_predictions
from test_phase1 import fixture


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.array([0, 1, 0, 1])
        self.probabilities = np.array([0.1, 0.8, 0.7, 0.4])

    def evaluate(self, **kwargs):
        return evaluate_binary(self.labels, self.probabilities, **kwargs)

    def test_hand_calculated_classification(self):
        metrics = self.evaluate()["classification"]
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        for name in ("precision", "recall", "specificity", "accuracy", "balanced_accuracy", "f1"):
            self.assertEqual(metrics[name], 0.5)
        self.assertEqual(metrics["predicted_positive_count"], 2)
        self.assertEqual(metrics["undefined_reasons"], {})

    def test_hand_calculated_ranking(self):
        metrics = self.evaluate()["ranking"]
        self.assertAlmostEqual(metrics["roc_auc"], 3 / 4)
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["pr_auc_trapezoidal"], 19 / 24)

    def test_hand_calculated_probability_scores(self):
        scores = self.evaluate()["probability_scores"]
        self.assertAlmostEqual(scores["brier_score"], 0.225)
        self.assertAlmostEqual(scores["log_loss"], -math.log(0.9 * 0.8 * 0.3 * 0.4) / 4)
        self.assertAlmostEqual(scores["brier_score"], brier_score_loss(self.labels, self.probabilities))
        self.assertAlmostEqual(scores["log_loss"], log_loss(self.labels, self.probabilities))

    def test_hand_calculated_calibration(self):
        calibration = self.evaluate(n_bins=2)["calibration"]
        self.assertEqual([b["count"] for b in calibration["bins"]], [2, 2])
        self.assertEqual([b["mean_probability"] for b in calibration["bins"]], [0.25, 0.75])
        self.assertEqual([b["observed_positive_fraction"] for b in calibration["bins"]], [0.5, 0.5])
        self.assertEqual(calibration["expected_calibration_error"], 0.25)

    def test_perfect_and_inverted_predictions(self):
        for probabilities, auc_expected, brier_expected in ((self.labels, 1, 0), (1-self.labels, 0, 1)):
            result = evaluate_binary(self.labels, probabilities)
            self.assertEqual(result["ranking"]["roc_auc"], auc_expected)
            self.assertEqual(result["probability_scores"]["brier_score"], brier_expected)
            self.assertTrue(math.isfinite(result["probability_scores"]["log_loss"]))

    def test_constant_scores_and_ties_have_correct_ranking(self):
        result = evaluate_binary(self.labels, [0.5] * 4)
        self.assertEqual(result["ranking"]["roc_auc"], 0.5)
        self.assertEqual(result["ranking"]["average_precision"], 0.5)
        self.assertEqual(result["ranking"]["pr_auc_trapezoidal"], 0.75)

    def test_no_predicted_positives_keeps_precision_undefined(self):
        result = evaluate_binary(self.labels, [0.1] * 4)["classification"]
        self.assertIsNone(result["precision"])
        self.assertIn("precision", result["undefined_reasons"])
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)

    def test_all_negative_sample_retains_only_supported_scores(self):
        result = evaluate_binary([0, 0], [0.1, 0.2])
        metrics = result["classification"]
        for name in ("precision", "recall", "f1", "balanced_accuracy"):
            self.assertIsNone(metrics[name])
            self.assertIn(name, metrics["undefined_reasons"])
        self.assertEqual(metrics["specificity"], 1)
        self.assertEqual(metrics["accuracy"], 1)
        for name in ("roc_auc", "average_precision", "pr_auc_trapezoidal"):
            self.assertIsNone(result["ranking"][name])
        self.assertIn("single_class", [w["code"] for w in result["warnings"]])
        json.dumps(result, allow_nan=False)

    def test_all_positive_sample_and_single_row_are_supported(self):
        result = evaluate_binary([1], [0.8])
        metrics = result["classification"]
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], 1)
        self.assertEqual(metrics["f1"], 1)
        self.assertIsNone(metrics["specificity"])
        self.assertIsNone(result["ranking"]["roc_auc"])
        self.assertAlmostEqual(result["probability_scores"]["brier_score"], 0.04)

    def test_false_positives_on_negative_sample_have_zero_f1(self):
        metrics = evaluate_binary([0, 0], [0.8, 0.9])["classification"]
        self.assertEqual(metrics["precision"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertIsNone(metrics["recall"])

    def test_bin_boundaries_include_zero_one_and_right_bin(self):
        result = evaluate_binary([0, 0, 1, 1, 1], [0, 0.25, 0.5, 0.75, 1], n_bins=4)
        bins = result["calibration"]["bins"]
        self.assertEqual([b["count"] for b in bins], [1, 1, 1, 2])
        self.assertEqual([b["upper_inclusive"] for b in bins], [False, False, False, True])

    def test_decimal_bin_boundaries_go_to_their_right_bin(self):
        probabilities = [i / 10 for i in range(11)]
        result = evaluate_binary([0, 1] * 5 + [1], probabilities, n_bins=10)
        self.assertEqual([b["count"] for b in result["calibration"]["bins"]], [1] * 9 + [2])
        adjacent = [np.nextafter(0.3, 0), 0.3, np.nextafter(0.3, 1)]
        counts = [b["count"] for b in evaluate_binary([0, 1, 1], adjacent)["calibration"]["bins"]]
        self.assertEqual(counts[2:4], [1, 2])

    def test_empty_bins_are_kept_with_null_values(self):
        bins = evaluate_binary([0, 1], [0, 1], n_bins=4)["calibration"]["bins"]
        self.assertEqual(bins[1]["count"], 0)
        for name in ("mean_probability", "observed_positive_fraction", "absolute_gap"):
            self.assertIsNone(bins[1][name])

    def test_sparse_support_warning_excludes_empty_bins_and_respects_equality(self):
        result = self.evaluate(n_bins=2, minimum_bin_rows=2)
        self.assertEqual(result["warnings"], [])
        result = self.evaluate(n_bins=2, minimum_bin_rows=3)
        self.assertEqual(result["warnings"][0]["bin_indices"], [0, 1])
        result = evaluate_binary([0, 1], [0, 1], n_bins=4)
        self.assertEqual(result["warnings"][0]["bin_indices"], [0, 3])

    def test_one_bin_ece_compares_mean_probability_and_prevalence(self):
        result = self.evaluate(n_bins=1)
        self.assertAlmostEqual(result["calibration"]["expected_calibration_error"], 0)
        self.assertEqual(result["calibration"]["bins"][0]["count"], 4)

    def test_threshold_equality_and_endpoint_behavior(self):
        result = evaluate_binary([0, 1, 1], [0, 0.5, 1], thresholds=[0, 0.5, 1])
        rows = result["threshold_analysis"]["rows"]
        self.assertEqual([r["predicted_positive_count"] for r in rows], [3, 2, 1])
        self.assertEqual(result["classification"]["confusion_matrix"], [[1, 0], [0, 2]])
        self.assertFalse(result["threshold_analysis"]["selection_performed"])

    def test_threshold_sweep_support_and_recall_are_nonincreasing(self):
        rows = self.evaluate(thresholds=[0, 0.2, 0.4, 0.6, 0.8, 1])["threshold_analysis"]["rows"]
        for key in ("predicted_positive_count", "recall"):
            values = [r[key] for r in rows]
            self.assertEqual(values, sorted(values, reverse=True))
        self.assertTrue(all(sum(map(sum, r["confusion_matrix"])) == 4 for r in rows))

    def test_fixed_threshold_is_independent_of_analysis_grid(self):
        result = self.evaluate(threshold=0.7, thresholds=[0.1, 0.9])
        self.assertEqual(result["classification"]["threshold"], 0.7)
        self.assertEqual(result["classification"]["predicted_positive_count"], 2)
        self.assertEqual([r["threshold"] for r in result["threshold_analysis"]["rows"]], [0.1, 0.9])

    def test_invalid_labels_and_shapes_rejected(self):
        for labels, probabilities in (([], []), ([[0, 1]], [[0.1, 0.8]]),
                                      ([0, 1], [0.1]), ([0, 2], [0.1, 0.8]),
                                      ([0, np.nan], [0.1, 0.8]), ([0, 0.5], [0.1, 0.8]),
                                      ([False, True], [0.1, 0.8]), (["0", "1"], [0.1, 0.8])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                evaluate_binary(labels, probabilities)

    def test_invalid_probabilities_rejected_without_coercion(self):
        for values in ([0, np.nan], [0, np.inf], [-0.1, 0.5], [0.1, 1.1],
                       ["0.1", "0.5"], [False, True], [0.1j, 0.5j], [None, 0.5]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                evaluate_binary([0, 1], values)

    def test_invalid_options_are_rejected(self):
        for options in ({"threshold": True}, {"threshold": np.nan}, {"threshold": -0.1},
                        {"threshold": 1.1}, {"threshold": "0.5"}, {"n_bins": 0},
                        {"n_bins": True}, {"n_bins": 2.5}, {"n_bins": 101},
                        {"minimum_bin_rows": 0}, {"minimum_bin_rows": False},
                        {"thresholds": []}, {"thresholds": [0.5, 0.5]},
                        {"thresholds": [0.8, 0.2]}, {"thresholds": [0.1, 2]},
                        {"thresholds": [np.nan]}, {"thresholds": [True]},
                        {"thresholds": np.linspace(0, 1, 202)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.evaluate(**options)

    def test_inputs_are_preserved_and_output_is_strict_json(self):
        before = self.labels.copy(), self.probabilities.copy()
        result = self.evaluate(threshold=np.float32(0.5), n_bins=np.int64(2))
        np.testing.assert_array_equal(before[0], self.labels)
        np.testing.assert_array_equal(before[1], self.probabilities)
        json.dumps(result, allow_nan=False)
        self.assertEqual(result["support"], {"rows": 4, "positives": 2, "negatives": 2, "positive_prevalence": 0.5})

    def test_row_permutation_does_not_change_measurements(self):
        first = self.evaluate(n_bins=2)
        second = evaluate_binary(self.labels[::-1], self.probabilities[::-1], n_bins=2)
        self.assertEqual(first, second)

    def test_fixed_metrics_match_phase1_without_changing_it(self):
        baseline = baseline_metrics(self.labels, self.probabilities)
        evaluated = self.evaluate()
        for name in ("precision", "recall", "f1", "confusion_matrix", "threshold"):
            self.assertEqual(evaluated["classification"][name], baseline[name])
        for name in ("average_precision", "pr_auc_trapezoidal"):
            self.assertEqual(evaluated["ranking"][name], baseline[name])


class PredictionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "raw.csv"
        fixture(raw)
        self.prepared, self.models = self.root / "prepared", self.root / "models"
        self.models.mkdir()
        manifest = prepare_dataset(raw, self.prepared, expected_sha256=fingerprint(raw))
        _, reference, _ = load_partitions(self.prepared)
        self.labels = reference["Churn"].to_numpy()
        self.summary = {"format_version": 1,
                        "preparation_manifest_sha256": fingerprint(self.prepared / "manifest.json"),
                        "partitions": manifest["files"], "models": {}}
        for name in BASELINE_NAMES:
            path = self.models / f"{name}_probabilities.npy"
            np.save(path, np.linspace(0.1, 0.9, len(reference)), allow_pickle=False)
            self.summary["models"][name] = {"probabilities_sha256": fingerprint(path)}
        self.summary_path = self.models / "summary.json"
        write_json(self.summary_path, self.summary)
        self.output = self.root / "reports" / "audit.json"

    def save_summary(self):
        self.summary_path.write_text(json.dumps(self.summary), encoding="utf-8")

    def audit(self):
        return audit_baseline_predictions(self.prepared, self.models, self.output)

    def test_integration_preserves_all_inputs_and_never_loads_joblib(self):
        before = {str(p): fingerprint(p) for d in (self.prepared, self.models) for p in d.iterdir()}
        with patch("joblib.load", side_effect=AssertionError("No deserialization allowed")):
            result = self.audit()
        self.assertEqual(result, json.loads(self.output.read_text()))
        self.assertFalse(result["model_deserialization_performed"])
        self.assertEqual(set(result["models"]), set(BASELINE_NAMES))
        self.assertEqual(len(result["inputs"]), 6)
        self.assertEqual(before, {str(p): fingerprint(p) for d in (self.prepared, self.models) for p in d.iterdir()})

    def test_corrupt_probability_bytes_stop_before_writing(self):
        path = self.models / "logistic_regression_probabilities.npy"
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_changed_reference_order_stops_even_with_new_manifest(self):
        path = self.prepared / "reference.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0]] + list(reversed(lines[1:]))) + "\n")
        manifest_path = self.prepared / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["reference.csv"]["sha256"] = fingerprint(path)
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_wrong_partition_binding_is_rejected(self):
        self.summary["partitions"]["reference.csv"]["rows"] += 1
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.audit()

    def test_malformed_summary_models_and_version_rejected(self):
        original = json.loads(json.dumps(self.summary))
        variants = [[], {**original, "format_version": True}, {**original, "format_version": 2},
                    {**original, "models": {}}, {**original, "models": {"../outside": {}}},
                    {**original, "models": {name: None for name in BASELINE_NAMES}}]
        for variant in variants:
            with self.subTest(summary=variant), self.assertRaises(ValueError):
                self.summary_path.write_text(json.dumps(variant))
                self.audit()
            self.assertFalse(self.output.exists())

    def test_bad_array_shape_values_and_object_pickle_are_rejected(self):
        path = self.models / "logistic_regression_probabilities.npy"
        for values in (np.array([0.1]), np.zeros((10, 1)), np.full(10, np.nan),
                       np.full(10, 2.0), np.array([{}] * 10, dtype=object)):
            with self.subTest(shape=values.shape):
                np.save(path, values, allow_pickle=True)
                self.summary["models"]["logistic_regression"]["probabilities_sha256"] = fingerprint(path)
                self.save_summary()
                with self.assertRaises(ValueError):
                    self.audit()
                self.assertFalse(self.output.exists())

    def test_empty_array_file_with_updated_hash_is_rejected_cleanly(self):
        path = self.models / "logistic_regression_probabilities.npy"
        path.write_bytes(b"")
        self.summary["models"]["logistic_regression"]["probabilities_sha256"] = fingerprint(path)
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "Invalid numerical probability file"):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_npz_disguised_as_npy_is_rejected(self):
        path = self.models / "logistic_regression_probabilities.npy"
        with path.open("wb") as stream:
            np.savez(stream, values=np.ones(10))
        self.summary["models"]["logistic_regression"]["probabilities_sha256"] = fingerprint(path)
        self.save_summary()
        with self.assertRaisesRegex(ValueError, "single numerical"):
            self.audit()

    def test_existing_output_is_preserved(self):
        self.audit()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual(before, self.output.read_bytes())

    def test_changed_input_during_evaluation_does_not_write_report(self):
        def mutate(*args, **kwargs):
            self.summary_path.write_text("changed during evaluation")
            return evaluate_binary(*args, **kwargs)
        with patch("modelaudit.model_audit.evaluate_binary", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.audit()
        self.assertFalse(self.output.exists())

    def test_development_command_and_overwrite_exit(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_models.py"
        command = [sys.executable, str(script), "--prepared", str(self.prepared),
                   "--models", str(self.models), "--output", str(self.output)]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("Evaluated 2", first.stdout)
        second = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn("Choose a new report", second.stderr)


if __name__ == "__main__":
    unittest.main()
