import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import ks_2samp

from modelaudit.artifacts import fingerprint
from modelaudit.drift import compare_snapshots
from modelaudit.drift_audit import audit_snapshot_files


class FeatureDriftTests(unittest.TestCase):
    def numeric(self, first, second, **options):
        return compare_snapshots(pd.DataFrame({"x": first}), pd.DataFrame({"x": second}),
                                 numeric=["x"], **options)["features"]["x"]

    def categorical(self, first, second, **options):
        return compare_snapshots(pd.DataFrame({"c": first}), pd.DataFrame({"c": second}),
                                 categorical=["c"], **options)["features"]["c"]

    def test_equal_distributions_with_different_sizes_have_zero_distance(self):
        self.assertEqual(self.numeric([1, 2], [1, 1, 2, 2])["ks_distance"], 0)
        self.assertEqual(self.categorical(["a", "b"], ["b", "a", "b", "a"])["total_variation"], 0)

    def test_numeric_ties_and_hand_calculated_distance(self):
        self.assertEqual(self.numeric([0, 0, 1, 2], [0, 1, 1, 1])["ks_distance"], 0.25)

    def test_disjoint_constants_have_unit_distance(self):
        self.assertEqual(self.numeric([1, 1], [2, 2, 2])["ks_distance"], 1)

    def test_numeric_statistic_agrees_with_independent_scipy_reference(self):
        rng = np.random.default_rng(42)
        for first, second in ((rng.normal(size=80), rng.normal(0.5, size=51)),
                              (rng.integers(0, 9, 70), rng.integers(0, 9, 30))):
            with self.subTest(rows=len(first)):
                expected = ks_2samp(first, second).statistic
                self.assertAlmostEqual(self.numeric(first, second)["ks_distance"], expected)

    def test_missingness_is_separate_from_observed_distribution(self):
        result = self.numeric([1, None, 2, None], [1, 2])
        self.assertEqual(result["ks_distance"], 0)
        self.assertEqual(result["missingness"]["fraction_difference"], -0.5)
        self.assertEqual(result["reference_observed"], 2)

    def test_all_missing_is_undefined_not_zero(self):
        for first, second in (([None], [1]), ([None], [np.nan]), ([1], [pd.NA])):
            with self.subTest(first=first):
                result = self.numeric(first, second)
                self.assertIsNone(result["ks_distance"])
                self.assertTrue(result["undefined_reason"])
                json.dumps(result, allow_nan=False)

    def test_categorical_union_and_unseen_counts(self):
        result = self.categorical(["a", "a", "b", "b"], ["b", "b", "c", "c"])
        self.assertEqual(result["total_variation"], 0.5)
        self.assertEqual(result["new_category_rows"], 2)
        self.assertEqual(result["new_category_fraction"], 0.5)
        self.assertEqual([r["value"] for r in result["categories"]], ["a", "b", "c"])

    def test_missing_category_does_not_collide_with_strings(self):
        result = self.categorical(["__MISSING__", "a"], [None, " ", "__MISSING__", "a"])
        self.assertEqual(result["new_category_rows"], 0)
        self.assertEqual(result["missingness"]["current_fraction"], 0.5)
        self.assertEqual(len(result["categories"]), 3)
        self.assertEqual(result["total_variation"], 0.5)

    def test_all_missing_categorical_is_measurable(self):
        result = self.categorical([None, ""], [pd.NA])
        self.assertEqual(result["total_variation"], 0)
        self.assertEqual(result["categories"][0]["value"], None)

    def test_unused_categories_are_not_added(self):
        first = pd.Series(pd.Categorical(["a", "a"], categories=["a", "unused"]))
        self.assertEqual(len(self.categorical(first, ["a"])["categories"]), 1)

    def test_support_uses_observed_numeric_counts_and_inclusive_boundary(self):
        result = self.numeric([1, 2, None], [1, None, None], minimum_rows=2)
        self.assertEqual(result["warnings"], [{"code": "low_rows", "snapshot": "current",
                                              "count": 1, "minimum": 2}])
        self.assertEqual(self.categorical(["a", None], ["a", None], minimum_rows=2)["warnings"], [])

    def test_category_limit_counts_union_and_missing(self):
        with self.assertRaisesRegex(ValueError, "maximum_categories"):
            self.categorical(["a", None], ["b"], maximum_categories=2)
        self.assertEqual(len(self.categorical(["a", None], ["b"], maximum_categories=3)["categories"]), 3)

    def test_distances_symmetric_and_missing_deltas_change_sign(self):
        a, b = [1, 2, None], [4, 4, 4, 5]
        first, second = self.numeric(a, b), self.numeric(b, a)
        self.assertEqual(first["ks_distance"], second["ks_distance"])
        self.assertEqual(first["missingness"]["fraction_difference"], -second["missingness"]["fraction_difference"])

    def test_numeric_rejects_bad_types_infinities_and_lossy_integers(self):
        for values in (["1"], [True], [np.inf], [-np.inf], [1j], [[1]], [2**53 + 1], [10**400]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.numeric(pd.Series(values, dtype=object), [1])

    def test_categorical_rejects_nonstring_nonmissing_values(self):
        for values in ([1], [True], [np.inf], [["a"]]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.categorical(values, ["a"])

    def test_unknown_missing_duplicate_overlapping_roles_rejected(self):
        frame = pd.DataFrame({"x": [1]})
        for options in ({}, {"numeric": "x"}, {"numeric": ["x", "x"]},
                        {"numeric": ["x"], "categorical": ["x"]}, {"numeric": [1]},
                        {"numeric": ["z"]}, {"target": "x"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                compare_snapshots(frame, frame, **options)
        with self.assertRaises(ValueError):
            compare_snapshots(frame.assign(extra=1), frame, numeric=["x"])

    def test_invalid_frames_rejected(self):
        for frame in (pd.DataFrame({"x": []}), pd.DataFrame([[1, 2]], columns=["x", "x"])):
            with self.assertRaises(ValueError):
                compare_snapshots(frame, frame, numeric=["x"])
        with self.assertRaises(TypeError):
            compare_snapshots([1], [2], numeric=["x"])

    def test_invalid_policy_rejected(self):
        for options in ({"minimum_rows": True}, {"minimum_class_rows": 0},
                        {"minimum_predicted_positives": 1.2}, {"maximum_categories": 1001},
                        {"n_bins": 101}, {"n_bins": False}, {"threshold": np.nan}, {"threshold": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.numeric([1], [1], **options)

    def test_inputs_unchanged_and_json_strict(self):
        first = pd.DataFrame({"x": [1.0, np.nan], "c": ["a", None]}, index=[7, 4])
        second = pd.DataFrame({"x": [2.0, 3.0], "c": ["b", "a"]}, index=[4, 7])
        before = first.copy(deep=True), second.copy(deep=True)
        result = compare_snapshots(first, second, numeric=["x"], categorical=["c"],
                                   minimum_rows=np.int64(2), n_bins=np.int64(10))
        assert_frame_equal(before[0], first)
        assert_frame_equal(before[1], second)
        json.dumps(result, allow_nan=False)


class PredictionAndPerformanceTests(unittest.TestCase):
    def setUp(self):
        self.first = pd.DataFrame({"p": [0.1, 0.9, 0.2, 0.8], "y": [0, 1, 0, 1]})
        self.second = pd.DataFrame({"p": [0.9, 0.1, 0.8, 0.2], "y": [0, 1, 0, 1]})

    def compare(self, **options):
        settings = dict(probability="p", target="y", minimum_rows=4,
                        minimum_class_rows=2, minimum_predicted_positives=2)
        settings.update(options)
        return compare_snapshots(self.first, self.second, **settings)

    def test_unchanged_prediction_distribution_can_hide_performance_loss(self):
        result = self.compare()
        self.assertEqual(result["predictions"]["ks_distance"], 0)
        self.assertEqual(result["predictions"]["histogram"]["total_variation"], 0)
        self.assertEqual(result["performance"]["difference"]["f1"], -1)
        self.assertAlmostEqual(result["performance"]["difference"]["brier_score"], 0.7)
        self.assertEqual(result["performance"]["metric_direction"]["brier_score"], "lower_is_better")

    def test_prediction_disjoint_endpoints_and_threshold_equality(self):
        self.first = pd.DataFrame({"p": [0., 0.]})
        self.second = pd.DataFrame({"p": [0.5, 1.]})
        result = self.compare()["predictions"]
        self.assertEqual(result["ks_distance"], 1)
        self.assertEqual(result["mean_probability_difference"], 0.75)
        self.assertEqual(result["predicted_positive_fraction_difference"], 1)
        self.assertEqual(result["histogram"]["total_variation"], 1)
        self.assertEqual(result["histogram"]["bins"][-1]["current_count"], 1)

    def test_decimal_bins_and_neighbors(self):
        self.first = pd.DataFrame({"p": [np.nextafter(.3, 0), .3, np.nextafter(.3, 1)]})
        self.second = self.first.copy()
        bins = self.compare()["predictions"]["histogram"]["bins"]
        self.assertEqual(bins[2]["reference_count"], 1)
        self.assertEqual(bins[3]["reference_count"], 2)
        self.assertEqual(sum(b["current_count"] for b in bins), 3)

    def test_unlabeled_current_skips_performance_only(self):
        self.second = self.second.drop(columns="y")
        result = self.compare()
        self.assertEqual(result["predictions"]["status"], "measured")
        self.assertEqual(result["performance"]["status"], "skipped")
        self.assertTrue(result["performance"]["reason"])

    def test_features_without_predictions_skip_both_prediction_sections(self):
        result = compare_snapshots(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]}), numeric=["x"])
        self.assertEqual(result["predictions"]["status"], "skipped")
        self.assertEqual(result["performance"]["status"], "skipped")

    def test_single_class_preserves_defined_metrics_but_excludes_ranking_delta(self):
        self.second["y"] = 0
        result = self.compare(minimum_rows=1, minimum_class_rows=1)["performance"]
        self.assertIsNone(result["difference"]["roc_auc"])
        self.assertIsNotNone(result["difference"]["accuracy"])
        self.assertEqual(result["positive_prevalence_difference"], -0.5)
        self.assertIn("current", result["difference_undefined_reasons"]["roc_auc"])

    def test_denominator_support_and_equal_threshold(self):
        eligible = self.compare()["performance"]
        self.assertEqual(eligible["difference"]["precision"], -1)
        result = self.compare(minimum_predicted_positives=3)["performance"]
        self.assertIsNone(result["difference"]["precision"])
        self.assertIsNotNone(result["difference"]["recall"])
        self.assertEqual(set(result["difference_undefined_reasons"]["precision"]), {"reference", "current"})

    def test_low_rows_withholds_all_deltas_and_retains_scores(self):
        result = self.compare(minimum_rows=5)["performance"]
        self.assertTrue(all(value is None for value in result["difference"].values()))
        self.assertEqual(result["reference"]["metrics"]["accuracy"], 1)

    def test_no_predicted_positives_is_not_zero_precision(self):
        self.second["p"] = 0.1
        result = self.compare()["performance"]
        self.assertIsNone(result["current"]["metrics"]["precision"])
        self.assertIsNone(result["difference"]["precision"])
        self.assertEqual(result["current"]["metrics"]["recall"], 0)

    def test_invalid_probabilities_and_partial_or_invalid_labels_rejected(self):
        for column, values in (("p", [np.nan, .1, .2, .3]), ("p", [1.1, .1, .2, .3]),
                               ("p", [True]*4), ("p", ["0.5"]*4),
                               ("y", [0, 1, 0, np.nan]), ("y", [0, 2, 0, 1]),
                               ("y", [False]*4)):
            with self.subTest(column=column, values=values):
                self.setUp()
                self.second[column] = values
                with self.assertRaises(ValueError):
                    self.compare()

    def test_invalid_present_labels_rejected_even_if_other_side_unlabeled(self):
        self.first = self.first.drop(columns="y")
        self.second["y"] = np.nan
        with self.assertRaises(ValueError):
            self.compare()

    def test_joint_row_permutation_preserves_comparison_with_different_indices(self):
        first = self.compare()
        self.first = self.first.iloc[::-1].set_axis([100, 1, 1, 200])
        self.second = self.second.iloc[::-1].set_axis([9, 7, 4, 1])
        # Floating-point reductions can differ in their last bit after reordering.
        def equivalent(left, right):
            if isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    equivalent(left[key], right[key])
            elif isinstance(left, list):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    equivalent(a, b)
            elif isinstance(left, float):
                self.assertAlmostEqual(left, right, places=14)
            else:
                self.assertEqual(left, right)
        equivalent(first, self.compare())


class SnapshotFileTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.first, self.second, self.output = [self.root / name for name in ("first.csv", "second.csv", "report.json")]
        for path in (self.first, self.second):
            path.write_text('x,c,p,y\n1,NA,0.1,0\n2,001,0.9,1\n3,,0.2,0\n4,"a,b",0.8,1\n', encoding="utf-8")

    def audit(self, **options):
        return audit_snapshot_files(self.first, self.second, self.output, numeric=["x"],
                                    categorical=["c"], probability="p", target="y", **options)

    def test_roundtrip_preserves_inputs_text_categories_and_never_loads_models(self):
        before = [fingerprint(p) for p in (self.first, self.second)]
        with patch("joblib.load", side_effect=AssertionError("No models")):
            result = self.audit(minimum_rows=1, minimum_class_rows=1, minimum_predicted_positives=1)
        self.assertEqual(result, json.loads(self.output.read_text()))
        self.assertEqual(before, [fingerprint(p) for p in (self.first, self.second)])
        self.assertEqual([r["value"] for r in result["comparison"]["features"]["c"]["categories"]],
                         ["001", "NA", "a,b", None])
        self.assertTrue(all(v == 0 for v in result["comparison"]["performance"]["difference"].values()))
        self.assertEqual(result["inputs"], dict(reference=before[0], current=before[1]))

    def test_existing_report_and_input_output_alias_are_preserved(self):
        self.audit()
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual(before, self.output.read_bytes())
        self.output = self.first
        before = self.first.read_bytes()
        with self.assertRaises(FileExistsError):
            self.audit()
        self.assertEqual(before, self.first.read_bytes())

    def test_changed_input_during_comparison_prevents_report(self):
        def mutate(*args, **kwargs):
            self.second.write_text("changed", encoding="utf-8")
            return compare_snapshots(*args, **kwargs)
        with patch("modelaudit.drift_audit.compare_snapshots", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "changed during"):
                self.audit()
        self.assertFalse(self.output.exists())

    def test_duplicate_headers_ragged_empty_and_extra_fields_fail(self):
        for content in ('x,c,p,p\n1,a,.1,.2\n', 'x,c,p,y\n1,a,.1\n', '', 'x,c,p,y\n',
                        'x,c,p,y,extra\n1,a,.1,0,z\n'):
            with self.subTest(content=content):
                self.second.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.audit()
                self.assertFalse(self.output.exists())

    def test_bad_numeric_and_partial_probability_csv_fail(self):
        for value in ('oops', 'inf', '9007199254740993'):
            with self.subTest(value=value):
                self.second.write_text(f'x,c,p,y\n{value},a,.1,0\n', encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.audit()
        self.second.write_text('x,c,p,y\n1,a,,0\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            self.audit()
        self.assertFalse(self.output.exists())

    def test_csv_missing_target_and_unequal_lengths_work(self):
        self.second.write_text('x,c,p\n1,a,.1\n', encoding="utf-8")
        result = self.audit()["comparison"]
        self.assertEqual((result["reference_rows"], result["current_rows"]), (4, 1))
        self.assertEqual(result["performance"]["status"], "skipped")

    def test_repeat_reports_identical_except_context(self):
        first = self.audit()
        self.output = self.root / "repeat.json"
        second = self.audit()
        first.pop("context")
        second.pop("context")
        self.assertEqual(first, second)

    def test_command_success_overwrite_and_invalid_schema_exit(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_drift.py"
        command = [sys.executable, str(script), "--reference", str(self.first), "--current", str(self.second),
                   "--output", str(self.output), "--numeric", "x", "--categorical", "c",
                   "--probability", "p", "--target", "y"]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("Compared 4 reference and 4 current rows", first.stdout)
        self.assertEqual(subprocess.run(command, text=True, capture_output=True).returncode, 2)
        invalid = self.root / "invalid.json"
        result = subprocess.run(command + ["--output", str(invalid), "--target", "wrong"], text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(invalid.exists())


if __name__ == "__main__":
    unittest.main()
