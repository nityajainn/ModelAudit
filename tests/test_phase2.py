import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from modelaudit.artifacts import fingerprint
from modelaudit.data_audit import audit_prepared
from modelaudit.data_checks import CheckPolicy, DataSchema, audit_data
from modelaudit.dataset import prepare_dataset
from test_phase1 import fixture

SCHEMA = DataSchema(("amount",), ("group",), "target", "id")


def frames():
    train = pd.DataFrame({"id": [f"train-{i}" for i in range(40)],
                          "amount": np.arange(40, dtype=float),
                          "group": ["a", "a", "b", "b"] * 10,
                          "target": [0, 1] * 20})
    reference = train.copy(deep=True)
    reference["id"] = [f"ref-{i}" for i in range(40)]
    reference["amount"] += 100
    return train, reference


def finding(result, check, dataset="train", column=None):
    return next(item for item in result["findings"] if item["check"] == check
                and item["dataset"] == dataset
                and (column is None or item["evidence"].get("column") == column))


class DataChecksTests(unittest.TestCase):
    def setUp(self):
        self.train, self.reference = frames()

    def audit(self, **kwargs):
        return audit_data(self.train, self.reference, SCHEMA, **kwargs)

    def test_clean_inputs_pass_without_mutation_and_serialize_strictly(self):
        saved = self.train.copy(deep=True), self.reference.copy(deep=True)
        result = self.audit()
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["counts"]["skipped"], 0)
        self.assertEqual(sum(result["counts"].values()), len(result["findings"]))
        json.dumps(result, allow_nan=False)
        assert_frame_equal(self.train, saved[0])
        assert_frame_equal(self.reference, saved[1])
        self.assertNotIn("train-0", json.dumps(result))

    def test_missing_and_extra_columns_fail_and_dependent_checks_skip(self):
        self.train = self.train.drop(columns="amount").assign(unexpected=1)
        result = self.audit()
        schema = finding(result, "schema")
        self.assertEqual(schema["status"], "fail")
        self.assertEqual(schema["evidence"]["missing_columns"], ["amount"])
        self.assertEqual(schema["evidence"]["unexpected_columns"], ["unexpected"])
        self.assertEqual(finding(result, "predictor_overlap", "cross_split")["status"], "skipped")

    def test_duplicate_headers_fail_without_crashing_other_partition(self):
        self.train.columns = ["id", "amount", "group", "group"]
        result = self.audit()
        self.assertEqual(finding(result, "schema")["status"], "fail")
        self.assertEqual(finding(result, "data_checks")["status"], "skipped")
        self.assertEqual(finding(result, "schema", "reference")["status"], "pass")

    def test_numeric_strings_bool_complex_and_category_numbers_fail(self):
        variants = [("amount", ["1"] * 40), ("amount", [True] * 40),
                    ("amount", [1j] * 40), ("group", [1] * 40),
                    ("id", [12] * 40), ("group", [["a"]] * 40)]
        for column, values in variants:
            with self.subTest(column=column, value=values[0]):
                self.train, self.reference = frames()
                self.train[column] = values
                result = self.audit()
                self.assertIn(column, finding(result, "schema")["evidence"]["invalid_type_columns"])

    def test_infinity_fails_and_is_not_exported_as_nonstandard_json(self):
        self.train.loc[0, "amount"] = np.inf
        self.train.loc[1, "amount"] = -np.inf
        result = self.audit()
        self.assertEqual(finding(result, "schema")["evidence"]["nonfinite_counts"], {"amount": 2})
        self.assertEqual(finding(result, "leakage_heuristic", column="amount")["status"], "skipped")
        json.dumps(result, allow_nan=False)

    def test_missing_fraction_threshold_equality_passes_above_warns(self):
        self.train.loc[:1, "amount"] = np.nan
        self.assertEqual(finding(self.audit(), "missingness", column="amount")["status"], "pass")
        self.train.loc[2, "amount"] = np.nan
        result = finding(self.audit(), "missingness", column="amount")
        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["evidence"]["count"], 3)
        self.assertEqual(result["evidence"]["fraction"], 3 / 40)

    def test_blank_categories_count_as_missing_without_changing_input(self):
        self.train.loc[:2, "group"] = "  "
        saved = self.train.copy(deep=True)
        result = self.audit()
        self.assertEqual(finding(result, "missingness", column="group")["evidence"]["count"], 3)
        assert_frame_equal(self.train, saved)

    def test_missing_id_and_target_fail_even_below_missingness_threshold(self):
        self.train.loc[0, "id"] = " "
        self.train["target"] = self.train["target"].astype("Int64")
        self.train.loc[1, "target"] = pd.NA
        result = self.audit()
        for column in ("id", "target"):
            self.assertEqual(finding(result, "missingness", column=column)["status"], "fail")
        self.assertEqual(finding(result, "target")["status"], "fail")

    def test_invalid_and_single_class_targets_fail(self):
        for labels in ([0] * 40, [0, 2] * 20, [True, False] * 20):
            with self.subTest(labels=labels[:2]):
                self.train["target"] = labels
                result = self.audit()
                self.assertEqual(result["status"], "fail")
                self.assertEqual(finding(result, "leakage_heuristic", column="amount")["status"], "skipped")

    def test_exact_and_predictor_duplicates_count_rows_beyond_first(self):
        self.train = pd.concat([self.train, self.train.iloc[[0, 0]]], ignore_index=True)
        result = self.audit()
        for check in ("duplicate_rows", "duplicate_predictors", "duplicate_identifiers"):
            self.assertEqual(finding(result, check)["evidence"]["count"], 2)
        self.assertEqual(finding(result, "duplicate_identifiers")["status"], "fail")

    def test_predictor_duplicates_ignore_target_and_identifier(self):
        self.train.loc[1, ["amount", "group"]] = self.train.loc[0, ["amount", "group"]]
        result = self.audit()
        self.assertEqual(finding(result, "duplicate_predictors")["evidence"]["count"], 1)
        self.assertEqual(finding(result, "duplicate_rows")["evidence"]["count"], 0)

    def test_identifier_overlap_counts_reference_rows_and_distinct_ids(self):
        self.reference.loc[:1, "id"] = self.train.loc[0, "id"]
        result = finding(self.audit(), "identifier_overlap", "cross_split")
        self.assertEqual(result["status"], "fail")
        self.assertEqual(result["evidence"], {"reference_rows": 2, "distinct_identifiers": 1})

    def test_missing_identifiers_do_not_match_across_splits(self):
        self.train.loc[0, "id"] = None
        self.reference.loc[0, "id"] = None
        result = self.audit()
        self.assertEqual(finding(result, "identifier_overlap", "cross_split")["evidence"]["reference_rows"], 0)

    def test_predictor_overlap_counts_each_reference_row_once(self):
        self.reference.loc[:1, ["amount", "group"]] = [0, "a"]
        self.train = pd.concat([self.train, self.train.iloc[[0, 0]]], ignore_index=True)
        result = finding(self.audit(), "predictor_overlap", "cross_split")
        self.assertEqual(result["status"], "warn")
        self.assertEqual(result["evidence"]["reference_rows"], 2)
        self.assertEqual(result["evidence"]["reference_fraction"], 0.05)

    def test_overlap_matches_missing_predictors_and_handles_categorical_dtypes(self):
        self.train.loc[0, "amount"] = np.nan
        self.reference.loc[0, "amount"] = np.nan
        self.train["group"] = self.train["group"].astype("category")
        result = self.audit()
        self.assertEqual(finding(result, "predictor_overlap", "cross_split")["evidence"]["reference_rows"], 1)

    def test_all_missing_category_has_warning_and_skipped_heuristic(self):
        self.train["group"] = np.nan
        result = self.audit()
        self.assertEqual(finding(result, "missingness", column="group")["status"], "warn")
        self.assertEqual(finding(result, "leakage_heuristic", column="group")["status"], "skipped")
        json.dumps(result, allow_nan=False)

    def test_target_copy_inverse_and_affine_proxy_flagged(self):
        for values, expected in ((self.train.target, "target_copy"),
                                 (1 - self.train.target, "target_inverse"),
                                 (self.train.target * 10 + 4, "high_target_correlation")):
            with self.subTest(expected=expected):
                self.train["amount"] = values
                result = finding(self.audit(), "leakage_heuristic", column="amount")
                self.assertEqual(result["status"], "warn")
                self.assertIn(expected, result["evidence"]["signals"])

    def test_exact_target_copy_warns_even_with_small_sample(self):
        self.train = self.train.iloc[:2].copy()
        self.train["amount"] = self.train.target
        result = finding(self.audit(), "leakage_heuristic", column="amount")
        self.assertEqual(result["status"], "warn")
        self.assertIn("target_copy", result["evidence"]["signals"])

    def test_supported_category_mapping_warns_but_unique_values_do_not(self):
        self.train["group"] = ["negative", "positive"] * 20
        result = finding(self.audit(), "leakage_heuristic", column="group")
        self.assertIn("category_determines_target", result["evidence"]["signals"])
        self.train["group"] = self.train.id
        result = finding(self.audit(), "leakage_heuristic", column="group")
        self.assertEqual(result["evidence"]["signals"], [])

    def test_small_or_constant_inputs_skip_association(self):
        self.train = self.train.iloc[:10].copy()
        self.assertEqual(finding(self.audit(), "leakage_heuristic", column="amount")["status"], "skipped")
        self.train, self.reference = frames()
        self.train["amount"] = 1
        self.assertEqual(finding(self.audit(), "leakage_heuristic", column="amount")["status"], "skipped")

    def test_empty_table_fails_and_undefined_fraction_is_null(self):
        self.train = self.train.iloc[:0].copy()
        result = self.audit()
        self.assertEqual(finding(result, "row_count")["status"], "fail")
        self.assertIsNone(finding(result, "missingness", column="amount")["evidence"]["fraction"])
        self.assertEqual(finding(result, "predictor_overlap", "cross_split")["status"], "skipped")
        json.dumps(result, allow_nan=False)

    def test_column_order_row_order_and_index_do_not_change_findings(self):
        first = self.audit()
        self.train = self.train.iloc[::-1, ::-1]
        self.train.index = [0] * len(self.train)
        second = self.audit()
        self.assertEqual(first, second)

    def test_nullable_numeric_and_optional_identifier(self):
        self.train["amount"] = self.train.amount.astype("Float64")
        self.train["target"] = self.train.target.astype("Int64")
        self.train["id"] = self.train.id.astype("string")
        self.assertEqual(self.audit()["status"], "pass")
        schema = DataSchema(("amount",), ("group",), "target")
        result = audit_data(self.train.drop(columns="id"), self.reference.drop(columns="id"), schema)
        self.assertEqual(finding(result, "identifier_overlap", "cross_split")["status"], "skipped")

    def test_large_finite_numbers_do_not_overflow_correlation(self):
        self.train["amount"] = np.where(self.train.target == 1, 1e308, -1e308)
        result = finding(self.audit(), "leakage_heuristic", column="amount")
        self.assertAlmostEqual(result["evidence"]["correlation"], 1)

    def test_internal_merge_marker_is_not_a_reserved_feature_name(self):
        for name in ("_match", "_left_indicator", "_right_indicator"):
            with self.subTest(name=name):
                schema = DataSchema((name,), ("group",), "target", "id")
                result = audit_data(self.train.rename(columns={"amount": name}),
                                    self.reference.rename(columns={"amount": name}), schema)
                self.assertEqual(result["status"], "pass")

    def test_large_integers_losing_float_precision_skip_correlation(self):
        self.train["amount"] = np.array([2**60 + i for i in range(40)], dtype=np.int64)
        result = finding(self.audit(), "leakage_heuristic", column="amount")
        self.assertEqual(result["status"], "skipped")
        self.assertIsNone(result["evidence"]["correlation"])

    def test_numpy_policy_values_still_produce_standard_json(self):
        result = self.audit(policy=CheckPolicy(missing_fraction=np.float32(0.1)))
        json.dumps(result, allow_nan=False)

    def test_sparse_categories_are_explicitly_skipped(self):
        self.train["group"] = ["rare"] + ["a", "b", "a"] * 13
        result = finding(self.audit(), "leakage_heuristic", column="group")
        self.assertEqual(result["status"], "skipped")

    def test_invalid_schema_policy_and_api_arguments_rejected(self):
        for args in (((), (), "target"), (("x",), ("x",), "target"),
                     (("x",), (), "x"), ("x", (), "target"), (("",), (), "target")):
            with self.subTest(schema=args), self.assertRaises(ValueError):
                DataSchema(*args)
        for args in ({"missing_fraction": np.nan}, {"missing_fraction": -1},
                     {"missing_fraction": True}, {"correlation_threshold": 0},
                     {"minimum_pairs": True}, {"maximum_categories": 1},
                     {"minimum_category_rows": 1.5}):
            with self.subTest(policy=args), self.assertRaises(ValueError):
                CheckPolicy(**args)
        with self.assertRaises(TypeError):
            audit_data([], self.reference, SCHEMA)
        with self.assertRaises(TypeError):
            self.audit(policy={})


class PreparedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.csv"
        fixture(self.raw)
        self.prepared = self.root / "prepared"
        prepare_dataset(self.raw, self.prepared, expected_sha256=fingerprint(self.raw))
        self.output = self.root / "reports" / "audit.json"

    def test_integration_writes_provenance_and_preserves_input_bytes(self):
        before = {p.name: fingerprint(p) for p in self.prepared.iterdir()}
        result = audit_prepared(self.prepared, self.output)
        self.assertEqual(result, json.loads(self.output.read_text()))
        self.assertEqual(result["inputs"]["train.csv"], before["train.csv"])
        self.assertEqual(before, {p.name: fingerprint(p) for p in self.prepared.iterdir()})
        self.assertIn("created_at", result["context"])

    def test_existing_report_remains_unchanged(self):
        audit_prepared(self.prepared, self.output)
        before = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            audit_prepared(self.prepared, self.output)
        self.assertEqual(before, self.output.read_bytes())

    def test_corrupt_prepared_file_does_not_create_report(self):
        with (self.prepared / "train.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            audit_prepared(self.prepared, self.output)
        self.assertFalse(self.output.exists())

    def test_development_command_success_and_overwrite_error(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "audit_data.py"
        command = [sys.executable, str(script), "--prepared", str(self.prepared),
                   "--output", str(self.output)]
        first = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertIn("Data audit:", first.stdout)
        second = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(second.returncode, 2)
        self.assertIn("Choose a new report", second.stderr)


if __name__ == "__main__":
    unittest.main()
