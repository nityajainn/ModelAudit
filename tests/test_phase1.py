import csv
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from modelaudit.artifacts import fingerprint, write_json
from modelaudit.baselines import baseline_metrics, build_pipeline, train_baselines
from modelaudit.dataset import (
    CATEGORICAL, COLUMNS, FEATURES, IDENTIFIER, MISSING_CATEGORY, NUMERIC, TARGET,
    download_source, load_partitions, prepare_dataset, read_table,
)


def fixture(path, count=50):
    rows = []
    for index in range(count):
        row = {column: f"category-{index % 3}" for column in CATEGORICAL}
        row.update(customerID=f"customer-{index:03}", tenure=str(index % 36),
                   MonthlyCharges=str(20 + index / 3), TotalCharges=str(10 * index),
                   Churn="Yes" if index % 2 else "No")
        rows.append(row)
    rows[0]["TotalCharges"] = " "
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "source.csv"
        fixture(self.raw)

    def prepare(self, name="prepared", **kwargs):
        return prepare_dataset(self.raw, self.root / name,
                               expected_sha256=fingerprint(self.raw), **kwargs)

    def mutate_cell(self, row_index, column, value):
        with self.raw.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.reader(stream))
        rows[row_index + 1][rows[0].index(column)] = value
        with self.raw.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    def test_parsing_retains_missing_numbers_and_categorical_strings(self):
        self.mutate_cell(1, "SeniorCitizen", "0")
        data = read_table(self.raw)
        self.assertTrue(np.isnan(data.loc[0, "TotalCharges"]))
        self.assertEqual(data.loc[1, "SeniorCitizen"], "0")
        self.assertEqual(set(data[TARGET]), {0, 1})

    def test_header_duplicates_missing_and_extra_columns_rejected(self):
        original = self.raw.read_text()
        header, rest = original.split("\n", 1)
        for candidate in (header.replace("tenure", "Churn"),
                          header.replace("tenure,", ""), header + ",unexpected"):
            with self.subTest(header=candidate):
                self.raw.write_text(candidate + "\n" + rest)
                with self.assertRaises(ValueError):
                    read_table(self.raw)

    def test_ragged_record_is_rejected(self):
        with self.raw.open("a") as stream:
            stream.write("too,few,cells\n")
        with self.assertRaises(ValueError):
            read_table(self.raw)

    def test_bad_ids_targets_and_reserved_category_are_rejected(self):
        cases = [(IDENTIFIER, ""), (IDENTIFIER, "customer-001"), (TARGET, "maybe"),
                 ("Contract", " "), ("gender", MISSING_CATEGORY)]
        for column, value in cases:
            with self.subTest(column=column, value=value):
                fixture(self.raw)
                self.mutate_cell(0, column, value)
                with self.assertRaises(ValueError):
                    read_table(self.raw)

    def test_invalid_nonblank_numbers_rejected(self):
        for value in ("oops", "inf", "-inf", "NaN"):
            with self.subTest(value=value):
                fixture(self.raw)
                self.mutate_cell(0, "TotalCharges", value)
                with self.assertRaises(ValueError):
                    read_table(self.raw)

    def test_requires_both_classes(self):
        self.raw.write_text(self.raw.read_text().replace("Yes", "No"))
        with self.assertRaises(ValueError):
            read_table(self.raw)

    def test_stratified_split_is_disjoint_complete_and_repeatable(self):
        before = self.raw.read_bytes()
        first = self.prepare()
        second = self.prepare("repeat")
        train, reference, _ = load_partitions(self.root / "prepared")
        self.assertEqual((len(train), len(reference)), (40, 10))
        self.assertEqual((int(train[TARGET].sum()), int(reference[TARGET].sum())), (20, 5))
        self.assertFalse(set(train[IDENTIFIER]) & set(reference[IDENTIFIER]))
        self.assertEqual(set(train[IDENTIFIER]) | set(reference[IDENTIFIER]),
                         set(read_table(self.raw)[IDENTIFIER]))
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(before, self.raw.read_bytes())
        self.assertIsNone(first["source"]["url"])
        self.assertEqual(train[IDENTIFIER].tolist(), sorted(train[IDENTIFIER]))

    def test_split_does_not_depend_on_input_row_order(self):
        first = self.prepare()
        lines = self.raw.read_text().splitlines()
        self.raw.write_text("\n".join([lines[0]] + list(reversed(lines[1:]))) + "\n")
        second = self.prepare("shuffled")
        self.assertEqual(first["files"], second["files"])

    def test_wrong_source_hash_leaves_no_output(self):
        with self.assertRaises(ValueError):
            prepare_dataset(self.raw, self.root / "blocked")
        self.assertFalse((self.root / "blocked").exists())

    def test_existing_preparation_is_not_overwritten(self):
        self.prepare()
        before = {p.name: p.read_bytes() for p in (self.root / "prepared").iterdir()}
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, {p.name: p.read_bytes() for p in
                                 (self.root / "prepared").iterdir()})

    def test_bad_split_parameters_fail_before_writing(self):
        for arguments in ({"seed": True}, {"seed": -1}, {"reference_fraction": 1},
                          {"reference_fraction": 0}, {"reference_fraction": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.prepare(**arguments)
        self.assertFalse((self.root / "prepared").exists())

    def test_modified_partition_is_rejected(self):
        self.prepare()
        with (self.root / "prepared/train.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Checksum"):
            load_partitions(self.root / "prepared")

    def test_identifier_overlap_is_rejected_even_with_updated_hashes(self):
        self.prepare()
        directory = self.root / "prepared"
        train, reference, _ = load_partitions(directory)
        reference.loc[0, IDENTIFIER] = train.loc[0, IDENTIFIER]
        reference.to_csv(directory / "reference.csv", index=False)
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["files"]["reference.csv"]["sha256"] = fingerprint(directory / "reference.csv")
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "overlap"):
            load_partitions(directory)

    def test_incorrect_manifest_counts_are_rejected(self):
        self.prepare()
        manifest_path = self.root / "prepared/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["files"]["train.csv"]["rows"] += 1
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "counts"):
            load_partitions(manifest_path.parent)


class AcquisitionTests(unittest.TestCase):
    def test_checked_download_and_reuse_do_not_overwrite(self):
        content = b"fixed-source"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "raw.csv"
            with patch("modelaudit.dataset.SOURCE_SHA256", sha256(content).hexdigest()):
                with patch("modelaudit.dataset.urlopen", return_value=io.BytesIO(content)):
                    download_source(target)
                with patch("modelaudit.dataset.urlopen") as request:
                    download_source(target)
                    request.assert_not_called()
                self.assertEqual(target.read_bytes(), content)

    def test_bad_download_is_not_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "raw.csv"
            with patch("modelaudit.dataset.urlopen", return_value=io.BytesIO(b"wrong")):
                with self.assertRaises(ValueError):
                    download_source(target)
            self.assertFalse(target.exists())

    def test_existing_wrong_source_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "raw.csv"
            target.write_bytes(b"preserve me")
            with self.assertRaises(ValueError):
                download_source(target)
            self.assertEqual(target.read_bytes(), b"preserve me")


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        result = baseline_metrics([0, 1, 0, 1], [0.1, 0.8, 0.7, 0.4])
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(result[metric], 0.5)
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["pr_auc_trapezoidal"], 19 / 24)

    def test_no_predicted_positive_preserves_undefined_precision(self):
        result = baseline_metrics([0, 1], [0.1, 0.2])
        self.assertIsNone(result["precision"])
        self.assertEqual(result["recall"], 0)
        self.assertEqual(result["f1"], 0)
        self.assertIsNotNone(result["precision_undefined_reason"])

    def test_threshold_equality_is_positive(self):
        self.assertEqual(baseline_metrics([0, 1], [0.1, 0.5])["f1"], 1)

    def test_invalid_metric_inputs_rejected(self):
        for labels, probabilities in (([0], [0.5]), ([0, 1], [0.1]),
                                      ([0, 1], [0.1, np.nan]), ([0, 1], [0.1, 2]),
                                      ([[0, 1]], [[0.1, 0.5]])):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                baseline_metrics(labels, probabilities)

    def test_json_is_strict_and_non_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                write_json(path, {"value": np.nan})
            self.assertFalse(path.exists())
            write_json(path, {"value": None})
            with self.assertRaises(FileExistsError):
                write_json(path, {"value": 1})
            self.assertEqual(json.loads(path.read_text()), {"value": None})


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.csv"
        fixture(self.raw)
        self.prepared = self.root / "prepared"
        prepare_dataset(self.raw, self.prepared, expected_sha256=fingerprint(self.raw))

    def test_pipeline_fits_preprocessing_only_on_training(self):
        train, reference, _ = load_partitions(self.prepared)
        pipeline = build_pipeline("logistic_regression")
        pipeline.fit(train[FEATURES], train[TARGET])
        numeric = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
        np.testing.assert_allclose(numeric.named_steps["impute"].statistics_,
                                   train[NUMERIC].median().to_numpy())
        before = numeric.named_steps["scale"].mean_.copy()
        reference.loc[:, "MonthlyCharges"] = 1_000_000
        reference.loc[:, "gender"] = "reference-only-category"
        predictions = pipeline.predict_proba(reference[FEATURES])
        self.assertTrue(np.isfinite(predictions).all())
        np.testing.assert_array_equal(before, numeric.named_steps["scale"].mean_)
        self.assertNotIn(IDENTIFIER, pipeline.feature_names_in_)
        self.assertNotIn(TARGET, pipeline.feature_names_in_)

    def test_training_repeatability_serialization_and_input_preservation(self):
        before = {p.name: fingerprint(p) for p in self.prepared.iterdir()}
        first = train_baselines(self.prepared, self.root / "first")
        second = train_baselines(self.prepared, self.root / "second")
        _, reference, _ = load_partitions(self.prepared)
        for name in first["models"]:
            self.assertEqual(first["models"][name]["metrics"], second["models"][name]["metrics"])
            self.assertTrue(first["models"][name]["serialization_verified"])
            a = np.load(self.root / "first" / f"{name}_probabilities.npy", allow_pickle=False)
            b = np.load(self.root / "second" / f"{name}_probabilities.npy", allow_pickle=False)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
            model = joblib.load(self.root / "first" / f"{name}.joblib")
            np.testing.assert_allclose(model.predict_proba(reference[FEATURES])[:, 1],
                                       a, rtol=0, atol=1e-12)
        self.assertEqual(before, {p.name: fingerprint(p) for p in self.prepared.iterdir()})

    def test_training_rejects_corrupted_input_before_creating_outputs(self):
        (self.prepared / "train.csv").write_text("corrupted")
        with self.assertRaises(ValueError):
            train_baselines(self.prepared, self.root / "output")
        self.assertFalse((self.root / "output").exists())

    def test_training_refuses_existing_outputs(self):
        output = self.root / "output"
        output.mkdir()
        (output / "keep").write_text("unchanged")
        with self.assertRaises(FileExistsError):
            train_baselines(self.prepared, output)
        self.assertEqual((output / "keep").read_text(), "unchanged")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            build_pipeline("unknown")


if __name__ == "__main__":
    unittest.main()
