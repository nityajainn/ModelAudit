"""Pinned Telco acquisition, strict parsing, and reproducible preparation."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .artifacts import fingerprint, run_context, write_json

SOURCE_REVISION = "d5371f5d83a446ad5673cbcca3b814b926491f8a"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    + SOURCE_REVISION + "/data/Telco-Customer-Churn.csv"
)
SOURCE_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
IDENTIFIER = "customerID"
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
COLUMNS = [IDENTIFIER] + FEATURES + [TARGET]
MISSING_CATEGORY = "__MODELAUDIT_MISSING__"


def download_source(destination: str | Path) -> Path:
    """Reuse only a verified local source; never overwrite a different file."""
    destination = Path(destination)
    if destination.exists():
        if fingerprint(destination) != SOURCE_SHA256:
            raise ValueError("Existing source file does not match the pinned checksum.")
        return destination
    with urlopen(SOURCE_URL, timeout=60) as response:
        content = response.read(5_000_001)
    if len(content) > 5_000_000 or sha256(content).hexdigest() != SOURCE_SHA256:
        raise ValueError("Downloaded source does not match the pinned checksum.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(content)
    return destination


def read_table(path: str | Path, *, encoded: bool = False) -> pd.DataFrame:
    """Preserve lexical categories and reject invalid headers and ragged rows."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, strict=True))
    if len(rows) < 2:
        raise ValueError("A header and at least one data row are required.")
    header, records = rows[0], rows[1:]
    if len(header) != len(set(header)) or set(header) != set(COLUMNS):
        raise ValueError("Expected the exact Telco columns without duplicate headers.")
    if any(len(row) != len(header) for row in records):
        raise ValueError("CSV rows must have exactly as many cells as the header.")
    table = pd.DataFrame(records, columns=header).loc[:, COLUMNS]
    table = table.replace(r"^\s*$", np.nan, regex=True)
    ids = table[IDENTIFIER]
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("Customer identifiers must be nonmissing and unique.")
    mapping = {"0": 0, "1": 1} if encoded else {"No": 0, "Yes": 1}
    if table[TARGET].isna().any() or not table[TARGET].isin(mapping).all():
        raise ValueError("Unexpected or missing target label.")
    table[TARGET] = table[TARGET].map(mapping).astype("int64")
    if set(table[TARGET]) != {0, 1}:
        raise ValueError("Both target classes are required.")
    for column in NUMERIC:
        values = pd.to_numeric(table[column], errors="raise").astype("float64")
        if not np.isfinite(values.dropna()).all():
            raise ValueError(f"Nonfinite numerical value in {column}.")
        # Nonblank strings such as 'NaN' must not silently become missing.
        if (values.isna() & table[column].notna()).any():
            raise ValueError(f"Unparseable numerical value in {column}.")
        table[column] = values
    if table["Contract"].isna().any():
        raise ValueError("Contract is required for later segment comparisons.")
    if any(table[column].eq(MISSING_CATEGORY).any() for column in CATEGORICAL):
        raise ValueError("A category collides with the reserved missing-value token.")
    return table


def prepare_dataset(
    source: str | Path, destination: str | Path, *, seed: int = 42,
    reference_fraction: float = 0.2, expected_sha256: str = SOURCE_SHA256,
) -> dict:
    """Prepare an exclusive output directory; no fitting occurs before splitting.

    An alternate checksum is for explicit local fixtures. Such manifests do not
    claim IBM provenance. The development command always uses the pinned source.
    """
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a new preparation directory: {destination}")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32).")
    if isinstance(reference_fraction, bool) or not 0 < reference_fraction < 1:
        raise ValueError("Reference fraction must be strictly between zero and one.")
    checksum = fingerprint(source)
    if checksum != expected_sha256:
        raise ValueError("Source checksum mismatch; preparation stopped.")
    table = read_table(source).sort_values(IDENTIFIER).reset_index(drop=True)
    train, reference = train_test_split(
        table, test_size=reference_fraction, random_state=seed, stratify=table[TARGET]
    )
    partitions = {"train.csv": train, "reference.csv": reference}
    if any(set(part[TARGET]) != {0, 1} for part in partitions.values()):
        raise ValueError("Both classes must be represented in each partition.")
    audit = {
        "rows": len(table), "columns": len(table.columns),
        "class_counts": {str(key): int(value) for key, value in
                         table[TARGET].value_counts().sort_index().items()},
        "missing_by_column": {key: int(value) for key, value in table.isna().sum().items()},
        "duplicate_predictor_rows": int(table.duplicated(subset=FEATURES).sum()),
        "dtypes": {key: str(value) for key, value in table.dtypes.items()},
    }
    manifest = {
        "format_version": 1, "context": run_context(),
        "source": {"sha256": checksum, "bytes": source.stat().st_size,
                   "url": SOURCE_URL if checksum == SOURCE_SHA256 else None},
        "split": {"seed": seed, "reference_fraction": reference_fraction,
                  "stratified_by": TARGET, "sorted_by": IDENTIFIER},
        "files": {},
    }
    destination.mkdir(parents=True, exist_ok=False)
    for filename, part in partitions.items():
        part = part.sort_values(IDENTIFIER).reset_index(drop=True)
        with (destination / filename).open("x", encoding="utf-8", newline="") as stream:
            part.to_csv(stream, index=False, lineterminator="\n")
        manifest["files"][filename] = {
            "sha256": fingerprint(destination / filename),
            "rows": len(part), "positives": int(part[TARGET].sum()),
        }
    write_json(destination / "audit.json", audit)
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_partitions(directory: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify bytes and partition integrity before any model is fitted."""
    directory = Path(directory)
    with (directory / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict) or manifest.get("format_version") != 1:
        raise ValueError("Unsupported preparation manifest.")
    partitions = []
    try:
        for filename in ("train.csv", "reference.csv"):
            expected = manifest["files"][filename]
            if fingerprint(directory / filename) != expected["sha256"]:
                raise ValueError(f"Checksum mismatch for {filename}.")
            table = read_table(directory / filename, encoded=True)
            if len(table) != expected["rows"] or int(table[TARGET].sum()) != expected["positives"]:
                raise ValueError(f"Partition counts do not match {filename}.")
            partitions.append(table)
    except (KeyError, TypeError) as exc:
        raise ValueError("Preparation manifest is incomplete or malformed.") from exc
    train, reference = partitions
    if set(train[IDENTIFIER]) & set(reference[IDENTIFIER]):
        raise ValueError("Training and reference customer identifiers overlap.")
    return train, reference, manifest
