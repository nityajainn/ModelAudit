"""Strict CSV snapshot adapter with integrity checks and exclusive reports."""

import csv
from pathlib import Path

import pandas as pd

from .artifacts import fingerprint, run_context, write_json
from .drift import _roles, compare_snapshots


def _read_snapshot(path, numeric, categorical, probability, target):
    required = _roles(numeric, categorical, probability, target)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if (not header or len(set(header)) != len(header) or not required <= set(header)
                or set(header) - required - ({target} if target is not None else set())):
            raise ValueError("Snapshot CSV header does not match declared unique roles.")
        rows = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError("Snapshot CSV has a row with the wrong number of fields.")
            rows.append(row)
    if not rows:
        raise ValueError("Snapshot CSV must contain observations.")
    frame = pd.DataFrame(rows, columns=header)
    numbers = list(numeric) + ([probability] if probability is not None else [])
    numbers += [target] if target is not None and target in frame else []
    for name in numbers:
        # Preserve exact integers until the core numeric validator has checked them.
        parsed = []
        for value in frame[name]:
            if not value.strip():
                parsed.append(float("nan"))
            else:
                try:
                    parsed.append(int(value))
                except ValueError:
                    try:
                        parsed.append(float(value))
                    except ValueError as exc:
                        raise ValueError(f"Column {name} must contain numeric CSV values.") from exc
        # object preserves large integers in feature columns for precision checks.
        frame[name] = pd.Series(parsed, dtype=object if name in numeric else None)
    return frame


def audit_snapshot_files(reference, current, output, *, numeric=(), categorical=(),
                         probability=None, target=None, **options):
    """No model loading; labels and probabilities must share snapshot row order."""
    reference, current, output = Path(reference), Path(current), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    paths = {"reference": reference, "current": current}
    before = {name: fingerprint(path) for name, path in paths.items()}
    frames = [_read_snapshot(path, numeric, categorical, probability, target)
              for path in (reference, current)]
    comparison = compare_snapshots(*frames, numeric=numeric, categorical=categorical,
                                   probability=probability, target=target, **options)
    if before != {name: fingerprint(path) for name, path in paths.items()}:
        raise ValueError("An input changed during comparison; no report was written.")
    result = {"format_version": 1, "context": run_context(), "inputs": before,
              "model_deserialization_performed": False, "comparison": comparison}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
