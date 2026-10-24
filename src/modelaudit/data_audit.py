"""Run Phase 2 checks on the verified Phase 1 prepared partitions."""

from pathlib import Path

from .artifacts import fingerprint, run_context, write_json
from .data_checks import CheckPolicy, DataSchema, audit_data
from .dataset import CATEGORICAL, IDENTIFIER, NUMERIC, TARGET, load_partitions

TELCO_SCHEMA = DataSchema(tuple(NUMERIC), tuple(CATEGORICAL), TARGET, IDENTIFIER)


def audit_prepared(directory: str | Path, output: str | Path,
                   *, policy: CheckPolicy | None = None) -> dict:
    """Reject corrupt Phase 1 artifacts and never overwrite an existing report."""
    directory, output = Path(directory), Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    files = [directory / name for name in ("manifest.json", "train.csv", "reference.csv")]
    before = {path.name: fingerprint(path) for path in files}
    train, reference, _ = load_partitions(directory)
    result = audit_data(train, reference, TELCO_SCHEMA, policy)
    after = {path.name: fingerprint(path) for path in files}
    if before != after:
        raise ValueError("Prepared input changed during the audit; no report was written.")
    result["context"] = run_context()
    result["inputs"] = before
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result
