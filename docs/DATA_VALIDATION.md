# Phase 2: data validation

The original phase protocol below remains applicable to its saved-artifact API.
The October 30 scope corrections add the [raw fitted-model workflow](RAW_AUDITS.md);
see the [correction record](SCOPE_CORRECTIONS.md) for current capabilities.

## Acceptance criteria

- Inspect schema, missingness, duplicate rows and identifiers, cross-split overlap,
  and limited target-association heuristics without modifying inputs or fitting models.
- Return aggregate evidence with pass, warn, fail, or skipped for each check.
- Detect deliberately defective fixtures, preserve missing/undefined measurements
  as JSON null, and do not claim skipped checks passed.
- Integrate with the checksum-verified Phase 1 partitions; save provenance and
  strict JSON in a new report file without overwriting existing outputs.
- Run the complete regression suite and record real Telco results. Model metrics,
  calibration, segment analysis, drift, and the full validation CLI remain later phases.

## Development command

From the ModelAudit project directory, with Phase 1 data already prepared:

```powershell
.venv\Scripts\python scripts/audit_data.py
```

Defaults: `data/prepared` and `reports/generated/phase2-data-audit.json`.
Use `--prepared` and `--output` to select other paths. Each report must have a new
filename. Reports stay ignored by Git. Exit status is 0 for pass or review warnings,
1 for failed checks, and 2 for invalid input/artifacts or an existing output file.
This is a development entry point; the complete audit CLI is planned for Phase 6.

The command uses the existing Phase 1 loader, so malformed headers, invalid labels,
corrupt manifests/files, or overlapping IDs already rejected there fail before a
report is written. To investigate defective tables and collect multiple findings,
use the independent in-memory API below. It never downloads data or fits a model.

## Python interface

```python
from modelaudit.data_checks import CheckPolicy, DataSchema, audit_data

schema = DataSchema(
    numeric=("amount",), categorical=("contract",),
    target="churn", identifier="customer_id",
)
report = audit_data(train, reference, schema, CheckPolicy(missing_fraction=0.05))
```

Inputs are pandas DataFrames with unique, nonempty string column names. Schema
roles cannot overlap. Predictors must include at least one column. Numeric roles
accept real numerical pandas dtypes (including nullable numbers), not numeric
strings, booleans, or complex values. Categories and IDs contain strings; all-missing
columns have no observed values to type-check. Targets are numeric 0/1 with both
classes present in each partition. IDs are optional but, when declared, must be
nonmissing and unique within a partition and disjoint across partitions.
Whitespace-only strings are missing; other strings are preserved exactly.
Invalid API arguments raise TypeError/ValueError; invalid datasets produce findings.
Duplicate headers prevent unambiguous interpretation, so that table's other checks
are skipped while the other table is still inspected. No row values are emitted.

## Finding semantics

| Check | Failure or warning rule | Evidence |
| --- | --- | --- |
| Schema | Fail on missing/extra columns, incompatible values/dtypes, infinity | Column names and counts |
| Row count | Fail on an empty table | Row count |
| Missingness | Fail on any missing ID/target; warn when predictor fraction is strictly greater than 0.05 | Count, fraction, threshold |
| Target | Fail unless nonmissing, binary 0/1, and both classes occur | Invalid/missing counts, class count |
| Exact duplicate rows | Warn on repeats beyond first | Duplicate count |
| Predictor duplicates | Warn on repeats, excluding target and ID | Duplicate count |
| Duplicate identifiers | Fail on repeated nonmissing IDs | Duplicate count |
| ID overlap | Fail when reference IDs occur in training | Reference row count, distinct ID count |
| Predictor overlap | Warn when reference predictors exactly match training predictors | Reference row count and fraction |
| Leakage heuristics | Warn on target copies/inverses, high numerical correlation, supported deterministic categorical mapping | Signal names, paired support, correlation |

Predictor overlap matches missing cells with missing cells, excludes target/ID,
and counts each reference row once even if training has multiple matches. It is
not a duplicate-customer claim. Equal predictors can be legitimate. Duplicates
are counted after blank normalization. Column order and DataFrame indexes do not
serve as identifiers. Checks requiring invalid/missing columns or empty data are
explicitly skipped. Empty denominators and undefined correlations use JSON null.
The overall status prioritizes fail, then warn, then pass; the separate skipped
count must also be reviewed. An overall pass is not deployment approval.

## Leakage limits

- Exact numerical target copies and inverses are flagged with at least two paired
  observations containing both target classes, even on a small dataset.
- Numerical association uses absolute Pearson correlation of at least 0.98,
  after dropping missing predictor/target pairs. It needs at least 20 paired rows
  and variation in both columns. Values are scaled to avoid overflow.
- A categorical predictor is flagged when each category maps to one target class,
  with 2–20 categories and at least five observations per category, using at least
  20 paired rows. Unique customer-like values cannot qualify by memorization alone.
- Constant predictors, insufficient paired/category support, excessive category
  counts, lost numerical precision, or an invalid target skip association analysis. A direct target-copy warning may still apply.
- These rules are heuristics. Useful predictors may trigger them. Absence of a
  warning cannot establish absence of leakage; nonlinear, multicolumn, temporal,
  provenance and post-outcome leakage are outside this phase. High-cardinality
  category mappings are not exhaustively assessed.

`CheckPolicy` exposes these thresholds for the Python API. This phase deliberately
adds no full YAML audit configuration, automatic repair, model acceptance decision,
or deployment behavior. Prepared-input hashes, runtime, actual time, and current
Git state accompany development-command reports. The reports contain aggregate
findings, not raw records. Keep generated artifacts local.
